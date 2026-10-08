"""Phase 4 — validate on labeled data (leakage-controlled), then score VUS.

Rigor:
  * The 3D-PPS is RECOMPUTED inside every CV fold using ONLY training-fold labels, so
    test-set pathogenicity never leaks into a variant's structural feature. This is the
    spatial-leakage control that a naive "compute once, then CV" pipeline gets wrong.
  * Ablation across feature sets isolates the value the 3D axis adds over sequence tools
    and over a matched 1D (sequence-distance) control.
  * Anti-circularity variant excludes supervised predictors trained on ClinVar-like labels
    (REVEL, ClinPred, MetaLR, VEST4), keeping population/unsupervised signals + structure.
  * Metrics: AUROC, AUPRC (primary, given imbalance), precision@k, Brier calibration.

Outputs: results/tables/cv_metrics.csv, results/tables/oof_predictions.csv,
         data/processed/vus_scored.parquet (Reclassification Confidence Score per VUS).

Run:  python -m src.model
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from src.structure import load_ca, PPSCalculator

OUT = Path("data/processed")
TAB = Path("results/tables"); TAB.mkdir(parents=True, exist_ok=True)
SEED = 42

# Axis A predictor panel
PRED_ALL = ["sift", "sift4g", "polyphen2_hdiv", "polyphen2_hvar", "revel",
            "alphamissense", "metalr", "clinpred", "primateai", "vest4",
            "provean", "mpc", "esm1b", "cadd_phred", "phylop100"]
# ClinVar-trained supervised predictors to drop for the anti-circularity model
CLINVAR_TRAINED = ["revel", "clinpred", "metalr", "vest4"]
PRED_POP = [c for c in PRED_ALL if c not in CLINVAR_TRAINED]
STRUCT_CTX = ["coordination", "plddt"]   # label-free structural context (no leakage)


def _pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])


def precision_at_k(y, p, k):
    order = np.argsort(-p)
    topk = order[:k]
    return float(np.mean(np.asarray(y)[topk]))


def fold_pps(ca, train_df, score_df, **kw):
    """Compute 3D-PPS + 1D control for score_df rows using TRAIN-fold labels only."""
    path_res = set(train_df.loc[train_df.label.eq("P"), "aapos"].astype(int))
    ben_res = set(train_df.loc[train_df.label.eq("B"), "aapos"].astype(int))
    calc = PPSCalculator(ca, path_res, ben_res, **kw)
    rows = []
    for _, r in score_df.iterrows():
        rn = int(r["aapos"])
        sc = calc.score(rn, exclude={rn})   # always exclude own residue
        rows.append({"hgvs_g": r["hgvs_g"], "pps3d_norm": sc["pps3d_norm"],
                     "pps3d": sc["pps3d"], "pps1d_norm": sc["pps1d_norm"]})
    return pd.DataFrame(rows).set_index("hgvs_g")


def build_X(df, cols, pps=None, use_3d=False, use_1d=False):
    X = df[cols].copy()
    if pps is not None:
        p = pps.reindex(df["hgvs_g"].values)
        if use_3d:
            X["pps3d_norm"] = p["pps3d_norm"].values
        if use_1d:
            X["pps1d_norm"] = p["pps1d_norm"].values
    return X


def cross_validate(lab, ca, feature_set: str):
    """feature_set in {A_only, A_3d, A_1d, threeD_only, pop_3d}."""
    y = lab["label"].eq("P").astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(lab))
    for tr, te in skf.split(lab, y):
        train_df, test_df = lab.iloc[tr], lab.iloc[te]
        pps = None
        if feature_set in ("A_3d", "A_1d", "threeD_only", "pop_3d"):
            pps = fold_pps(ca, train_df, lab)   # score all; only train labels used
        if feature_set == "A_only":
            cols = PRED_ALL + STRUCT_CTX; u3 = u1 = False
        elif feature_set == "A_3d":
            cols = PRED_ALL + STRUCT_CTX; u3, u1 = True, False
        elif feature_set == "A_1d":
            cols = PRED_ALL + STRUCT_CTX; u3, u1 = False, True
        elif feature_set == "threeD_only":
            cols = STRUCT_CTX; u3, u1 = True, False
        elif feature_set == "pop_3d":
            cols = PRED_POP + STRUCT_CTX; u3, u1 = True, False
        Xtr = build_X(train_df, cols, pps, u3, u1)
        Xte = build_X(test_df, cols, pps, u3, u1)
        pipe = _pipe().fit(Xtr, y[tr])
        oof[te] = pipe.predict_proba(Xte)[:, 1]
    return {
        "feature_set": feature_set,
        "auroc": round(roc_auc_score(y, oof), 4),
        "auprc": round(average_precision_score(y, oof), 4),
        "prec@50": round(precision_at_k(y, oof, 50), 3),
        "prec@100": round(precision_at_k(y, oof, 100), 3),
        "brier": round(brier_score_loss(y, oof), 4),
    }, oof, y


def univariate_aucs(lab):
    """Univariate AUROC of each feature (incl. full-labeled 3D-PPS) for context."""
    y = lab["label"].eq("P").astype(int).values
    rows = []
    for c in PRED_ALL + ["pps3d_norm", "pps1d_norm", "coordination", "plddt", "gnomad_af"]:
        v = pd.to_numeric(lab[c], errors="coerce")
        m = v.notna()
        if m.sum() < 50 or y[m.values].sum() == 0:
            continue
        a = roc_auc_score(y[m.values], v[m].values)
        rows.append({"feature": c, "auroc": round(max(a, 1 - a), 3),
                     "direction": "+" if a >= 0.5 else "-", "n": int(m.sum())})
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


def main():
    df = pd.read_parquet(OUT / "features.parquet")
    ca = load_ca()
    lab = df[df["label"].isin(["P", "B"])].reset_index(drop=True)
    print(f"[model] labeled: P={int(lab.label.eq('P').sum())} B={int(lab.label.eq('B').sum())}")

    # univariate context
    uni = univariate_aucs(lab)
    uni.to_csv(TAB / "univariate_aucs.csv", index=False)
    print("[model] top univariate AUROCs:\n", uni.head(10).to_string(index=False))

    # leakage-controlled CV across feature sets
    results, oofs = [], {}
    for fs in ["A_only", "A_1d", "A_3d", "threeD_only", "pop_3d"]:
        m, oof, y = cross_validate(lab, ca, fs)
        results.append(m); oofs[fs] = oof
        print(f"[model] {fs:11s} AUROC={m['auroc']} AUPRC={m['auprc']} "
              f"P@50={m['prec@50']} P@100={m['prec@100']} Brier={m['brier']}")
    resdf = pd.DataFrame(results)
    resdf.to_csv(TAB / "cv_metrics.csv", index=False)
    oofdf = pd.DataFrame({"hgvs_g": lab["hgvs_g"], "y": lab["label"].eq("P").astype(int),
                          **{f"oof_{k}": v for k, v in oofs.items()}})
    oofdf.to_csv(TAB / "oof_predictions.csv", index=False)

    # ---- final model on ALL labeled (full-labeled 3D-PPS already in features.parquet) ----
    y = lab["label"].eq("P").astype(int).values
    cols = PRED_ALL + STRUCT_CTX
    Xtr = lab[cols].copy(); Xtr["pps3d_norm"] = lab["pps3d_norm"]
    final = _pipe().fit(Xtr, y)

    vus = df[df["label"].eq("VUS")].reset_index(drop=True)
    Xv = vus[cols].copy(); Xv["pps3d_norm"] = vus["pps3d_norm"]
    vus = vus.copy()
    vus["RCS"] = final.predict_proba(Xv)[:, 1]   # Reclassification Confidence Score
    vus.sort_values("RCS", ascending=False, inplace=True)
    vus.to_parquet(OUT / "vus_scored.parquet", index=False)
    vus.to_csv(OUT / "vus_scored.csv", index=False)

    coef = pd.Series(final.named_steps["clf"].coef_[0], index=Xtr.columns).sort_values()
    coef.to_csv(TAB / "final_model_coefficients.csv")
    print("\n[model] final-model coefficients (standardized):\n",
          coef.round(3).to_string())
    print(f"\n[model] VUS scored: {len(vus)} | RCS>0.9: {(vus.RCS>0.9).sum()} "
          f"| RCS>0.5: {(vus.RCS>0.5).sum()} | RCS<0.1: {(vus.RCS<0.1).sum()}")
    print(f"[model] saved -> {OUT/'vus_scored.parquet'}, {TAB/'cv_metrics.csv'}")


if __name__ == "__main__":
    main()
