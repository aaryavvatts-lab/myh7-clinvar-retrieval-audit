"""Phase 4b — statistical rigor: cluster-bootstrap CIs, paired tests, equivalence (TOST),
calibration, and the structural spatial-leakage inflation analysis.

Motivated by leakage-free evaluation practice in variant-effect benchmarking: predictors
are correlated across variants that share a residue, so significance uses a CLUSTER
bootstrap that resamples residue positions, not individual variants. We also quantify the
inflation introduced when the 3D structural feature is computed once on all labels
(spatial leakage) versus recomputed inside each CV fold (leakage-free) — the structural
analogue of the circular/non-circular distinction.

Outputs: results/tables/bootstrap_cis.csv, paired_tests.csv, calibration.csv,
         results/tables/leakage_inflation.csv; results/figures/fig9_calibration.png
Run:  python -m src.stats
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import calibration_curve
from src.model import PRED_ALL, STRUCT_CTX, univariate_aucs

OUT = Path("data/processed")
TAB = Path("results/tables")
FIG = Path("results/figures")
SEED = 42
N_BOOT = 3000


def cluster_bootstrap_auroc(y, score, groups, n=N_BOOT, seed=SEED):
    """AUROC 95% CI by resampling unique residue positions (clusters)."""
    rng = np.random.default_rng(seed)
    y, score, groups = np.asarray(y), np.asarray(score), np.asarray(groups)
    uniq = np.unique(groups)
    pos_index = {g: np.where(groups == g)[0] for g in uniq}
    aucs = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pos_index[g] for g in samp])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, score[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lo), float(hi)


def paired_diff(y, sA, sB, groups, n=N_BOOT, seed=SEED):
    """Cluster-bootstrap CI of AUROC(A) - AUROC(B); TOST-style equivalence readout."""
    rng = np.random.default_rng(seed)
    y, sA, sB, groups = map(np.asarray, (y, sA, sB, groups))
    uniq = np.unique(groups)
    pos_index = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pos_index[g] for g in samp])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(roc_auc_score(yy, sA[idx]) - roc_auc_score(yy, sB[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    df = pd.read_parquet(OUT / "features.parquet")
    lab = df[df.label.isin(["P", "B"])].reset_index(drop=True)
    y = lab["label"].eq("P").astype(int).values
    groups = lab["aapos"].astype(int).values
    oof = pd.read_csv(TAB / "oof_predictions.csv").set_index("hgvs_g").reindex(lab["hgvs_g"].values)

    # ---- (1) cluster-bootstrap AUROC CIs for key single features + models ----
    rows = []
    feat_scores = {c: lab[c].values for c in ["alphamissense", "revel", "cadd_phred",
                   "esm1b", "sift", "polyphen2_hvar", "pps3d_norm", "pps1d_norm"]}
    # orient every score so higher = more damaging (SIFT, ESM1b are inverted in dbNSFP)
    feat_scores["sift"] = -pd.to_numeric(lab["sift"], errors="coerce").values
    feat_scores["esm1b"] = -pd.to_numeric(lab["esm1b"], errors="coerce").values
    feat_scores["polyphen2_hvar"] = pd.to_numeric(lab["polyphen2_hvar"], errors="coerce").values
    model_scores = {"model_A_only": oof["oof_A_only"].values,
                    "model_A_3d": oof["oof_A_3d"].values,
                    "model_pop_3d": oof["oof_pop_3d"].values,
                    "model_structure_only": oof["oof_threeD_only"].values}
    for name, s in {**feat_scores, **model_scores}.items():
        s = pd.to_numeric(pd.Series(s), errors="coerce").values
        m = ~np.isnan(s)
        mean, lo, hi = cluster_bootstrap_auroc(y[m], s[m], groups[m])
        rows.append({"predictor": name, "auroc": round(mean, 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3), "n": int(m.sum())})
    cis = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    cis.to_csv(TAB / "bootstrap_cis.csv", index=False)
    print("[stats] cluster-bootstrap AUROC (95% CI):")
    print(cis.to_string(index=False))

    # ---- (2) paired tests (cluster bootstrap) ----
    pairs = [("model_A_3d", "model_A_only", "Does adding 3D-PPS help vs sequence-only?"),
             ("pps3d_norm", "revel", "3D-PPS vs REVEL (single features)"),
             ("model_A_3d", "alphamissense", "Ensemble vs best single predictor (equivalence?)"),
             ("model_pop_3d", "model_A_only", "ClinVar-free+3D vs full sequence panel")]
    allscores = {**{k: pd.to_numeric(pd.Series(v), errors="coerce").values for k, v in feat_scores.items()},
                 **model_scores}
    prows = []
    for a, b, q in pairs:
        sA, sB = allscores[a], allscores[b]
        m = ~(np.isnan(sA) | np.isnan(sB))
        d, lo, hi = paired_diff(y[m], sA[m], sB[m], groups[m])
        equiv = (lo > -0.05) and (hi < 0.05)
        sig = (lo > 0) or (hi < 0)
        prows.append({"comparison": f"{a} - {b}", "question": q, "delta_auroc": round(d, 3),
                      "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                      "significant": sig, "equivalent_±0.05": equiv})
    pt = pd.DataFrame(prows)
    pt.to_csv(TAB / "paired_tests.csv", index=False)
    print("\n[stats] paired cluster-bootstrap AUROC differences:")
    print(pt.to_string(index=False))

    # ---- (3) spatial-leakage inflation: naive global 3D-PPS vs leakage-free per-fold ----
    def _pipe():
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    cols = PRED_ALL + STRUCT_CTX
    Xnaive = lab[cols].copy(); Xnaive["pps3d_norm"] = lab["pps3d_norm"]  # global (leaky) PPS
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    naive_oof = cross_val_predict(_pipe(), Xnaive, y, cv=skf, method="predict_proba")[:, 1]
    auc_naive = roc_auc_score(y, naive_oof)
    auc_free = roc_auc_score(y, oof["oof_A_3d"].values)
    infl = pd.DataFrame([{"eval": "naive global 3D-PPS (spatial leak)", "auroc": round(auc_naive, 4)},
                         {"eval": "leakage-free per-fold 3D-PPS", "auroc": round(auc_free, 4)},
                         {"eval": "inflation (Δ)", "auroc": round(auc_naive - auc_free, 4)}])
    infl.to_csv(TAB / "leakage_inflation.csv", index=False)
    print("\n[stats] spatial-leakage inflation of the structural feature:")
    print(infl.to_string(index=False))

    # ---- (4) calibration ----
    p = oof["oof_A_3d"].values
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    cal = pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})
    cal.to_csv(TAB / "calibration.csv", index=False)
    brier = brier_score_loss(y, p)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot([0, 1], [0, 1], "--", color="#9aa0a6", label="perfect")
    ax.plot(mean_pred, frac_pos, "-o", color="#c1435a", label=f"model (Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed pathogenic fraction")
    ax.set_title("Figure 9. Calibration of the leakage-free model")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig9_calibration.png", dpi=300); plt.close(fig)
    print(f"\n[stats] Brier (leakage-free A_3d): {brier:.3f}; figures -> fig9_calibration.png")
    print("[stats] saved bootstrap_cis, paired_tests, leakage_inflation, calibration")


if __name__ == "__main__":
    main()
