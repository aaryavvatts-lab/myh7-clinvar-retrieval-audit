"""v6 models: leakage-controlled CV (3D-PPS recomputed per fold, exact with abstention),
ablations with coverage controls, gradient boosting, paired comparisons, calibration,
final model + VUS scoring.

Feature handling for the LOGISTIC model: an undefined 3D/1D score is filled with 0 AND an
explicit `*_defined` indicator is added (so the model can see abstention rather than have
it silently equated with "balanced evidence"). Gradient boosting receives NaN natively.

Outputs (results/tables): v6_cv_metrics_<set>.csv, v6_oof_<set>.csv, v6_model_features.json,
v6_paired_models.csv, v6_calibration_<set>.csv, v6_calibration_summary.csv, v6_leakage.csv,
final_model_coefficients_v6.csv; data/processed/vus_scored_model.parquet
Run: python -m src.model_v6
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from src.common import *   # noqa
from src.core3d import Kernels
from src.build_features import kernels

SETS_LR = ["A_only", "A_cov", "A_1d", "A_3d", "pop_only", "pop_3d", "threeD_only", "cov_only"]
SETS_GB = ["A_only", "A_cov", "A_3d"]
BASE = {"A": PRED_ALL + STRUCT_CTX, "pop": PRED_POP + STRUCT_CTX, "ctx": STRUCT_CTX}


def feature_names(fs):
    base = BASE["pop"] if fs.startswith("pop") else (BASE["ctx"] if fs in ("threeD_only", "cov_only") else BASE["A"])
    extra = []
    if fs in ("A_3d", "pop_3d", "threeD_only"): extra += ["pps3d_norm", "pps3d_defined"]
    if fs == "A_1d": extra += ["pps1d_norm", "pps1d_defined"]
    if fs in ("A_cov", "cov_only"): extra += ["n_nb", "covered_ge3"]
    return base + extra


def _lr():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))])


def _lr_unweighted():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, C=1.0))])


def _gb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          class_weight="balanced", random_state=SEED)


def design(lab, fs, K, train_idx, gb=False):
    """Feature matrix for ALL rows in `lab`, with structural features from TRAIN labels only."""
    tr = lab.iloc[train_idx]
    pm, bm = K.masks(tr.loc[tr.lab.eq("P"), "aapos"], tr.loc[tr.lab.eq("B"), "aapos"])
    sc = K.score(lab.aapos.values, pm, bm)
    X = lab[feature_names(fs)[:len(BASE['pop' if fs.startswith('pop') else ('ctx' if fs in ('threeD_only','cov_only') else 'A')])]].copy()
    for k in feature_names(fs)[len(X.columns):]:
        if k == "pps3d_norm": X[k] = sc["pps3d_norm"]
        elif k == "pps1d_norm": X[k] = sc["pps1d_norm"]
        elif k == "pps3d_defined": X[k] = (~np.isnan(sc["pps3d_norm"])).astype(float)
        elif k == "pps1d_defined": X[k] = (~np.isnan(sc["pps1d_norm"])).astype(float)
        elif k == "n_nb": X[k] = sc["n_nb"]
        elif k == "covered_ge3": X[k] = (sc["n_nb"] >= 3).astype(float)
    if not gb:   # logistic: abstention -> 0 plus indicator (indicator already present)
        for k in ("pps3d_norm", "pps1d_norm"):
            if k in X: X[k] = X[k].fillna(0.0)
    return X, sc


def cv_oof(lab, fs, K, kind="lr", seed=SEED):
    y = lab.y.values
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(lab))
    for tr, te in skf.split(lab, y):
        X, _ = design(lab, fs, K, tr, gb=(kind == "gb"))
        mdl = {"lr": _lr, "lr_unw": _lr_unweighted, "gb": _gb}[kind]().fit(X.iloc[tr], y[tr])
        oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]
    return oof


def metrics(lab, oof):
    y, g = lab.y.values, lab.aapos.values
    a, lo, hi, _ = auroc_ci(y, oof, g)
    return {"auroc": r3(a), "ci_low": r3(lo), "ci_high": r3(hi), "auprc": r3(average_precision_score(y, oof)),
            "auprc_baseline_prevalence": r3(y.mean()), "brier": r3(brier_score_loss(y, oof)),
            "brier_constant_prevalence": r3(y.mean() * (1 - y.mean()))}


def calibration(y, p):
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    cnt, _ = np.histogram(p, bins=np.quantile(p, np.linspace(0, 1, 11)))
    tab = pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac})
    pc = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(pc / (1 - pc))
    fit = sm.Logit(y, sm.add_constant(lg)).fit(disp=0)
    # intercept with slope fixed at 1 (calibration-in-the-large)
    off = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lg).fit()
    return tab, {"cal_slope": r3(fit.params[1]), "cal_slope_ci_low": r3(fit.conf_int()[1][0]),
                 "cal_slope_ci_high": r3(fit.conf_int()[1][1]), "cal_intercept_free": r3(fit.params[0]),
                 "cal_in_the_large": r3(off.params[0]), "brier": r3(brier_score_loss(y, p)),
                 "brier_constant_prevalence": r3(y.mean() * (1 - y.mean())),
                 "mean_pred": r3(p.mean()), "observed_prevalence": r3(y.mean())}


def main():
    df = load(); K = kernels()
    feats = {fs: feature_names(fs) for fs in SETS_LR}
    json.dump({"logistic_feature_sets": feats, "gradient_boosting_feature_sets": {k: feats[k] for k in SETS_GB},
               "predictor_panel_PRED_ALL": PRED_ALL, "clinvar_trained_dropped_in_pop": CLINVAR_TRAINED,
               "notes": "logistic: undefined 3D/1D score -> 0 plus *_defined indicator; GB: NaN native. "
                        "Structural features recomputed in each fold from training labels only."},
              open(TAB / "v6_model_features.json", "w"), indent=1)
    summ, paired_rows, cal_sum = [], [], []
    for setname in SETS:
        lab = labelled(df, setname)
        oofs = {}
        rows = []
        for fs in SETS_LR:
            oofs[("lr", fs)] = cv_oof(lab, fs, K, "lr")
            rows.append({"set": setname, "model": "logistic", "feature_set": fs, **metrics(lab, oofs[("lr", fs)])})
        for fs in SETS_GB:
            oofs[("gb", fs)] = cv_oof(lab, fs, K, "gb")
            rows.append({"set": setname, "model": "gradient_boosting", "feature_set": fs, **metrics(lab, oofs[("gb", fs)])})
        cvdf = pd.DataFrame(rows); cvdf.to_csv(TAB / f"v6_cv_metrics_{setname}.csv", index=False)
        pd.DataFrame({"hgvs_g": lab.hgvs_g, "y": lab.y, **{f"oof_{k[0]}_{k[1]}": v for k, v in oofs.items()}}
                     ).to_csv(TAB / f"v6_oof_{setname}.csv", index=False)
        summ.append(cvdf)
        y, g = lab.y.values, lab.aapos.values
        for a, b, q in [(("lr", "A_3d"), ("lr", "A_only"), "adding 3D-PPS (+indicator) to the full panel"),
                        (("lr", "A_cov"), ("lr", "A_only"), "adding coverage count only (no score value)"),
                        (("lr", "A_3d"), ("lr", "A_cov"), "3D-PPS value beyond coverage count"),
                        (("lr", "A_3d"), ("lr", "A_1d"), "3D vs 1D score added to the panel"),
                        (("lr", "pop_3d"), ("lr", "A_3d"), "DROPPING ClinVar-trained predictors (both with 3D)"),
                        (("lr", "pop_only"), ("lr", "A_only"), "DROPPING ClinVar-trained predictors (no 3D)"),
                        (("lr", "pop_3d"), ("lr", "pop_only"), "adding 3D-PPS to ClinVar-free panel"),
                        (("gb", "A_3d"), ("gb", "A_only"), "GB: adding 3D-PPS"),
                        (("gb", "A_3d"), ("lr", "A_3d"), "GB vs logistic, both with 3D")]:
            r = paired_diff(y, oofs[a], oofs[b], g)
            paired_rows.append({"set": setname, "A": "/".join(a), "B": "/".join(b), "question": q,
                                **{k: (r3(v) if isinstance(v, float) else v) for k, v in r.items()}})
        # leakage: global 3D-PPS (own residue excluded, all labels) vs fold-wise
        allidx = np.arange(len(lab))
        Xg, _ = design(lab, "A_3d", K, allidx)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        naive = np.zeros(len(lab))
        for tr, te in skf.split(lab, y):
            naive[te] = _lr().fit(Xg.iloc[tr], y[tr]).predict_proba(Xg.iloc[te])[:, 1]
        la, lo, hi, _ = auroc_ci(y, naive, g); fa = metrics(lab, oofs[("lr", "A_3d")])["auroc"]
        pd.DataFrame([{"set": setname, "eval": "naive global 3D-PPS", "auroc": r3(la)},
                      {"set": setname, "eval": "fold-wise 3D-PPS", "auroc": fa},
                      {"set": setname, "eval": "inflation", "auroc": r3(la - fa)}]
                     ).to_csv(TAB / f"v6_leakage_{setname}.csv", index=False)
        # calibration (A_3d logistic, balanced weights; and unweighted)
        for nm, p in [("balanced", oofs[("lr", "A_3d")]), ("unweighted", cv_oof(lab, "A_3d", K, "lr_unw"))]:
            tab, cs = calibration(y, p)
            tab.to_csv(TAB / f"v6_calibration_{setname}_{nm}.csv", index=False)
            cal_sum.append({"set": setname, "weights": nm, **cs})
        # final model on ALL labelled (wider set only is used for VUS scoring)
        if setname == "wider":
            X, sc = design(lab, "A_3d", K, allidx)
            fin = _lr().fit(X, y)
            coef = pd.Series(fin.named_steps["clf"].coef_[0], index=X.columns).sort_values()
            coef.to_csv(TAB / "final_model_coefficients_v6.csv")
            vus = df[df.label.eq("VUS")].reset_index(drop=True).copy()
            vus["lab"] = vus["label"]
            from src.build_features import add_scores
            full = df.copy(); full["lab"] = full["label"]
            full = add_scores(full, "lab")
            vus = full[full.label.eq("VUS")].reset_index(drop=True)
            Xv = vus[feature_names("A_3d")[:len(BASE["A"])]].copy()
            Xv["pps3d_norm"] = vus.pps3d_norm.fillna(0.0); Xv["pps3d_defined"] = vus.pps3d_norm.notna().astype(float)
            vus["RCS"] = fin.predict_proba(Xv[X.columns])[:, 1]
            vus.to_parquet("data/processed/vus_scored_model.parquet", index=False)
    allm = pd.concat(summ); allm.to_csv(TAB / "v6_cv_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(TAB / "v6_paired_models.csv", index=False)
    pd.DataFrame(cal_sum).to_csv(TAB / "v6_calibration_summary.csv", index=False)
    print(allm.to_string()); print(pd.DataFrame(paired_rows).to_string()); print(pd.DataFrame(cal_sum).to_string())


if __name__ == "__main__":
    main()
