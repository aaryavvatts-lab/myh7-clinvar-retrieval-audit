"""Corrected re-analysis (v2) addressing four flaws found in internal peer review.

FLAW A  Ascertainment/support confound. 3D-PPS is UNDEFINED where a residue has no
        labelled 3D neighbour; the original code silently set it to 0, conflating
        "no information" with "balanced evidence". Mere presence of a labelled neighbour
        alone gives AUROC 0.65. FIX: define an INFORMATIVE mask (>=k labelled non-local
        neighbours), report the score only there (abstention), and report coverage.
FLAW B  Structural orphans were supported by a single distant neighbour in 17/22 cases.
        FIX: require >=3 distinct non-local pathogenic neighbours.
FLAW C  Discordance % is threshold-dependent (46-62%) and the denominator was ambiguous.
        FIX: report a range over calibrated thresholds with explicit denominator.
FLAW D  Benign proxies at AF 1e-5..5e-5 sit in the "low-penetrance sarcomere variant"
        zone that remains HCM-enriched (Meisner 2024, Circulation). FIX: primary benign
        threshold AF >= 5e-5; 1e-5 retained as a sensitivity analysis.

Run: python -m src.reanalysis
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.structure import load_ca, PPSCalculator
from src.model import PRED_ALL, STRUCT_CTX, CLINVAR_TRAINED
from src.stats import cluster_bootstrap_auroc, paired_diff

OUT = Path("data/processed"); TAB = Path("results/tables")
SEED = 42
MIN_SUPPORT = 3          # >=3 labelled non-local neighbours to call the score informative
BENIGN_AF_PRIMARY = 5e-5  # Meisner 2024 / ClinGen-informed
BENIGN_AF_SENS = 1e-5


def neighbour_support(df, ca, radius=10.0, win=4):
    """Count distinct labelled (P or B) non-local neighbours per variant residue."""
    resn = np.array(sorted(ca)); xyz = np.array([ca[i][0] for i in resn])
    tree = cKDTree(xyz); idx = {int(r): k for k, r in enumerate(resn)}
    pres = set(df.loc[df.label.eq("P"), "aapos"].astype(int))
    bres = set(df.loc[df.label.eq("B"), "aapos"].astype(int))
    npath, nben = [], []
    for r in df["aapos"].astype(int):
        if r not in idx:
            npath.append(0); nben.append(0); continue
        nb = tree.query_ball_point(xyz[idx[r]], radius)
        p = b = 0
        for j in nb:
            q = int(resn[j])
            if q == r or abs(q - r) <= win:
                continue
            if q in pres: p += 1
            if q in bres: b += 1
        npath.append(p); nben.append(b)
    return np.array(npath), np.array(nben)


def relabel(df, af_thr):
    """Recompute benign proxies at a given AF threshold."""
    d = df.copy()
    is_p = d["clinsig"].astype(str).str.lower().str.contains("pathogenic") & \
           ~d["clinsig"].astype(str).str.lower().str.contains("benign|conflicting|uncertain")
    is_cb = d["benign_source"].eq("clinvar")
    is_vus = d["label"].eq("VUS")
    af = d["gnomad_af"].fillna(0.0)
    lab = pd.Series(pd.NA, index=d.index, dtype="object")
    lab[is_vus] = "VUS"
    lab[(af >= af_thr) & ~is_p & ~is_vus] = "B"
    lab[is_cb] = "B"
    lab[d["label"].eq("P")] = "P"
    d["label"] = lab
    return d


def _pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def cv_auc(lab, cols, use_pps=False, ca=None):
    """Leakage-free CV: 3D-PPS recomputed per fold from training labels only."""
    y = lab["label"].eq("P").astype(int).values
    oof = np.zeros(len(lab))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(lab, y):
        train = lab.iloc[tr]
        X = lab[cols].copy()
        if use_pps:
            pr = set(train.loc[train.label.eq("P"), "aapos"].astype(int))
            br = set(train.loc[train.label.eq("B"), "aapos"].astype(int))
            calc = PPSCalculator(ca, pr, br)
            X["pps3d_norm"] = [calc.score(int(r), exclude={int(r)})["pps3d_norm"]
                               for r in lab["aapos"]]
        pipe = _pipe().fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y, oof), average_precision_score(y, oof), oof, y


def main():
    df0 = pd.read_parquet(OUT / "features.parquet")
    ca = load_ca()
    rows = []

    for name, thr in [("primary (AF>=5e-5)", BENIGN_AF_PRIMARY),
                      ("sensitivity (AF>=1e-5)", BENIGN_AF_SENS)]:
        d = relabel(df0, thr)
        lab = d[d.label.isin(["P", "B"])].reset_index(drop=True)
        rows.append({"labelset": name, "n_P": int(lab.label.eq("P").sum()),
                     "n_B": int(lab.label.eq("B").sum())})
        print(f"[relabel] {name}: P={int(lab.label.eq('P').sum())} B={int(lab.label.eq('B').sum())}")
    pd.DataFrame(rows).to_csv(TAB / "v2_labelsets.csv", index=False)

    # ---- primary label set ----
    d = relabel(df0, BENIGN_AF_PRIMARY)
    lab = d[d.label.isin(["P", "B"])].reset_index(drop=True)
    npath, nben = neighbour_support(lab, ca)
    lab["n_nb"] = npath + nben
    lab["informative"] = lab["n_nb"] >= MIN_SUPPORT
    y = lab["label"].eq("P").astype(int).values
    grp = lab["aapos"].astype(int).values

    print(f"\n[coverage] informative (>= {MIN_SUPPORT} labelled 3D neighbours): "
          f"{lab['informative'].mean()*100:.1f}% of labelled variants "
          f"(P {lab.loc[y==1,'informative'].mean()*100:.0f}%, B {lab.loc[y==0,'informative'].mean()*100:.0f}%)")
    print(f"[confound] AUROC of mere-presence-of-neighbour: "
          f"{roc_auc_score(y,(lab['n_nb']>0).astype(int)):.3f}  <-- ascertainment signal")

    # ---- Table: full vs informative subset, with cluster-bootstrap CIs ----
    res = []
    sub = lab[lab.informative]
    ys, gs = sub["label"].eq("P").astype(int).values, sub["aapos"].astype(int).values
    for c in ["pps3d_norm", "path_density_3d", "pps1d_norm", "alphamissense", "revel",
              "cadd_phred", "sift", "polyphen2_hvar", "esm1b"]:
        for setname, dd, yy, gg in [("all labelled", lab, y, grp), ("informative only", sub, ys, gs)]:
            v = pd.to_numeric(dd[c], errors="coerce")
            if c in ("sift", "esm1b"):
                v = -v
            m = v.notna().values
            if m.sum() < 40 or len(np.unique(yy[m])) < 2:
                continue
            a, lo, hi = cluster_bootstrap_auroc(yy[m], v[m].values, gg[m])
            res.append({"feature": c, "subset": setname, "n": int(m.sum()),
                        "auroc": round(a, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    R = pd.DataFrame(res)
    R.to_csv(TAB / "v2_auroc_by_subset.csv", index=False)
    print("\n[v2] AUROC on all-labelled vs informative-only:")
    print(R.pivot(index="feature", columns="subset", values="auroc").to_string())

    # head-to-head on the informative subset: 3D-PPS vs AlphaMissense
    v3 = pd.to_numeric(sub["pps3d_norm"], errors="coerce").values
    vam = pd.to_numeric(sub["alphamissense"], errors="coerce").values
    m = ~(np.isnan(v3) | np.isnan(vam))
    dlt, lo, hi = paired_diff(ys[m], v3[m], vam[m], gs[m])
    print(f"\n[v2] 3D-PPS vs AlphaMissense on informative subset: "
          f"dAUROC={dlt:+.3f} [{lo:+.3f},{hi:+.3f}]  "
          f"{'(equivalent within 0.05)' if (lo>-0.05 and hi<0.05) else ''}")
    pd.DataFrame([{"comparison": "3D-PPS - AlphaMissense (informative subset)",
                   "delta": round(dlt, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)}]
                 ).to_csv(TAB / "v2_head_to_head.csv", index=False)

    # ---- model ablation on primary label set ----
    print("\n[v2] leakage-free model ablation (primary label set):")
    ab = []
    for nm, cols, up in [("sequence-only", PRED_ALL + STRUCT_CTX, False),
                         ("sequence+3D-PPS", PRED_ALL + STRUCT_CTX, True),
                         ("ClinVar-free+3D", [c for c in PRED_ALL if c not in CLINVAR_TRAINED] + STRUCT_CTX, True)]:
        a, ap, _, _ = cv_auc(lab, cols, use_pps=up, ca=ca)
        ab.append({"model": nm, "auroc": round(a, 3), "auprc": round(ap, 3)})
        print(f"   {nm:20s} AUROC={a:.3f} AUPRC={ap:.3f}")
    pd.DataFrame(ab).to_csv(TAB / "v2_ablation.csv", index=False)

    # ---- corrected structural orphans (require >=MIN_SUPPORT pathogenic neighbours) ----
    vus = pd.read_parquet(OUT / "vus_annotated.parquet")
    vp, vb = neighbour_support(pd.concat([vus.assign(label="VUS"), d[d.label.isin(["P","B"])]]),
                              ca)
    vus = vus.copy()
    vus["n_path_nb"] = vp[:len(vus)]
    vus["n_ben_nb"] = vb[:len(vus)]
    orph = vus[(vus.n_path_nb >= MIN_SUPPORT) & (vus.n_ben_nb == 0)
               & (vus.pps1d_norm < 0.3)
               & (vus.absent_from_gnomad | (vus.gnomad_af.fillna(0) < 1e-5))
               & ((vus.sift >= 0.05) | (vus.polyphen2_hvar < 0.5))]
    orph = orph.sort_values("n_path_nb", ascending=False)
    orph.to_csv(TAB / "v2_structural_orphans.csv", index=False)
    print(f"\n[v2] CORRECTED structural orphans (>= {MIN_SUPPORT} pathogenic neighbours, "
          f"0 benign, ultra-rare, legacy-tolerated): {len(orph)}")
    if len(orph):
        print(orph[["hgvs_p", "aapos", "n_path_nb", "RCS", "alphamissense", "am_pred",
                    "provisional_class"]].head(12).to_string(index=False))

    # ---- discordance range ----
    disc = []
    for rt, ct, pt, nm in [(0.5, 20, 0.5, "nominal"), (0.644, 20, 0.5, "ClinGen PP3 supporting"),
                           (0.773, 25, 0.85, "stringent"), (0.290, 15, 0.15, "lenient")]:
        c = []
        for _, r in vus.iterrows():
            calls = []
            if pd.notna(r.sift): calls.append(r.sift < 0.05)
            if pd.notna(r.polyphen2_hvar): calls.append(r.polyphen2_hvar > pt)
            if pd.notna(r.revel): calls.append(r.revel > rt)
            if pd.notna(r.am_pred): calls.append(r.am_pred == "P")
            if pd.notna(r.cadd_phred): calls.append(r.cadd_phred > ct)
            if len(calls) >= 3: c.append(0 < sum(calls) < len(calls))
        disc.append({"thresholds": nm, "n_evaluable": len(c), "pct_discordant": round(100*np.mean(c), 1)})
    D = pd.DataFrame(disc); D.to_csv(TAB / "v2_discordance_sensitivity.csv", index=False)
    print("\n[v2] discordance sensitivity:"); print(D.to_string(index=False))
    print("\n[v2] tables written to results/tables/v2_*.csv")


if __name__ == "__main__":
    main()
