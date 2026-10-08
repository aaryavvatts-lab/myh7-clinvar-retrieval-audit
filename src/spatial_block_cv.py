"""v6 NEW analysis: spatially blocked / region-held-out evaluation.

All labelled variants in a held-out region are removed from the training labels, 3D-PPS
for the held-out variants is computed ONLY from labels outside the region (so it must
generalise through 3D contacts to labelled residues elsewhere in the sequence), and
predictors are compared on the pooled held-out predictions.

Schemes: contiguous 100-residue blocks (20 blocks) and four coarse regions
(motor 1-780, lever arm / light-chain binding 781-843, S2 844-1200, light meromyosin 1201-1935;
boundaries approximate and fixed a priori, not tuned).

Methods compared on the same held-out variants:
  3D-PPS (scoreable only), 1D-PPS, coverage flag (>=1 train neighbour), AlphaMissense, REVEL,
  logistic models A_only / A_cov / A_3d trained on the out-of-region labelled variants.
CIs: residue-cluster bootstrap (3,000, seed 42). Paired differences use the same variants.

Outputs: results/tables/spatial_block_cv.csv, spatial_block_cv_paired.csv,
         spatial_block_cv_by_block.csv
Run: python -m src.spatial_block_cv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.common import *   # noqa
from src.build_features import kernels
from src.model_v6 import design, _lr

DOMAINS = [("motor 1-780", 1, 780), ("lever arm 781-843", 781, 843), ("S2 844-1200", 844, 1200),
           ("LMM 1201-1935", 1201, 1935)]


def blocks_of(aapos, scheme):
    if scheme == "blocks100":
        return ((aapos - 1) // 100).astype(int)
    out = np.full(len(aapos), -1)
    for k, (_, lo, hi) in enumerate(DOMAINS):
        out[(aapos >= lo) & (aapos <= hi)] = k
    return out


def run_scheme(lab, scheme, K):
    y = lab.y.values; res = lab.aapos.values
    blk = blocks_of(res, scheme)
    n = len(lab)
    P = {k: np.full(n, np.nan) for k in ("pps3d", "pps1d", "flag1", "A_only", "A_cov", "A_3d", "am", "revel")}
    cover = np.zeros(n)
    for b in np.unique(blk):
        te = np.where(blk == b)[0]; tr = np.where(blk != b)[0]
        if len(te) == 0 or y[tr].min() == y[tr].max():
            continue
        pm, bm = K.masks(lab.iloc[tr].loc[lab.iloc[tr].lab.eq("P"), "aapos"], lab.iloc[tr].loc[lab.iloc[tr].lab.eq("B"), "aapos"])
        sc = K.score(res[te], pm, bm)
        P["pps3d"][te] = sc["pps3d_norm"]; P["pps1d"][te] = sc["pps1d_norm"]
        P["flag1"][te] = (sc["n_nb"] >= 1).astype(float); cover[te] = sc["n_nb"]
        for fs in ("A_only", "A_cov", "A_3d"):
            X, _ = design(lab, fs, K, tr)
            P[fs][te] = _lr().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    P["am"] = lab.alphamissense.values.astype(float); P["revel"] = lab.revel.values.astype(float)
    return P, blk, cover


def main():
    from src.build_labels import spec_dir
    K = kernels()
    rows, pair_rows, blk_rows = [], [], []
    jobs = [("primary", "wider", "data/processed"), ("primary", "tighter", "data/processed"),
            ("v5rcv_v4", "wider", spec_dir("v4", "v5rcv")), ("v5rcv_v2", "wider", spec_dir("v2", "v5rcv"))]
    for cohort, setname, sd in jobs:
        df = load(sd)
        lab = labelled(df, setname); y = lab.y.values; g = lab.aapos.values
        for scheme in ("blocks100", "domains"):
            P, blk, cover = run_scheme(lab, scheme, K)
            sc = ~np.isnan(P["pps3d"])
            methods = {"3D-PPS (scoreable only)": P["pps3d"], "1D-PPS (scoreable only)": P["pps1d"],
                       "coverage flag (>=1 training neighbour)": P["flag1"], "AlphaMissense": P["am"],
                       "REVEL": P["revel"], "model A_only": P["A_only"], "model A_cov": P["A_cov"],
                       "model A_3d": P["A_3d"]}
            for subset, m in (("all held-out variants", np.ones(len(y), bool)),
                              ("3D-PPS scoreable subset", sc)):
                for nm, s in methods.items():
                    s2 = np.where(m, s, np.nan)
                    a, lo, hi, nn = auroc_ci(y, s2, g)
                    ok = ~np.isnan(s2)
                    rows.append({"cohort": cohort, "scheme": scheme, "set": setname, "subset": subset, "method": nm, "n": nn,
                                 "n_P": int(y[ok].sum()), "n_B": int((1 - y[ok]).sum()),
                                 "pct_of_heldout_scored": r3(100 * ok.mean(), 1),
                                 "auroc": r3(a), "ci_low": r3(lo), "ci_high": r3(hi)})
            for a_, b_ in [("pps3d", "am"), ("pps3d", "pps1d"), ("pps3d", "flag1"), ("A_3d", "A_only"), ("A_3d", "A_cov")]:
                for subset, m in (("all held-out (pairwise complete)", np.ones(len(y), bool)), ("3D-PPS scoreable subset", sc)):
                    r = paired_diff(y[m], P[a_][m], P[b_][m], g[m])
                    yy = y[m][~(np.isnan(P[a_][m]) | np.isnan(P[b_][m]))]
                    pair_rows.append({"cohort": cohort, "scheme": scheme, "set": setname, "subset": subset, "A": a_, "B": b_,
                                      "n_P": int(yy.sum()), "n_B": int((1 - yy).sum()),
                                      **{k: (r3(v) if isinstance(v, float) else v) for k, v in r.items()}})
            if setname == "wider" and cohort in ("primary", "v5rcv_v4"):
                for b in np.unique(blk):
                    m = blk == b
                    ok = m & sc
                    blk_rows.append({"cohort": cohort, "scheme": scheme, "block": int(b),
                                     "label": (DOMAINS[b][0] if scheme == "domains" else f"{b*100+1}-{b*100+100}"),
                                     "n": int(m.sum()), "n_P": int(y[m].sum()), "n_B": int((1 - y[m]).sum()),
                                     "pct_3d_abstained": r3(100 * (~sc[m]).mean(), 1) if m.sum() else None,
                                     "auroc_3d_pps": r3(auroc(y[ok], P["pps3d"][ok])) if ok.sum() and len(np.unique(y[ok])) > 1 else None,
                                     "auroc_alphamissense": r3(auroc(y[m & ~np.isnan(P["am"])], P["am"][m & ~np.isnan(P["am"])])) if len(np.unique(y[m])) > 1 else None})
    R = pd.DataFrame(rows)
    R["caveat"] = np.where(R.n_B < 10, "fewer than 10 benign: CI not reliable", "")
    R.to_csv(TAB / "spatial_block_cv.csv", index=False)
    PR = pd.DataFrame(pair_rows); PR["caveat"] = np.where(PR.n_B < 10, "fewer than 10 benign: CI not reliable", "")
    PR.to_csv(TAB / "spatial_block_cv_paired.csv", index=False)
    pd.DataFrame(blk_rows).to_csv(TAB / "spatial_block_cv_by_block.csv", index=False)
    print(R[(R.cohort=="v5rcv_v4")].to_string()); print(PR[PR.cohort=="v5rcv_v4"].to_string())


if __name__ == "__main__":
    main()
