"""Code validation: run the v6 machinery on the UNCHANGED v5 table (data/legacy_v5/features.parquet,
read-only) and check it reproduces the v5 audit numbers. Proves differences between v5 and v6
come from the data, not from the new code.

Run: python -m src.validate_on_v5
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from src.common import *   # noqa
from src.core3d import Kernels


def main():
    K = Kernels()
    old = pd.read_parquet("data/legacy_v5/features.parquet")
    lab = old[old.label.isin(["P", "B"]) & old.aapos.between(1, 1935)].reset_index(drop=True)
    y = (lab.label == "P").astype(int).values
    pm, bm = K.masks(lab.loc[lab.label == "P", "aapos"], lab.loc[lab.label == "B", "aapos"])
    sc = K.score(lab.aapos.values, pm, bm)
    s = sc["pps3d_norm"]; ok = ~np.isnan(s)
    cov = (sc["n_nb"] >= 3).astype(int)
    tab = [[((y == 1) & (cov == 1)).sum(), ((y == 1) & (cov == 0)).sum()], [((y == 0) & (cov == 1)).sum(), ((y == 0) & (cov == 0)).sum()]]
    # unconditional permutation on v5 data (same code path as perm_v6)
    n = len(lab); res = lab.aapos.values.astype(int)
    Inc = np.zeros((n, K.n)); Inc[np.arange(n), res - 1] = 1; Wsub = K.Wk[res - 1]
    def stat(yv):
        pmv = (Inc.T @ yv > 0).astype(float); bmv = (Inc.T @ (1 - yv) > 0).astype(float)
        p3, b3 = Wsub @ pmv, Wsub @ bmv
        with np.errstate(invalid="ignore", divide="ignore"):
            ss = np.where(p3 + b3 > 0, (p3 - b3) / (p3 + b3), np.nan)
        return auroc(yv[ok], ss[ok])
    rng = np.random.default_rng(42)
    null = np.array([stat(rng.permutation(y).astype(float)) for _ in range(10000)])
    obs = auroc(y[ok], s[ok])
    rows = [("scoreable n (v5 audit: 458)", int(ok.sum())), ("scoreable P/B (404/54)", f"{int(y[ok].sum())}/{int((1-y[ok]).sum())}"),
            ("exact 3D-PPS AUROC (v5 audit: 0.8415)", r3(obs, 4)), ("covered >=3 (v5 audit: 245)", int(cov.sum())),
            ("coverage flag>=1 AUROC, valid positions only (audit: 0.662)", r3(auroc(y, (sc["n_nb"] >= 1).astype(float)))),
            ("OR covered>=3, valid positions only (audit: 13.6)", r3(fisher_exact(tab)[0], 2)),
            ("null1 mean/sd (v5: 0.497/0.0435)", f"{null.mean():.4f}/{null.std():.4f}"),
            ("null1 max (v5: 0.6746)", r3(null.max(), 4)), ("null1 p (v5: 1e-4)", (int((null >= obs).sum()) + 1) / 10001)]
    pd.DataFrame(rows, columns=["check", "v6_code_on_v5_data"]).to_csv(TAB / "v6_validation_on_v5data.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
