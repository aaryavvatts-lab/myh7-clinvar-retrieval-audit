"""v6 permutation tests for the exact 3D-PPS (scoreable variants only; ONE observed value).

Null 1  (unconditional, as v5): shuffle class labels across all labelled variants.
        Tests whether class is spatially structured AT ALL; it cannot separate signal from
        coverage because the shuffle also destroys the coverage-class association.
Null 2  (coverage-preserving): shuffle class labels only WITHIN strata of
        (labelled-neighbour count bin) x (burial tertile). Coverage and burial are held
        exactly fixed by construction and so is the P/B proportion inside each stratum, so
        a rejection means label balance carries information BEYOND coverage and burial.
10,000 shuffles each, seed 42, add-one p-values.

Run: python -m src.perm_v6
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.common import *   # noqa
from src.build_features import kernels
from src.coverage_v6 import BINS

N_PERM = 10000


def run(setname, K, lab=None, tag="primary"):
    if lab is None:
        lab = pd.read_parquet(f"data/processed/labelled_{setname}.parquet")
    y = lab.y.values; res = lab.aapos.values.astype(int)
    n = len(lab)
    Inc = np.zeros((n, K.n))
    Inc[np.arange(n), res - 1] = 1
    Wsub = K.Wk[res - 1]                      # n x residues
    ok = lab.score_defined.values             # abstention pattern is invariant under shuffling

    def stat(yv):
        pm = (Inc.T @ yv > 0).astype(float); bm = (Inc.T @ (1 - yv) > 0).astype(float)
        p3, b3 = Wsub @ pm, Wsub @ bm
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(p3 + b3 > 0, (p3 - b3) / (p3 + b3), np.nan)
        return auroc(yv[ok], s[ok])

    obs = stat(y.astype(float))
    assert abs(obs - auroc(y[ok], lab.pps3d_norm.values[ok])) < 1e-9, "observed AUROC mismatch"
    # strata for null 2
    nb = pd.cut(lab.n_nb, bins=BINS, labels=False)
    bur = pd.qcut(lab.coordination, 3, labels=False, duplicates="drop")
    cell = (nb.astype(int) * 10 + bur.astype(int)).values
    cells = [np.where(cell == c)[0] for c in np.unique(cell)]
    rng1 = np.random.default_rng(42); rng2 = np.random.default_rng(42)
    null1 = np.empty(N_PERM); null2 = np.empty(N_PERM)
    for i in range(N_PERM):
        null1[i] = stat(rng1.permutation(y).astype(float))
        yp = y.copy()
        for idx in cells:
            yp[idx] = rng2.permutation(y[idx])
        null2[i] = stat(yp.astype(float))
    rows = []
    for nm, nl in (("null1_unconditional", null1), ("null2_coverage_burial_strata", null2)):
        nge = int((nl >= obs).sum())
        rows.append({"cohort": tag, "set": setname, "null": nm, "observed_auroc": r3(obs, 4), "n_scoreable": int(ok.sum()),
                     "n_P_scoreable": int(y[ok].sum()), "n_B_scoreable": int((1 - y[ok]).sum()),
                     "null_mean": r3(nl.mean(), 4), "null_sd": r3(nl.std(), 4),
                     "null_p95": r3(np.percentile(nl, 95), 4), "null_max": r3(nl.max(), 4),
                     "n_perm": N_PERM, "n_ge_observed": nge, "p_value": (nge + 1) / (N_PERM + 1),
                     "n_strata_cells": len(cells) if nm.startswith("null2") else None})
    if tag == "primary":
        np.save(TAB / f"v6_permutation_null1_{setname}.npy", null1)
        np.save(TAB / f"v6_permutation_null2_{setname}.npy", null2)
    return rows


def main():
    K = kernels()
    rows = []
    for setname in SETS:
        rows += run(setname, K)
        print(pd.DataFrame(rows).tail(2).to_string())
    from src.build_labels import spec_dir
    for af, rule in [("v4", "v5rcv"), ("v2", "aggregate"), ("v2", "v5rcv")]:
        lab = labelled(load(spec_dir(af, rule)), "wider")
        rows += run("wider", K, lab, tag=f"{rule}_{af}")
        print(pd.DataFrame(rows).tail(2).to_string())
    pd.DataFrame(rows).to_csv(TAB / "v6_permutation.csv", index=False)


if __name__ == "__main__":
    main()
