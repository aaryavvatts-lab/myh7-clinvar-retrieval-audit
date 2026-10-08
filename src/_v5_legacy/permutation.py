"""Proper permutation test with a real null distribution (v4).

Two problems in v3 are fixed here.
1. v3 ran only 30 permutations. The smallest p-value 30 permutations can support is
   1/31 = 0.032, so the reported "p < 0.001" was not supported by the data.
2. The v3 figure drew a smooth Gaussian curve instead of the actual permutation results.
   That is not acceptable, so this script saves the real values and the figure plots them.

Speed comes from building the kernel weight matrix once. After that each permutation is a
matrix product, so 10,000 permutations run in seconds.

Run: python -m src.permutation
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from src.structure import load_ca
from src.reanalysis import relabel

TAB = Path("results/tables"); TAB.mkdir(parents=True, exist_ok=True)
N_PERM = 10000
RADIUS, SIGMA, WIN, MIN_PLDDT = 10.0, 5.0, 4, 50.0


def weight_matrix(lab, ca):
    """W[i, k] = kernel weight from variant i's residue to labelled residue k."""
    res = lab["aapos"].astype(int).values
    uniq = np.array(sorted({r for r in res if r in ca}))
    coords = np.array([ca[r][0] for r in uniq], dtype=float)
    plddt = np.array([ca[r][1] for r in uniq], dtype=float)
    tree = cKDTree(coords)
    pos = {int(r): k for k, r in enumerate(uniq)}
    W = np.zeros((len(res), len(uniq)), dtype=float)
    for i, r in enumerate(res):
        if r not in ca:
            continue
        c = np.asarray(ca[r][0], dtype=float)
        for k in tree.query_ball_point(c, RADIUS):
            q = int(uniq[k])
            if q == r or abs(q - r) <= WIN or plddt[k] < MIN_PLDDT:
                continue
            d = float(np.linalg.norm(c - coords[k]))
            W[i, k] = np.exp(-d * d / (2 * SIGMA ** 2)) * (plddt[k] / 100.0)
    assert np.isfinite(W).all(), "non-finite weights"
    return W, uniq, pos, res


def score_from_labels(W, uniq, res, y):
    """Recompute 3D-PPS for every variant given a label vector y (1 = pathogenic)."""
    # residue is pathogenic if any variant at it is pathogenic; same for benign
    pmask = np.zeros(len(uniq)); bmask = np.zeros(len(uniq))
    pos = {int(r): k for k, r in enumerate(uniq)}
    for r, yy in zip(res, y):
        if r in pos:
            if yy == 1:
                pmask[pos[r]] = 1
            else:
                bmask[pos[r]] = 1
    rp = W @ pmask
    rb = W @ bmask
    # leave-one-residue-out: remove the variant's own residue contribution
    for i, r in enumerate(res):
        if r in pos:
            k = pos[r]
            rp[i] -= W[i, k] * pmask[k]
            rb[i] -= W[i, k] * bmask[k]
    denom = rp + rb
    out = np.full(len(res), np.nan)
    nz = denom > 0
    out[nz] = (rp[nz] - rb[nz]) / denom[nz]
    return out


def main():
    df0 = pd.read_parquet("data/processed/features.parquet")
    ca = load_ca()
    lab = relabel(df0, 1e-5)
    lab = lab[lab.label.isin(["P", "B"])].reset_index(drop=True)
    y = lab.label.eq("P").astype(int).values
    W, uniq, pos, res = weight_matrix(lab, ca)

    obs_scores = score_from_labels(W, uniq, res, y)
    ok = ~np.isnan(obs_scores)
    obs = roc_auc_score(y[ok], obs_scores[ok])
    print(f"[perm] observed AUROC = {obs:.4f}  (n scoreable = {ok.sum()} of {len(y)})")

    rng = np.random.default_rng(42)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        yp = rng.permutation(y)
        s = score_from_labels(W, uniq, res, yp)
        m = ~np.isnan(s)
        null[i] = roc_auc_score(yp[m], s[m]) if len(np.unique(yp[m])) > 1 else 0.5
    n_ge = int((null >= obs).sum())
    p = (n_ge + 1) / (N_PERM + 1)          # add-one estimator, never reports zero
    print(f"[perm] null: mean={null.mean():.4f} sd={null.std():.4f} "
          f"p95={np.percentile(null,95):.4f} max={null.max():.4f}")
    print(f"[perm] permutations >= observed: {n_ge} / {N_PERM}  ->  p = {p:.5f}")
    print(f"[perm] smallest reportable p with {N_PERM} permutations = {1/(N_PERM+1):.5f}")

    np.save(TAB / "permutation_null.npy", null)
    pd.DataFrame({"observed_auroc": [obs], "null_mean": [null.mean()],
                  "null_sd": [null.std()], "null_p95": [np.percentile(null, 95)],
                  "null_max": [null.max()], "n_perm": [N_PERM],
                  "n_ge_observed": [n_ge], "p_value": [p]}
                 ).to_csv(TAB / "permutation_result.csv", index=False)
    print(f"[perm] saved {TAB/'permutation_null.npy'} and permutation_result.csv")


if __name__ == "__main__":
    main()
