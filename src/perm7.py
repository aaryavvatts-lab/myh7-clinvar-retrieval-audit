"""v7 permutation tests (PREREG section 4). Primary = null 2 (coverage x burial strata); null 1 (unconditional) and a
residue-level strata shuffle are secondary. The statistic is recomputed from the permuted labels (masks, densities,
S0/S1/S2) exactly as in v6. One-sided: P higher. p = (count(null >= obs) + 1)/(N + 1).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.eng import *   # noqa

N_PERM = 10000
SCORES = ("S0a", "S0b", "S1", "S2")


def _prep(lab, geom):
    res0 = lab.aapos.values.astype(int) - 1
    Wsub = geom.Wk[res0]
    return res0, Wsub


def _stats(y, res0, Wsub, ok, L, k=K_SHRINK):
    pm = (np.bincount(res0, weights=y, minlength=L) > 0).astype(float)
    bm = (np.bincount(res0, weights=1 - y, minlength=L) > 0).astype(float)
    dP, dB = Wsub @ pm, Wsub @ bm
    den = dP + dB
    with np.errstate(invalid="ignore", divide="ignore"):
        s0 = np.where(den > 0, (dP - dB) / den, np.nan)
    NP, NB = pm.sum(), bm.sum()
    s1 = (dP - dB) / (den + k)
    Nbar = (NP + NB) / 2
    dPp = dP * Nbar / max(NP, 1); dBp = dB * Nbar / max(NB, 1)
    s2 = (dPp - dBp) / (dPp + dBp + k)
    return (auroc(y[ok], s0[ok]), auroc(y, np.nan_to_num(s0)), auroc(y, s1), auroc(y, s2))


def strata_cells(lab):
    nb = pd.cut(lab.n_nb, bins=BINS, labels=False)
    bur = pd.qcut(lab.coordination, 3, labels=False, duplicates="drop")
    cell = (nb.astype(int) * 10 + bur.astype(int)).values
    return [np.where(cell == c)[0] for c in np.unique(cell)]


def run(lab, geom, nperm=N_PERM, seed=SEED, kinds=("strata", "all"), residue_level=False):
    """Returns dict kind -> dict(score -> null array) plus observed dict."""
    res0, Wsub = _prep(lab, geom)
    y = lab.y.values.astype(float); L = geom.n
    ok = lab.S0.notna().values
    obs = _stats(y, res0, Wsub, ok, L)
    chk = auroc(y[ok], lab.S0.values[ok])
    assert abs(obs[0] - chk) < 1e-9 or np.isnan(chk), "observed S0 mismatch"
    cells = strata_cells(lab)
    out = {"observed": dict(zip(SCORES, obs)), "n_cells": len(cells)}
    rng = {k: np.random.default_rng(seed) for k in ("strata", "all", "resid")}
    nulls = {k: np.empty((nperm, 4)) for k in (list(kinds) + (["resid"] if residue_level else []))}
    # residue-level structure for the residue shuffle: permute class composition among residues with the same
    # number of labelled variants within the (residue's modal) cell
    if residue_level:
        resid_cells = []
        cell_of = np.zeros(len(y), int)
        for ci, idx in enumerate(cells):
            cell_of[idx] = ci
        df = pd.DataFrame({"r": res0, "cell": cell_of, "n": 1})
        grp = df.groupby("r").agg(cell=("cell", lambda s: s.mode().iloc[0]), n=("n", "sum"))
        key = grp.cell.astype(str) + "_" + grp.n.astype(str)
        res_groups = [grp.index[key == kk].values for kk in key.unique()]
        var_of_res = {r: np.where(res0 == r)[0] for r in grp.index}
    for i in range(nperm):
        if "all" in nulls:
            nulls["all"][i] = _stats(rng["all"].permutation(y), res0, Wsub, ok, L)
        if "strata" in nulls:
            yp = y.copy()
            for idx in cells:
                yp[idx] = rng["strata"].permutation(y[idx])
            nulls["strata"][i] = _stats(yp, res0, Wsub, ok, L)
        if "resid" in nulls:
            yp = y.copy()
            for rs in res_groups:
                if len(rs) < 2:
                    continue
                perm = rng["resid"].permutation(len(rs))
                for a, b in zip(rs, rs[perm]):
                    yp[var_of_res[a]] = y[var_of_res[b]]
            nulls["resid"][i] = _stats(yp, res0, Wsub, ok, L)
    out["nulls"] = nulls
    return out


def summarize(res, nperm=N_PERM):
    rows = []
    for kind, arr in res["nulls"].items():
        for j, sname in enumerate(SCORES):
            o = res["observed"][sname]; nl = arr[:, j]
            nge = int((nl >= o - 1e-12).sum())
            rows.append({"null": {"strata": "null2_coverage_burial_strata", "all": "null1_unconditional", "resid": "null2_residue_level"}[kind],
                         "score": sname, "observed": o, "null_mean": float(nl.mean()), "null_sd": float(nl.std()),
                         "null_p95": float(np.percentile(nl, 95)), "n_perm": nperm, "n_ge_obs": nge,
                         "p_one_sided": (nge + 1) / (nperm + 1), "p_lower_tail": (int((nl <= o + 1e-12).sum()) + 1) / (nperm + 1),
                         "n_cells": res["n_cells"] if kind != "all" else None})
    return rows
