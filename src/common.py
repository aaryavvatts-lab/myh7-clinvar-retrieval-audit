"""Shared helpers for v6 analyses: data loading, label sets, AUROC + residue-cluster
bootstrap, TOST, and the numbers registry (results/numbers.json)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.build_features import add_scores

TAB = Path("results/tables"); FIG = Path("results/figures")
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
N_BOOT = 3000
SETS = {"wider": "label", "tighter": "label_tight"}      # AF >= 1e-5, AF >= 5e-5

PRED_ALL = ["sift", "sift4g", "polyphen2_hdiv", "polyphen2_hvar", "revel",
            "alphamissense", "metalr", "clinpred", "primateai", "vest4",
            "provean", "mpc", "esm1b", "cadd_phred", "phylop100"]
CLINVAR_TRAINED = ["revel", "clinpred", "metalr", "vest4"]
PRED_POP = [c for c in PRED_ALL if c not in CLINVAR_TRAINED]
STRUCT_CTX = ["coordination", "plddt"]
# dbNSFP columns where LOW = damaging; flipped so higher = more damaging everywhere
FLIP = {"sift", "sift4g", "provean", "esm1b"}


def load(spec_dir="data/processed") -> pd.DataFrame:
    return pd.read_parquet(Path(spec_dir) / "features.parquet")


def labelled(df: pd.DataFrame, setname="wider") -> pd.DataFrame:
    """Labelled (P/B) variants with scores recomputed from THIS label set's labels."""
    col = SETS[setname]
    d = df.copy()
    d["lab"] = d[col]
    d = add_scores(d, "lab")
    lab = d[d.lab.isin(["P", "B"])].reset_index(drop=True)
    lab["y"] = lab.lab.eq("P").astype(int)
    return lab


def oriented(lab, c):
    v = pd.to_numeric(lab[c], errors="coerce")
    return -v if c in FLIP else v


# ---------------------------------------------------------------- AUROC
def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _cluster_index(groups):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    return uniq, members


def boot_stats(fn, y, groups, n=N_BOOT, seed=SEED):
    """Residue-cluster bootstrap of fn(idx)->float. Returns list of replicate values."""
    rng = np.random.default_rng(seed)
    uniq, members = _cluster_index(groups)
    out = []
    for _ in range(n):
        samp = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in samp])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        v = fn(idx)
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def auroc_ci(y, s, groups, n=N_BOOT, seed=SEED):
    """Point-estimate AUROC (NOT the bootstrap mean) with 95% residue-cluster percentile CI."""
    y = np.asarray(y); s = np.asarray(s, dtype=float); groups = np.asarray(groups)
    m = ~np.isnan(s)
    y, s, groups = y[m], s[m], groups[m]
    pt = auroc(y, s)
    if len(np.unique(y)) < 2:
        return pt, np.nan, np.nan, int(m.sum())
    b = boot_stats(lambda i: auroc(y[i], s[i]), y, groups, n, seed)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(pt), float(lo), float(hi), int(m.sum())


def paired_diff(y, sA, sB, groups, margin=0.05, n=N_BOOT, seed=SEED):
    """AUROC(A)-AUROC(B) on the same variants; 95% CI, 90% CI, bootstrap TOST p."""
    y = np.asarray(y); sA = np.asarray(sA, float); sB = np.asarray(sB, float); groups = np.asarray(groups)
    m = ~(np.isnan(sA) | np.isnan(sB))
    y, sA, sB, groups = y[m], sA[m], sB[m], groups[m]
    pt = auroc(y, sA) - auroc(y, sB)
    if len(np.unique(y)) < 2:
        return dict(delta=np.nan, ci_low=np.nan, ci_high=np.nan, p_tost=np.nan, n=int(m.sum()))
    b = boot_stats(lambda i: auroc(y[i], sA[i]) - auroc(y[i], sB[i]), y, groups, n, seed)
    lo, hi = np.percentile(b, [2.5, 97.5]); lo90, hi90 = np.percentile(b, [5, 95])
    p_lower = (np.sum(b <= -margin) + 1) / (len(b) + 1)   # H0: delta <= -margin
    p_upper = (np.sum(b >= margin) + 1) / (len(b) + 1)    # H0: delta >= +margin
    return dict(delta=float(pt), ci_low=float(lo), ci_high=float(hi), ci90_low=float(lo90),
                ci90_high=float(hi90), p_tost=float(max(p_lower, p_upper)),
                equivalent_tost_0p05=bool(max(p_lower, p_upper) < 0.05),
                ci_inside_margin=bool(lo > -margin and hi < margin), n=int(m.sum()))


def r3(x, k=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


# ---------------------------------------------------------------- registry
REG = Path("results/numbers.json")


def record(**kv):
    d = json.loads(REG.read_text()) if REG.exists() else {}
    for k, v in kv.items():
        d[k] = v.item() if hasattr(v, "item") else v
    REG.write_text(json.dumps(d, indent=1, default=str))


def registry():
    return json.loads(REG.read_text()) if REG.exists() else {}
