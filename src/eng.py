"""v7 engine: gene-agnostic 3D-PPS geometry, scores S0/S1/S2 (PREREG section 3), tier datasets, AUROC with
ties, tie decomposition, residue-cluster bootstrap (paired), jackknife over benign clusters, TOST, Holm.

Index convention: residue r (1..L) -> row r-1. Scores for a variant use labels of OTHER residues only
(own residue and |q-r| <= WIN excluded in the kernels).
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from scipy.stats import rankdata, t as tdist
from src.genes import *   # noqa

SEED = 42
N_BOOT = 3000
RADIUS, SIGMA, WIN, MIN_PLDDT, SEQ_SIGMA, SEQ_CUT = 10.0, 5.0, 4, 50.0, 10.0, 30
PRED_ALL = ["sift", "sift4g", "polyphen2_hdiv", "polyphen2_hvar", "revel", "alphamissense", "metalr", "clinpred",
            "primateai", "vest4", "provean", "mpc", "esm1b", "cadd_phred", "phylop100"]
FLIP = {"sift", "sift4g", "provean", "esm1b"}
BINS = [-0.1, 0.5, 2.5, 5.5, 10.5, 1e9]


def pdb_path(gene):
    if gene == "MYH7":
        return Path("data/external/AF-P12883-F1-model_v6.pdb")
    return EXT / "alphafold" / f"AF-{UNIPROT[gene]}-F1-model.pdb"


# ------------------------------------------------------------------ geometry
class Geom:
    def __init__(self, pdb, radius=RADIUS, sigma=SIGMA, win=WIN, min_plddt=MIN_PLDDT, seq_sigma=SEQ_SIGMA, seq_cut=SEQ_CUT):
        s = PDBParser(QUIET=True).get_structure("m", str(pdb))
        ca = {}
        for r in s[0]["A"]:
            if "CA" in r:
                ca[r.id[1]] = (r["CA"].get_coord().astype(float), float(r["CA"].get_bfactor()))
        n = max(ca); self.n = n
        coords = np.full((n, 3), np.nan); plddt = np.full(n, np.nan)
        for r, (xyz, pl) in ca.items():
            coords[r - 1], plddt[r - 1] = xyz, pl
        self.plddt, self.coords = plddt, coords
        have = ~np.isnan(plddt); idx = np.where(have)[0]
        tree = cKDTree(coords[have])
        Wk = np.zeros((n, n)); A = np.zeros((n, n), dtype=np.int8); Ag = np.zeros((n, n), dtype=np.int8)
        coord = np.full(n, np.nan)
        for ii, i in enumerate(idx):
            nb = tree.query_ball_point(coords[i], radius)
            coord[i] = len(nb) - 1
            for jj in nb:
                j = idx[jj]
                if j == i or abs(j - i) <= win:
                    continue
                A[i, j] = 1
                if plddt[j] >= min_plddt:
                    d = np.linalg.norm(coords[i] - coords[j])
                    Wk[i, j] = np.exp(-d * d / (2 * sigma ** 2)) * (plddt[j] / 100.0)
                    Ag[i, j] = 1
        W1 = np.zeros((n, n))
        for i in idx:
            for q in range(max(0, i - seq_cut), min(n, i + seq_cut + 1)):
                if have[q] and q != i and abs(q - i) > win:
                    W1[i, q] = np.exp(-(q - i) ** 2 / (2 * seq_sigma ** 2))
        self.Wk, self.A, self.Ag, self.W1, self.coordination = Wk, A, Ag, W1, coord

    def masks(self, res_p, res_b):
        pm = np.zeros(self.n); bm = np.zeros(self.n)
        for r in set(int(x) for x in res_p):
            if 1 <= r <= self.n: pm[r - 1] = 1
        for r in set(int(x) for x in res_b):
            if 1 <= r <= self.n: bm[r - 1] = 1
        return pm, bm

    def dens(self, res, pm, bm):
        ri = np.asarray(res, dtype=int) - 1
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
        Wk = self.Wk[ri]; W1 = self.W1[ri]; A = self.A[ri]
        return {"dP": Wk @ pm, "dB": Wk @ bm, "dP1": W1 @ pm, "dB1": W1 @ bm, "nP": A @ pm, "nB": A @ bm}


def derive(d, NP, NB, k=K_SHRINK):
    """Scores from densities. NP, NB = numbers of labelled P and B residues (for S2)."""
    dP, dB = d["dP"], d["dB"]
    den = dP + dB
    with np.errstate(invalid="ignore", divide="ignore"):
        S0 = np.where(den > 0, (dP - dB) / den, np.nan)
        S01d = np.where(d["dP1"] + d["dB1"] > 0, (d["dP1"] - d["dB1"]) / (d["dP1"] + d["dB1"]), np.nan)
    S1 = (dP - dB) / (den + k)
    S11d = (d["dP1"] - d["dB1"]) / (d["dP1"] + d["dB1"] + k)
    Nbar = (NP + NB) / 2.0
    dPp = dP * (Nbar / max(NP, 1)); dBp = dB * (Nbar / max(NB, 1))
    S2 = (dPp - dBp) / (dPp + dBp + k)
    nn = d["nP"] + d["nB"]
    return {"S0": S0, "S0_1d": S01d, "S1": S1, "S1_1d": S11d, "S2": S2, "raw_diff": dP - dB, "raw_diff_1d": d["dP1"] - d["dB1"],
            "n_nb": nn, "n_pth": d["nP"], "n_ben": d["nB"], "dP": dP, "dB": dB}


def build_ds(lab, geom, k=K_SHRINK):
    """lab: DataFrame with columns aapos, lab ('P'/'B') and (optionally) predictors. Adds y, scores, structure columns."""
    lab = lab.copy().reset_index(drop=True)
    lab["aapos"] = lab.aapos.astype(int)
    lab["y"] = (lab.lab == "P").astype(int)
    pm, bm = geom.masks(lab.loc[lab.lab == "P", "aapos"], lab.loc[lab.lab == "B", "aapos"])
    sc = derive(geom.dens(lab.aapos.values, pm, bm), pm.sum(), bm.sum(), k)
    for c, v in sc.items():
        lab[c] = v
    lab["S0_neutral"] = lab.S0.fillna(0.0)
    lab["S0_lowest"] = lab.S0.fillna(-1.0)
    lab["S0_1d_neutral"] = lab.S0_1d.fillna(0.0)
    lab["coordination"] = geom.coordination[lab.aapos.values - 1]
    lab["plddt"] = geom.plddt[lab.aapos.values - 1]
    lab["covered_ge1"] = (lab.n_nb >= 1).astype(float)
    lab["covered_ge3"] = (lab.n_nb >= 3).astype(float)
    for c in PRED_ALL:
        if c in lab:
            v = pd.to_numeric(lab[c], errors="coerce")
            lab["o_" + c] = -v if c in FLIP else v
    return lab


def tier_dataset(gene, tier, geom, protein_table=None, k=K_SHRINK):
    """Labelled P/B protein changes of a tier with S0/S1/S2/1D/count scores (own-residue-excluded)."""
    pt = protein_table if protein_table is not None else pd.read_parquet(PROC / gene / "protein_table.parquet")
    col = f"lab_{tier}"
    lab = pt[pt[col].isin(["P", "B"]) & pt.seq_ok & pt.aapos.between(1, geom.n)].copy()
    lab["lab"] = lab[col]
    return build_ds(lab, geom, k)


# ------------------------------------------------------------------ AUROC machinery
def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    m = ~np.isnan(s); y = y[m]; s = s[m]
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pair_decomp(y, s):
    """Shares of P-B pairs where the pathogenic score is greater / tied / smaller; number of distinct values."""
    y = np.asarray(y); s = np.asarray(s, float); m = ~np.isnan(s); y, s = y[m], s[m]
    P = s[y == 1]; B = s[y == 0]
    if len(P) == 0 or len(B) == 0:
        return dict(n_P=len(P), n_B=len(B), p_greater=np.nan, p_tie=np.nan, p_smaller=np.nan, n_distinct=len(np.unique(s)))
    vals, inv = np.unique(np.concatenate([P, B]), return_inverse=True)
    cP = np.bincount(inv[:len(P)], minlength=len(vals)); cB = np.bincount(inv[len(P):], minlength=len(vals))
    tot = len(P) * len(B)
    tie = float((cP * cB).sum()) / tot
    cumB = np.cumsum(cB) - cB          # benign strictly below each value
    greater = float((cP * cumB).sum()) / tot
    return dict(n_P=len(P), n_B=len(B), p_greater=greater, p_tie=tie, p_smaller=1 - greater - tie, n_distinct=len(vals))


def _members(groups):
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    return uniq, [np.where(inv == k)[0] for k in range(len(uniq))]


def boot_draws(groups, y, n=N_BOOT, seed=SEED, stratified=False):
    """Yield index arrays of residue-cluster bootstrap resamples (both classes present)."""
    rng = np.random.default_rng(seed)
    uniq, members = _members(groups)
    y = np.asarray(y)
    if stratified:
        isB = np.array([bool((y[m] == 0).any()) for m in members])
        cB = np.where(isB)[0]; cO = np.where(~isB)[0]
    got = 0; tries = 0
    while got < n and tries < n * 3:
        tries += 1
        if stratified:
            samp = np.concatenate([rng.choice(cB, len(cB)), rng.choice(cO, len(cO))])
        else:
            samp = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in samp])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        got += 1
        yield idx


def boot_auc(y, groups, scores, pairs=(), n=N_BOOT, seed=SEED, stratified=False):
    """Point AUROC + percentile CI per score, and paired differences (same resamples).
    scores: dict name -> array (NaN = abstain). pairs: list of (a, b) -> a minus b on the variants where both defined."""
    y = np.asarray(y); groups = np.asarray(groups)
    names = list(scores); S = {k: np.asarray(v, float) for k, v in scores.items()}
    rep = {k: [] for k in names}; repd = {p: [] for p in pairs}
    for idx in boot_draws(groups, y, n, seed, stratified):
        yy = y[idx]
        for k in names:
            rep[k].append(auroc(yy, S[k][idx]))
        for a, b in pairs:
            sa, sb = S[a][idx], S[b][idx]; m = ~(np.isnan(sa) | np.isnan(sb))
            repd[(a, b)].append(auroc(yy[m], sa[m]) - auroc(yy[m], sb[m]) if m.sum() > 1 else np.nan)
    out = {}
    for k in names:
        r = np.array(rep[k]); r = r[np.isfinite(r)]
        pt = auroc(y, S[k])
        out[k] = dict(auroc=pt, ci_low=np.percentile(r, 2.5) if len(r) else np.nan, ci_high=np.percentile(r, 97.5) if len(r) else np.nan,
                      boot_sd=r.std() if len(r) else np.nan, n_boot=len(r))
    for a, b in pairs:
        m = ~(np.isnan(S[a]) | np.isnan(S[b]))
        pt = auroc(y[m], S[a][m]) - auroc(y[m], S[b][m])
        r = np.array(repd[(a, b)]); r = r[np.isfinite(r)]
        out[(a, b)] = dict(delta=pt, ci_low=np.percentile(r, 2.5), ci_high=np.percentile(r, 97.5), boot_sd=r.std(), n=int(m.sum()),
                           mdd_80=2.8 * r.std(), reps=r)
    return out


def tost_p(reps, delta=0.02):
    """Bootstrap TOST: H0 delta <= -margin and H0 delta >= +margin; add-one p; floor 1/(B+1)."""
    reps = np.asarray(reps)
    return float(max((np.sum(reps <= -delta) + 1) / (len(reps) + 1), (np.sum(reps >= delta) + 1) / (len(reps) + 1)))


def diff_p(reps):
    """Two-sided percentile-bootstrap p for delta != 0; floor 2/(B+1)."""
    reps = np.asarray(reps)
    return float(min(1.0, 2 * min((np.sum(reps <= 0) + 1) / (len(reps) + 1), (np.sum(reps >= 0) + 1) / (len(reps) + 1))))


def jackknife_benign(y, groups, fn):
    """Delete-one-cluster jackknife over clusters that contain a benign variant; returns (theta, se, df, ci_low, ci_high)."""
    y = np.asarray(y); groups = np.asarray(groups)
    theta = fn(np.arange(len(y)))
    uniq, members = _members(groups)
    bcl = [k for k, m in enumerate(members) if (y[m] == 0).any()]
    vals = []
    for k in bcl:
        keep = np.ones(len(y), bool); keep[members[k]] = False
        vals.append(fn(np.where(keep)[0]))
    vals = np.array(vals); m = len(vals)
    se = float(np.sqrt((m - 1) / m * ((vals - vals.mean()) ** 2).sum()))
    q = tdist.ppf(0.975, m - 1)
    return float(theta), se, m - 1, float(theta - q * se), float(theta + q * se)


def holm(pvals):
    """Holm step-down adjusted p-values (same order)."""
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def r3(x, k=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)
