"""Vectorised, exact 3D-PPS machinery (v6).

Differences from v5 `PPSCalculator`:
  * NO epsilon and NO zero-fill. 3D-PPS = (dP - dB)/(dP + dB), undefined (NaN, the
    score abstains) when dP + dB == 0.
  * All kernels are precomputed once as residue x residue matrices, so cross validation,
    permutation nulls and spatial block CV are matrix products.
  * A residue never sees its own label (q == r excluded) nor residues within +-WIN in
    sequence; neighbours below the pLDDT floor are excluded from the SCORE (not from the
    v5-style coverage count, which keeps v5's definition; a pLDDT-gated count is also kept).

Index convention: residue number r (1..1935) -> row r-1.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

PDB = Path("data/external/AF-P12883-F1-model_v6.pdb")
RADIUS, SIGMA, WIN, MIN_PLDDT = 10.0, 5.0, 4, 50.0
SEQ_SIGMA, SEQ_CUTOFF = 10.0, 30
MIN_SUPPORT = 3


def load_ca(pdb_path: Path = PDB):
    s = PDBParser(QUIET=True).get_structure("m", str(pdb_path))
    out = {}
    for r in s[0]["A"]:
        if "CA" in r:
            ca = r["CA"]
            out[r.id[1]] = (ca.get_coord().astype(float), float(ca.get_bfactor()))
    return out


class Kernels:
    def __init__(self, ca=None):
        ca = ca or load_ca()
        self.n = max(ca)
        n = self.n
        coords = np.full((n, 3), np.nan)
        plddt = np.full(n, np.nan)
        for r, (xyz, pl) in ca.items():
            coords[r - 1], plddt[r - 1] = xyz, pl
        self.plddt, self.coords = plddt, coords
        have = ~np.isnan(plddt)
        tree = cKDTree(coords[have])
        idx = np.where(have)[0]
        Wk = np.zeros((n, n)); A = np.zeros((n, n), dtype=np.int8); Ag = np.zeros((n, n), dtype=np.int8)
        for i in idx:
            for jj in tree.query_ball_point(coords[i], RADIUS):
                j = idx[jj]
                if j == i or abs(j - i) <= WIN:
                    continue
                A[i, j] = 1                                    # v5 coverage definition
                if plddt[j] >= MIN_PLDDT:
                    d = np.linalg.norm(coords[i] - coords[j])
                    Wk[i, j] = np.exp(-d * d / (2 * SIGMA ** 2)) * (plddt[j] / 100.0)
                    Ag[i, j] = 1                               # pLDDT-gated count
        W1 = np.zeros((n, n))
        for i in idx:
            for q in range(i - SEQ_CUTOFF, i + SEQ_CUTOFF + 1):
                if 0 <= q < n and have[q] and q != i and abs(q - i) > WIN:
                    W1[i, q] = np.exp(-(q - i) ** 2 / (2 * SEQ_SIGMA ** 2))
        # coordination (burial): CA within 10 A, all neighbours incl. +-4, minus self
        coord = np.full(n, np.nan)
        for i in idx:
            coord[i] = len(tree.query_ball_point(coords[i], RADIUS)) - 1
        self.Wk, self.A, self.Ag, self.W1, self.coordination = Wk, A, Ag, W1, coord

    def masks(self, res_p, res_b):
        pm = np.zeros(self.n); bm = np.zeros(self.n)
        for r in set(int(x) for x in res_p):
            if 1 <= r <= self.n: pm[r - 1] = 1
        for r in set(int(x) for x in res_b):
            if 1 <= r <= self.n: bm[r - 1] = 1
        return pm, bm

    def score(self, res, pm, bm):
        """Scores for variants at residues `res` given pathogenic/benign residue masks."""
        ri = np.asarray(res, dtype=int) - 1
        # macOS Accelerate emits spurious "invalid value in matmul" RuntimeWarnings even for
        # finite inputs (flagged in the v5 audit). Inputs are asserted finite instead.
        assert np.isfinite(pm).all() and np.isfinite(bm).all() and np.isfinite(self.Wk).all()
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
        p3, b3 = self.Wk[ri] @ pm, self.Wk[ri] @ bm
        p1, b1 = self.W1[ri] @ pm, self.W1[ri] @ bm
        npth, nben = self.A[ri] @ pm, self.A[ri] @ bm
        gp, gb = self.Ag[ri] @ pm, self.Ag[ri] @ bm
        with np.errstate(invalid="ignore", divide="ignore"):
            s3 = np.where(p3 + b3 > 0, (p3 - b3) / (p3 + b3), np.nan)
            s1 = np.where(p1 + b1 > 0, (p1 - b1) / (p1 + b1), np.nan)
        return {"pps3d_norm": s3, "pps1d_norm": s1, "path_density_3d": p3, "ben_density_3d": b3,
                "path_density_1d": p1, "ben_density_1d": b1,
                "n_path_nb": npth, "n_ben_nb": nben, "n_nb": npth + nben,
                "n_nb_gated": gp + gb,
                "pps3d_raw": p3 - b3, "pps1d_raw": p1 - b1,
                "plddt": self.plddt[ri], "coordination": self.coordination[ri]}
