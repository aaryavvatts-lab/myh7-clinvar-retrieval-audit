"""Phase 2/3 — Axis B: AlphaFold structure + the 3D Pathogenic Proximity Score (3D-PPS).

Core methodological contribution. For each variant residue r we compute a signed,
pLDDT-weighted, kernel density of KNOWN pathogenic vs. benign residues in 3D space,
using only NON-LOCAL neighbours (|seq_r - seq_q| > window) so the score captures
structural clustering invisible to linear conservation. A matched 1D control uses
sequence separation instead of Euclidean distance, isolating the value the fold adds.

Leave-one-out: a variant never sees its own residue when scored (no label leakage).

Public API:
    load_ca(pdb_path) -> dict{resnum: (np.array xyz, plddt)}
    coordination(resnum, coords, tree, radius) -> burial proxy (CA neighbour count)
    PPSCalculator(coords, plddt, path_residues, ben_residues, ...).score(resnum, exclude)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

PDB = Path("data/external/AF-P12883-F1-model_v6.pdb")


def load_ca(pdb_path: Path = PDB):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("m", str(pdb_path))
    out = {}
    for r in s[0]["A"]:
        if "CA" in r:
            ca = r["CA"]
            out[r.id[1]] = (ca.get_coord().astype(float), float(ca.get_bfactor()))
    return out


class PPSCalculator:
    """3D Pathogenic Proximity Score with a matched 1D control."""

    def __init__(self, ca: dict, path_residues: set, ben_residues: set,
                 radius=10.0, sigma=5.0, exclude_window=4, min_plddt=50.0,
                 lam=1.0, seq_sigma=10.0, seq_cutoff=30):
        self.resnums = np.array(sorted(ca.keys()))
        self.coords = np.array([ca[i][0] for i in self.resnums])
        self.plddt = np.array([ca[i][1] for i in self.resnums])
        self.idx = {int(r): k for k, r in enumerate(self.resnums)}
        self.tree = cKDTree(self.coords)
        self.path = set(int(x) for x in path_residues)
        self.ben = set(int(x) for x in ben_residues)
        self.radius, self.sigma = radius, sigma
        self.win, self.min_plddt, self.lam = exclude_window, min_plddt, lam
        self.seq_sigma, self.seq_cutoff = seq_sigma, seq_cutoff

    def _densities_3d(self, resnum, exclude):
        """Return (path_density, benign_density) in 3D for residue `resnum`."""
        if resnum not in self.idx:
            return np.nan, np.nan
        k = self.idx[resnum]
        neigh = self.tree.query_ball_point(self.coords[k], self.radius)
        pden = bden = 0.0
        for j in neigh:
            q = int(self.resnums[j])
            if q == resnum or q in exclude:
                continue
            if abs(q - resnum) <= self.win:      # non-local only
                continue
            if self.plddt[j] < self.min_plddt:   # gate low-confidence neighbours
                continue
            d = np.linalg.norm(self.coords[k] - self.coords[j])
            w = np.exp(-d * d / (2 * self.sigma ** 2)) * (self.plddt[j] / 100.0)
            if q in self.path:
                pden += w
            if q in self.ben:
                bden += w
        return pden, bden

    def _densities_1d(self, resnum, exclude):
        """Matched control: same idea but distance = sequence separation."""
        pden = bden = 0.0
        for q in range(resnum - self.seq_cutoff, resnum + self.seq_cutoff + 1):
            if q == resnum or q in exclude or q not in self.idx:
                continue
            if abs(q - resnum) <= self.win:
                continue
            sep = abs(q - resnum)
            w = np.exp(-sep * sep / (2 * self.seq_sigma ** 2))
            if q in self.path:
                pden += w
            if q in self.ben:
                bden += w
        return pden, bden

    def score(self, resnum, exclude=frozenset()):
        p3, b3 = self._densities_3d(resnum, exclude)
        p1, b1 = self._densities_1d(resnum, exclude)
        eps = 1e-9
        pps3 = p3 - self.lam * b3
        pps3_norm = (p3 - b3) / (p3 + b3 + eps)
        pps1 = p1 - self.lam * b1
        pps1_norm = (p1 - b1) / (p1 + b1 + eps)
        return {
            "pps3d": pps3, "pps3d_norm": pps3_norm,
            "path_density_3d": p3, "ben_density_3d": b3,
            "pps1d": pps1, "pps1d_norm": pps1_norm,
            "path_density_1d": p1, "ben_density_1d": b1,
            "plddt": float(self.plddt[self.idx[resnum]]) if resnum in self.idx else np.nan,
        }


def coordination(resnum, ca, tree, resnums, radius=10.0):
    """Burial proxy: number of CA atoms within `radius` (excluding self)."""
    if resnum not in ca:
        return np.nan
    k = list(resnums).index(resnum)
    return len(tree.query_ball_point(ca[resnum][0], radius)) - 1
