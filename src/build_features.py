"""v6 Phase 2/3: structure features (exact 3D-PPS with abstention) for every variant.

Writes features.parquet for the primary cohort and for each sensitivity cohort.
Scores in the file use the WIDER benign label set (`label`); analyses that need the tighter
set recompute scores from `label_tight` through the same function (src.core3d).

Run: python -m src.build_features
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.core3d import Kernels, MIN_SUPPORT
from src.build_labels import spec_dir

_K = None


def kernels():
    global _K
    if _K is None:
        _K = Kernels()
    return _K


def add_scores(df: pd.DataFrame, labcol: str = "label", K: Kernels = None) -> pd.DataFrame:
    """Append exact 3D/1D PPS, support counts, burial using labels in `labcol`."""
    K = K or kernels()
    pm, bm = K.masks(df.loc[df[labcol].eq("P"), "aapos"], df.loc[df[labcol].eq("B"), "aapos"])
    sc = K.score(df["aapos"].values, pm, bm)
    out = df.copy()
    for k, v in sc.items():
        out[k] = v
    out["score_defined"] = out["pps3d_norm"].notna()
    out["covered"] = out["n_nb"] >= MIN_SUPPORT
    return out


def main():
    for af in ("v4", "v2"):
        for rule in ("aggregate", "v5rcv"):
            d = spec_dir(af, rule)
            if not (d / "myh7_modeling.parquet").exists():
                continue
            df = pd.read_parquet(d / "myh7_modeling.parquet")
            out = add_scores(df, "label")
            out["motor_domain"] = out.aapos.between(1, 780)
            out["converter"] = out.aapos.between(709, 777)
            out.to_parquet(d / "features.parquet", index=False)
            out.to_csv(d / "features.csv", index=False)
            lab = out[out.label.isin(["P", "B"])]
            print(f"[features] {af}/{rule}: rows={len(out)} labelled={len(lab)} "
                  f"scoreable(labelled)={int(lab.score_defined.sum())} "
                  f"abstain P={100*(~lab[lab.label.eq('P')].score_defined).mean():.1f}% "
                  f"B={100*(~lab[lab.label.eq('B')].score_defined).mean():.1f}%")


if __name__ == "__main__":
    main()
