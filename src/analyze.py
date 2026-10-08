"""Phase 5/6 — discordance triage, ACMG/Bayesian classification, structural orphans.

* Discordance triage: per-variant agreement across SIFT, PolyPhen-2, REVEL, AlphaMissense,
  CADD -> concordant-damaging / concordant-tolerated / DISCORDANT (the clinically hard set).
* ACMG codes with ClinGen-SVI-calibrated in-silico strengths (Pejaver et al. 2022 REVEL
  thresholds) + PM2 (rarity) + PM1 (3D hotspot via 3D-PPS) + BS1/BA1 (population) ->
  Tavtigian et al. 2018 Bayesian points -> provisional classification.
* Structural orphans: VUS that are ultra-rare, buried in a 3D pathogenic hotspot (high
  3D-PPS) but NOT in a linear cluster (low 1D-PPS) AND called tolerated by legacy linear
  tools (SIFT/PolyPhen). These are the variants only the 3D axis surfaces.

Run:  python -m src.analyze
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed")
TAB = Path("results/tables"); TAB.mkdir(parents=True, exist_ok=True)

# ---- binary damaging calls per tool ----
def calls(r):
    c = {}
    c["SIFT"] = ("D" if r.sift is not None and r.sift < 0.05 else "T") if pd.notna(r.sift) else None
    c["PolyPhen2"] = ("D" if pd.notna(r.polyphen2_hvar) and r.polyphen2_hvar > 0.5 else "T") if pd.notna(r.polyphen2_hvar) else None
    c["REVEL"] = ("D" if pd.notna(r.revel) and r.revel > 0.5 else "T") if pd.notna(r.revel) else None
    c["AlphaMissense"] = ("D" if r.am_pred == "P" else "T") if pd.notna(r.am_pred) else None
    c["CADD"] = ("D" if pd.notna(r.cadd_phred) and r.cadd_phred > 20 else "T") if pd.notna(r.cadd_phred) else None
    return c


def triage_row(r):
    c = [v for v in calls(r).values() if v is not None]
    if len(c) < 3:
        return "insufficient"
    nd, nt = c.count("D"), c.count("T")
    if nt == 0:
        return "concordant-damaging"
    if nd == 0:
        return "concordant-tolerated"
    return "discordant"


# ---- ACMG evidence (points via Tavtigian 2018) ----
PTS = {"P_VeryStrong": 8, "P_Strong": 4, "P_Moderate": 2, "P_Supporting": 1,
       "B_Supporting": -1, "B_Moderate": -2, "B_Strong": -4, "B_Standalone": -8}


def acmg_codes(r):
    codes = {}
    af = 0.0 if pd.isna(r.gnomad_af) else r.gnomad_af
    # population
    if af >= 5e-4:
        codes["BA1"] = "B_Standalone"
    elif af >= 1e-4:
        codes["BS1"] = "B_Strong"
    if (pd.isna(r.gnomad_af)) or af < 1e-5:
        codes["PM2"] = "P_Supporting"
    # in-silico PP3/BP4 via calibrated REVEL (Pejaver 2022) + AlphaMissense concordance
    rv = r.revel
    if pd.notna(rv):
        if rv >= 0.932:
            codes["PP3"] = "P_Strong"
        elif rv >= 0.773:
            codes["PP3"] = "P_Moderate"
        elif rv >= 0.644:
            codes["PP3"] = "P_Supporting"
        elif rv <= 0.016:
            codes["BP4"] = "B_Strong"
        elif rv <= 0.183:
            codes["BP4"] = "B_Moderate"
        elif rv <= 0.290:
            codes["BP4"] = "B_Supporting"
    # AlphaMissense agreement can upgrade a supporting PP3 to moderate
    if r.am_pred == "P" and codes.get("PP3") == "P_Supporting":
        codes["PP3"] = "P_Moderate"
    # PM1 structural hotspot from 3D-PPS
    if pd.notna(r.pps3d_norm):
        if r.pps3d_norm >= 0.7:
            codes["PM1"] = "P_Moderate"
        elif r.pps3d_norm >= 0.5:
            codes["PM1"] = "P_Supporting"
    return codes


def classify(points, has_ba1):
    if has_ba1:
        return "Benign"
    if points >= 10:
        return "Pathogenic"
    if points >= 6:
        return "Likely pathogenic"
    if points <= -7:
        return "Benign"
    if points <= -1:
        return "Likely benign"
    return "VUS"


def annotate(df):
    df = df.copy()
    df["triage"] = df.apply(triage_row, axis=1)
    codes_list, pts_list, cls_list = [], [], []
    for _, r in df.iterrows():
        codes = acmg_codes(r)
        pts = sum(PTS[v] for v in codes.values())
        codes_list.append(";".join(f"{k}({v})" for k, v in codes.items()))
        pts_list.append(pts)
        cls_list.append(classify(pts, "BA1" in codes))
    df["acmg_codes"] = codes_list
    df["acmg_points"] = pts_list
    df["provisional_class"] = cls_list
    return df


def main():
    vus = pd.read_parquet(OUT / "vus_scored.parquet")
    lab = pd.read_parquet(OUT / "features.parquet")
    lab = lab[lab.label.isin(["P", "B"])]

    vus = annotate(vus)

    # triage distribution
    print("[analyze] VUS discordance triage:\n", vus["triage"].value_counts().to_string())
    print("\n[analyze] provisional reclassification of VUS:\n",
          vus["provisional_class"].value_counts().to_string())

    # ---- structural orphans ----
    orphan = (
        vus["label"].eq("VUS")
        & (vus["pps3d_norm"] >= 0.5)                 # in a 3D pathogenic hotspot
        & (vus["pps1d_norm"] < 0.3)                  # but NOT a linear cluster
        & (vus["absent_from_gnomad"] | (vus["gnomad_af"].fillna(0) < 1e-5))  # ultra-rare
        & ((vus["sift"] >= 0.05) | (vus["polyphen2_hvar"] < 0.5))            # legacy tools lenient
    )
    orphans = vus[orphan].sort_values("RCS", ascending=False)
    orphans.to_csv(TAB / "structural_orphans.csv", index=False)
    print(f"\n[analyze] STRUCTURAL ORPHANS (3D-flagged, linear-missed, ultra-rare): {len(orphans)}")
    cols = ["hgvs_p", "aapos", "RCS", "pps3d_norm", "pps1d_norm", "sift",
            "polyphen2_hvar", "alphamissense", "am_pred", "acmg_points", "provisional_class"]
    print(orphans[cols].head(15).to_string(index=False))

    # top prioritized VUS overall
    top = vus.sort_values(["acmg_points", "RCS"], ascending=False).head(30)
    keep = ["hgvs_p", "aapos", "RCS", "pps3d_norm", "triage", "gnomad_af",
            "alphamissense", "revel", "acmg_codes", "acmg_points", "provisional_class",
            "motor_domain", "converter"]
    top[keep].to_csv(TAB / "top_prioritized_vus.csv", index=False)
    vus.to_parquet(OUT / "vus_annotated.parquet", index=False)
    vus.to_csv(OUT / "vus_annotated.csv", index=False)

    print("\n[analyze] top prioritized VUS (by ACMG points then RCS):")
    print(top[["hgvs_p", "RCS", "pps3d_norm", "triage", "acmg_points",
               "provisional_class"]].head(15).to_string(index=False))
    print(f"\n[analyze] saved -> {TAB/'structural_orphans.csv'}, {TAB/'top_prioritized_vus.csv'}")


if __name__ == "__main__":
    main()
