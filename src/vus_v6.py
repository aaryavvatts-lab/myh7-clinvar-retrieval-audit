"""v6 VUS analyses: discordance, ACMG points (with and without the support floor),
hotspot ("structural orphan") screen with support floor, coverage-aware VUS export.

Outputs: results/tables/v6_discordance_sensitivity.csv, v6_acmg_summary.csv,
v6_orphans_nofloor.csv, v6_orphans_floor.csv, vus_scored_v6.csv, top_prioritized_vus_v6.csv
Run: python -m src.vus_v6
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.common import *   # noqa
from src.analyze import annotate
from src.core3d import MIN_SUPPORT


def discordance(vus):
    rows = []
    for rt, ct, pt, nm in [(0.5, 20, 0.5, "nominal"), (0.644, 20, 0.5, "ClinGen PP3 supporting"),
                           (0.773, 25, 0.85, "stringent"), (0.290, 15, 0.15, "lenient")]:
        c = []
        for _, r in vus.iterrows():
            calls = []
            if pd.notna(r.sift): calls.append(r.sift < 0.05)
            if pd.notna(r.polyphen2_hvar): calls.append(r.polyphen2_hvar > pt)
            if pd.notna(r.revel): calls.append(r.revel > rt)
            if pd.notna(r.am_pred): calls.append(r.am_pred == "P")
            if pd.notna(r.cadd_phred): calls.append(r.cadd_phred > ct)
            if len(calls) >= 3: c.append(0 < sum(calls) < len(calls))
        rows.append({"thresholds": nm, "REVEL_cut": rt, "CADD_cut": ct, "PolyPhen2_cut": pt,
                     "SIFT_cut": 0.05, "AlphaMissense": "class P", "n_evaluable": len(c),
                     "pct_discordant": r3(100 * np.mean(c), 1)})
    return pd.DataFrame(rows)


def main():
    vus = pd.read_parquet("data/processed/vus_scored_model.parquet")
    vus["abstained"] = vus.pps3d_norm.isna()
    base = annotate(vus)
    fl = vus.copy(); fl.loc[fl.n_nb < MIN_SUPPORT, "pps3d_norm"] = np.nan   # PM1 needs >= 3 labelled neighbours
    flo = annotate(fl)
    base["acmg_points_floor"] = flo.acmg_points.values
    base["provisional_class_floor"] = flo.provisional_class.values
    base["acmg_codes_floor"] = flo.acmg_codes.values
    base["has_PM1"] = base.acmg_codes.str.contains("PM1")

    def counts(col):
        return base[col].value_counts().to_dict()
    lp = base[base.provisional_class == "Likely pathogenic"]
    lpf = base[base.provisional_class_floor == "Likely pathogenic"]
    P_or_LP = ["Pathogenic", "Likely pathogenic"]
    summ = {
        "n_vus": len(base),
        "classes_no_floor": counts("provisional_class"), "classes_with_floor": counts("provisional_class_floor"),
        "n_LP_no_floor": int(len(lp)), "n_LP_no_floor_with_lt3_neighbours": int((lp.n_nb < 3).sum()),
        "n_LP_no_floor_with_1_neighbour": int((lp.n_nb == 1).sum()),
        "n_LP_no_floor_without_PM1": int((~lp.has_PM1).sum()),
        "n_LP_with_floor": int(len(lpf)),
        "n_P_or_LP_no_floor": int(base.provisional_class.isin(P_or_LP).sum()),
        "n_P_or_LP_with_floor": int(base.provisional_class_floor.isin(P_or_LP).sum()),
        "n_towards_benign_no_floor": int(base.provisional_class.isin(["Likely benign", "Benign"]).sum()),
        "n_PM1_bearing_VUS": int(base.has_PM1.sum()),
        "n_PM1_bearing_lt3_neighbours": int((base.has_PM1 & (base.n_nb < 3)).sum()),
        "n_abstained_VUS": int(base.abstained.sum()), "pct_abstained_VUS": r3(100 * base.abstained.mean(), 1),
        "n_covered_ge3_VUS": int((base.n_nb >= 3).sum()),
        "triage": counts("triage"),
    }
    pd.Series(summ).to_csv(TAB / "v6_acmg_summary.csv", header=["value"])
    D = discordance(base); D.to_csv(TAB / "v6_discordance_sensitivity.csv", index=False)

    # ---- hotspot ("structural orphan") screen ----
    rare = base.absent_from_gnomad | (base.gnomad_af.fillna(0) < 1e-5)
    lenient = (base.sift >= 0.05) | (base.polyphen2_hvar < 0.5)
    linear_low = base.pps1d_norm.isna() | (base.pps1d_norm < 0.3)
    nofloor = base[(base.pps3d_norm >= 0.5) & linear_low & rare & lenient]
    floor = base[(base.n_path_nb >= MIN_SUPPORT) & (base.n_ben_nb == 0) & linear_low & rare & lenient]
    cols = ["hgvs_g", "hgvs_p", "aapos", "n_path_nb", "n_ben_nb", "pps3d_norm", "pps1d_norm", "RCS",
            "alphamissense", "am_pred", "sift", "polyphen2_hvar", "gnomad_af", "provisional_class_floor"]
    nofloor.sort_values("n_path_nb", ascending=False)[cols].to_csv(TAB / "v6_orphans_nofloor.csv", index=False)
    floor.sort_values("n_path_nb", ascending=False)[cols].to_csv(TAB / "v6_orphans_floor.csv", index=False)
    one = nofloor[nofloor.n_path_nb == 1]
    orph = {"orphans_no_floor": int(len(nofloor)), "orphans_with_floor_ge3": int(len(floor)),
            "orphans_no_floor_resting_on_single_neighbour": int(len(one)),
            "orphans_floor_residues": sorted(set(floor.aapos.astype(int))),
            "orphans_floor_AM_calls": floor.am_pred.value_counts().to_dict(),
            "orphans_floor_variants": floor.hgvs_p.astype(str).tolist()}
    pd.Series(orph).to_csv(TAB / "v6_orphan_summary.csv", header=["value"])

    base["orphan_no_floor"] = base.index.isin(nofloor.index); base["orphan_floor"] = base.index.isin(floor.index)
    # coverage-aware export (as promised in the manuscript)
    keep = ["hgvs_g", "hgvs_p", "aapos", "aaref", "aaalt", "clinsig", "clinsig_raw", "stars", "n_submitters",
            "gnomad_af", "gnomad_af_v2", "absent_from_gnomad", "RCS", "pps3d_norm", "abstained", "n_nb", "n_path_nb", "n_ben_nb",
            "covered_ge3", "pps1d_norm", "coordination", "plddt", "alphamissense", "am_pred", "revel", "sift",
            "polyphen2_hvar", "cadd_phred", "triage", "acmg_codes", "acmg_points", "provisional_class",
            "acmg_points_floor", "provisional_class_floor", "orphan_no_floor", "orphan_floor", "motor_domain", "converter"]
    keep = [c for c in keep if c in base.columns]
    out = base.sort_values(["acmg_points_floor", "RCS"], ascending=False)[keep]
    out.to_csv(TAB / "vus_scored_v6.csv", index=False)
    out.head(30).to_csv(TAB / "top_prioritized_vus_v6.csv", index=False)
    record(**{f"vus_{k}": v for k, v in {**summ, **orph}.items()})
    print(pd.Series({**summ, **orph}).to_string()); print(D.to_string())


if __name__ == "__main__":
    main()
