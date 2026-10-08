"""v6 Phase 1b: assemble the modeling cohort with explicit label provenance.

Cohort = ClinVar missense SNVs (variant_summary) UNION gnomAD-observed missense variants.
Labels (same rules as v5):
  P   ClinVar aggregate Pathogenic / Likely pathogenic
  B   ClinVar Benign / Likely benign, OR gnomAD AF >= thr and not ClinVar-P and not ClinVar-VUS
  VUS ClinVar Uncertain significance or Conflicting
Two benign thresholds are built side by side: `label` (wider, AF >= 1e-5) and
`label_tight` (AF >= 5e-5). Variants with an invalid amino-acid position (outside
1..1935, stop-loss, missing) are dropped and counted.

AF source: gnomAD v4 (primary). `--af v2` rebuilds with gnomAD v2.1.1 (MyVariant); `--rule v5rcv` uses the v5
any-RCV-pathogenic-wins aggregation. Sensitivity cohorts go to data/spec_<rule>_<af>/.

Run: python -m src.build_labels [--af v4|v2]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed"); LOGS = Path("logs")
WIDE_AF, TIGHT_AF, STRICT_AF = 1e-5, 5e-5, 1e-4
MYH7_LEN = 1935


def spec_dir(af_source="v4", rule="aggregate"):
    return Path("data/processed") if (af_source, rule) == ("v4", "aggregate") else \
        Path(f"data/spec_{rule}_{af_source}")


def build(af_source="v4", rule="aggregate"):
    a = pd.read_parquet(OUT / "myh7_all_variants_annotated.parquet")
    if rule == "v5rcv":      # v5 any-RCV-pathogenic-wins aggregation (May-2025 MyVariant snapshot)
        a["y_aggregate"] = a["y"]
        has = a.in_clinvar & a.y_v5rcv.notna()
        a.loc[has, "y"] = a.loc[has, "y_v5rcv"]
        a.loc[has, "clinsig"] = a.loc[has, "clinsig_v5rcv"]
        a.loc[has, "stars"] = a.loc[has, "stars_v5rcv"]
        a.loc[a.in_clinvar & ~has, "y"] = "OTHER"
    acct = {"all_records": int(len(a))}
    a["gnomad_af"] = a["gnomad_af_v4"] if af_source == "v4" else a["gnomad_af_v2"]
    if af_source == "v2":
        # v2.1.1 sensitivity: gnomAD arm = variants observed in v2.1.1 (dbNSFP missense)
        gn_arm = a.gnomad_af_v2.notna() & a.aapos_dbnsfp.notna()
    else:
        gn_arm = a.in_gnomad_v4_missense & (a.ac_v4.fillna(0) > 0)
    cv_arm = a.is_missense_clinvar.fillna(False).astype(bool)
    df = a[cv_arm | gn_arm].copy()
    df["source"] = np.where(df.in_clinvar, "clinvar", "gnomad_only")
    acct["cohort_before_position_filter"] = int(len(df))
    acct["clinvar_missense_before_filter"] = int((df.source == "clinvar").sum())
    acct["gnomad_only_before_filter"] = int((df.source == "gnomad_only").sum())
    ok = df.aapos.notna() & df.aapos.between(1, MYH7_LEN) & df.aaref.notna() & df.aaalt.notna()
    acct["dropped_invalid_aapos"] = int((~ok).sum())
    acct["dropped_by_label"] = df.loc[~ok, "y"].value_counts().to_dict()
    df = df[ok].copy()
    df["aapos"] = df.aapos.astype(int)

    is_p = df.y.eq("P"); is_b = df.y.eq("B"); is_vus = df.y.eq("VUS")
    af = df.gnomad_af.fillna(0.0)
    for name, thr in (("label", WIDE_AF), ("label_tight", TIGHT_AF)):
        proxy = (af >= thr) & ~is_p & ~is_vus & ~is_b
        lab = pd.Series(pd.NA, index=df.index, dtype="object")
        lab[is_vus] = "VUS"; lab[proxy] = "B"; lab[is_b] = "B"; lab[is_p] = "P"
        df[name] = lab
        df[f"standin_{name}"] = proxy
    df["benign_source"] = np.where(df.label.eq("B") & is_b, "clinvar",
                                   np.where(df.label.eq("B"), "gnomad_proxy", None))
    df["benign_strict"] = df.label.eq("B") & (is_b | (af >= STRICT_AF))
    df["absent_from_gnomad"] = df.gnomad_af.isna()
    df["clinvar_class"] = df.y
    return df, acct


def main(af_source="v4", rule="aggregate"):
    df, acct = build(af_source, rule)
    dest = spec_dir(af_source, rule)
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "myh7_modeling.parquet", index=False)
    df.to_csv(dest / "myh7_modeling.csv", index=False)
    wide, tight = df.label, df.label_tight
    P = int((wide == "P").sum())
    s = df[df.label.eq("B") & df.benign_source.eq("gnomad_proxy")]
    s_t = df[df.label_tight.eq("B") & ~df.y.eq("B")]
    acct.update({
        "af_source": af_source, "label_rule": rule, "cohort_final": int(len(df)),
        "P": P, "P_by_source": df[wide.eq("P")].source.value_counts().to_dict(),
        "B_wide": int((wide == "B").sum()), "B_tight": int((tight == "B").sum()),
        "B_clinvar": int(df.y.eq("B").sum()),
        "B_standins_wide": int(len(s)), "B_standins_tight": int(len(s_t)),
        "standins_wide_AF_1e-5_to_5e-5": int(((s.gnomad_af >= 1e-5) & (s.gnomad_af < 5e-5)).sum()),
        "standins_wide_AF_ge_5e-5": int((s.gnomad_af >= 5e-5).sum()),
        "clinvar_B_with_gnomad_AF": int((df.y.eq("B") & df.gnomad_af.notna()).sum()),
        "benign_wide_AF_1e-5_to_5e-5_all": int(((df[wide.eq("B")].gnomad_af >= 1e-5) &
                                               (df[wide.eq("B")].gnomad_af < 5e-5)).sum()),
        "VUS": int((wide == "VUS").sum()),
        "VUS_conflicting": int(((wide == "VUS") & df.clinsig.eq("Conflicting")).sum()),
        "unlabelled_total": int(wide.isna().sum()),
        "unlabelled_gnomad_only": int((wide.isna() & df.source.eq("gnomad_only")).sum()),
        "unlabelled_clinvar_other": int((wide.isna() & df.source.eq("clinvar")).sum()),
        "P_2plus_stars": int((wide.eq("P") & (df.stars >= 2)).sum()),
        "B_clinvar_2plus_stars": int((df.y.eq("B") & (df.stars >= 2)).sum()),
        "P_residues": int(df[wide.eq("P")].aapos.nunique()),
        "B_residues_wide": int(df[wide.eq("B")].aapos.nunique()),
        "B_residues_tight": int(df[tight.eq("B")].aapos.nunique()),
        "absent_from_gnomad": int(df.absent_from_gnomad.sum()),
    })
    Path(LOGS).mkdir(exist_ok=True)
    (LOGS / f"cohort_accounting_{rule}_{af_source}.json").write_text(json.dumps(acct, indent=2, default=str))
    print(json.dumps(acct, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--af", default="v4", choices=["v4", "v2"])
    ap.add_argument("--rule", default="aggregate", choices=["aggregate", "v5rcv"])
    a = ap.parse_args(); main(a.af, a.rule)
