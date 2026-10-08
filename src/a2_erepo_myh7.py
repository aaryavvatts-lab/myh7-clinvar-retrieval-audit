"""v7 analysis 2 (roadmap 2.2): ClinGen Evidence Repository (public bulk TSV, no login) MYH7 curated calls matched to the cohort.
Output: results/tables/v7_erepo_myh7.csv (every erepo MYH7 missense assertion with its ClinVar aggregate/tier status),
v7_erepo_summary.json, and v7_erepo_availability.csv (which panel genes have VCEP rows at all).
"""
from __future__ import annotations
import json
import pandas as pd
from src.genes import *   # noqa
from src.mg_labels import load_erepo


def main():
    raw = pd.read_csv(RAW / "erepo_classifications_all.tsv", sep="\t", low_memory=False)
    av = raw.groupby("HGNC Gene Symbol").agg(n_rows=("Assertion", "size"), panels=("Expert Panel", lambda s: "; ".join(sorted(set(map(str, s)))))).reset_index()
    av["in_candidate_panel"] = av["HGNC Gene Symbol"].isin(GENES + EXTRA_GENES)
    av[av.in_candidate_panel].to_csv(TAB / "v7_erepo_availability.csv", index=False)
    e = load_erepo(); e = e[e.gene == "MYH7"]
    pt = pd.read_parquet(PROC / "MYH7" / "protein_table.parquet")
    m = e.merge(pt[["aaref", "aapos", "aaalt", "clinvar_protein_label", "stars_B_max", "stars_P_max", "lab_T1", "lab_T2", "lab_T3", "gn_af_legacy_max", "faf95_popmax"]],
                on=["aaref", "aapos", "aaalt"], how="left")
    m.to_csv(TAB / "v7_erepo_myh7.csv", index=False)
    s = {"erepo_total_rows_all_genes": int(len(raw)), "erepo_myh7_rows_all_variant_types": int((raw["HGNC Gene Symbol"] == "MYH7").sum()),
         "erepo_myh7_missense_rows": int(len(e)), "erepo_myh7_missense_by_class": e.erepo_class.value_counts().to_dict(),
         "erepo_myh7_B_in_cohort_as_clinvar_B": int(((m.erepo_class == "B") & (m.clinvar_protein_label == "B")).sum()),
         "erepo_myh7_B_in_T1": int(((m.erepo_class == "B") & (m.lab_T1 == "B")).sum()),
         "erepo_myh7_B_clinvar_other_label": m[(m.erepo_class == "B") & (m.clinvar_protein_label != "B")].clinvar_protein_label.fillna("none").value_counts().to_dict(),
         "erepo_myh7_P_in_cohort_as_clinvar_P": int(((m.erepo_class == "P") & (m.clinvar_protein_label == "P")).sum()),
         "erepo_myh7_P_clinvar_other_label": m[(m.erepo_class == "P") & (m.clinvar_protein_label != "P")].clinvar_protein_label.fillna("none").value_counts().to_dict(),
         "access": "GET https://erepo.clinicalgenome.org/evrepo/api/classifications/all (TSV, public, no authentication); the matchType/matchValue filter parameters are ignored by this endpoint, so the whole table is downloaded and filtered locally."}
    json.dump(s, open(TAB / "v7_erepo_summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
