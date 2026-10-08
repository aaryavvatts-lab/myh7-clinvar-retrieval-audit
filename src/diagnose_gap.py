"""v6 root-cause diagnosis of the v5 ClinVar truncation + subset check of the old data.

Reads the frozen v5 snapshot in data/legacy_v5 (copied from the v5 project; SHA256SUMS). Writes results/tables/v6_diag_*.csv and
numbers into results/numbers.json.

Run: python -m src.diagnose_gap
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from src.common import *   # noqa

OLD = Path("data/legacy_v5")   # frozen minimal v5 snapshot (SHA256SUMS inside)
LOGS = Path("logs")


def main():
    a = pd.read_parquet("data/processed/myh7_all_variants_annotated.parquet")
    cv = a[a.in_clinvar].copy()
    cvm = cv[cv.is_missense_clinvar.fillna(False).astype(bool)].copy()
    old = pd.read_parquet(OLD / "myh7_clinvar_annotated.parquet")
    oldm = pd.read_parquet(OLD / "myh7_modeling.parquet")
    res = {}

    # (1) was anything lost to pagination? the old query's own hit count vs MyVariant total
    tot = requests.get("https://myvariant.info/v1/query",
                       params={"q": "clinvar.gene.symbol:MYH7", "size": 0}, timeout=60).json()["total"]
    res["old_query_total_reported_by_myvariant_now"] = int(tot)
    res["old_records_fetched"] = int(len(old))
    res["pagination_loss"] = bool(tot != len(old))

    # (2) which gene symbol does MyVariant store for ClinVar MYH7 missense records?
    sym = cvm.mv_clinvar_gene_symbol.fillna("None").astype(str)
    cat = np.where(sym.eq("MYH7"), "MYH7 (found by v5 query)",
          np.where(sym.eq("None"), "no gene symbol stored", "other symbol: " + sym))
    t = pd.Series(cat).value_counts().rename_axis("myvariant_clinvar_gene_symbol").reset_index(name="n_clinvar_missense")
    t.to_csv(TAB / "v6_diag_gene_symbol.csv", index=False)
    res["clinvar_missense_total"] = int(len(cvm))
    res["clinvar_missense_found_by_symbol_query"] = int(sym.eq("MYH7").sum())
    res["clinvar_missense_missed_by_symbol_query"] = int((~sym.eq("MYH7")).sum())
    res["missed_pct"] = r3(100 * (~sym.eq("MYH7")).mean(), 1)
    res["variant_summary_gene_symbol_for_missed"] = cvm.loc[~sym.eq("MYH7"), "gene_symbols_clinvar"].value_counts().head(8).to_dict()
    res["myvariant_notfound_clinvar_missense"] = int((~cvm.mv_found).sum())

    # (3) residue distribution old vs new (ClinVar missense only)
    oldmis = old[old.aapos.notna() & (old.aapos > 0)]
    blk = lambda s: pd.cut(s, range(0, 2001, 100)).astype(str)
    gap = pd.DataFrame({"old_clinvar_missense": oldmis.groupby(blk(oldmis.aapos)).size(),
                        "v6_clinvar_missense": cvm.groupby(blk(cvm.aapos_clinvar)).size()}).fillna(0).astype(int)
    gap["lost_pct"] = (100 * (1 - gap.old_clinvar_missense / gap.v6_clinvar_missense.clip(lower=1))).round(1)
    gap = gap.reindex(sorted(gap.index, key=lambda s: int(s.split(",")[0].strip("(")))); gap.to_csv(TAB / "v6_diag_residue_blocks.csv")
    lost = cvm[~sym.eq("MYH7")]
    res["lost_residue_range_p25_p75"] = [float(lost.aapos_clinvar.quantile(.25)), float(lost.aapos_clinvar.quantile(.75))]
    res["old_missing_763_893"] = int(oldmis.aapos.between(763, 893).sum())
    res["old_missing_1549_1770"] = int(oldmis.aapos.between(1549, 1770).sum())
    res["v6_clinvar_missense_763_893"] = int(cvm.aapos_clinvar.between(763, 893).sum())
    res["v6_clinvar_missense_1549_1770"] = int(cvm.aapos_clinvar.between(1549, 1770).sum())
    for nm, lo, hi in [("763_893", 763, 893), ("1549_1770", 1549, 1770)]:
        g = cvm[cvm.aapos_clinvar.between(lo, hi)]
        res[f"v6_hg19_range_{nm}"] = [int(g.pos19.min()), int(g.pos19.max())]
        res[f"v6_missed_share_{nm}_pct"] = r3(100 * (~g.mv_clinvar_gene_symbol.fillna('None').eq('MYH7')).mean(), 1)

    # (4) subset check
    oldids = set(old.hgvs_g)
    newids = set(cv.hgvs_g)
    res["old_records"] = len(oldids); res["v6_clinvar_snv_records"] = len(newids)
    res["old_in_v6_clinvar"] = len(oldids & newids); res["old_not_in_v6_clinvar"] = len(oldids - newids)
    res["old_fraction_of_v6_all_snv"] = r3(len(oldids & newids) / len(newids), 3)
    oldmis_ids = set(oldmis.hgvs_g)
    newmis_ids = set(cvm.hgvs_g)
    res["old_missense_n"] = len(oldmis_ids); res["v6_missense_n"] = len(newmis_ids)
    res["old_missense_in_v6_missense"] = len(oldmis_ids & newmis_ids)
    res["old_missense_not_in_v6_missense"] = len(oldmis_ids - newmis_ids)
    res["old_fraction_of_v6_missense"] = r3(len(oldmis_ids & newmis_ids) / len(newmis_ids), 3)
    nm = old[old.hgvs_g.isin(oldids - newids)][["hgvs_g", "clinvar_variant_id", "hgvs_p", "aapos", "clinsig", "y"]]
    nm.to_csv(TAB / "v6_diag_old_not_in_v6.csv", index=False)
    # what are the old records absent from the new SNV set?
    res["old_not_in_v6_non_snv_id"] = int((~nm.hgvs_g.str.match(r"^chr14:g\.\d+[ACGT]>[ACGT]$")).sum())
    # are the old SNV records that are missing from our MYH7 extraction in the release at all?
    import gzip
    snv_missing = nm[nm.hgvs_g.str.match(r"^chr14:g\.\d+[ACGT]>[ACGT]$")]
    want = set(str(int(v)) for v in snv_missing.clinvar_variant_id.dropna())
    found = set()
    with gzip.open("data/raw/variant_summary.txt.gz", "rt") as fh:
        next(fh)
        for line in fh:
            f = line.split("\t", 31)
            if f[30] in want:
                found.add(f[30])
    res["old_snv_not_in_v6"] = int(len(snv_missing))
    res["old_snv_not_in_v6_VariationID_absent_from_whole_release"] = int(len(want - found))
    res["old_missense_def_includes_stop_gain_aaalt_X"] = int((old.aaalt == "X").sum())
    res["old_aapos_sentinel_minus1"] = int((old.aapos == -1).sum())
    # new minus old: split by whether the May-2025 MyVariant snapshot holds the record
    newonly = cvm[~cvm.hgvs_g.isin(oldmis_ids)]
    inmv = newonly.mv_clinvar_variant_id.notna()
    res["v6_missense_not_in_v5"] = int(len(newonly))
    res["of_which_in_myvariant_snapshot_under_other_or_no_symbol"] = int(inmv.sum())
    res["of_which_absent_from_myvariant_snapshot_new_since_2025_05_or_unmapped"] = int((~inmv).sum())
    # old labelled counts as fraction of v6
    for lab_old, lab_new in [("P", "P"), ("B", "B"), ("VUS", "VUS")]:
        res[f"old_{lab_old}_n"] = int(oldmis.y.eq(lab_old).sum())
        res[f"v6_aggregate_{lab_new}_n"] = int(cvm.y.eq(lab_new).sum())
        res[f"v6_v5rule_{lab_new}_n"] = int(cvm.y_v5rcv.eq(lab_new).sum())

    # (5) label-rule check: re-applying the v5 any-RCV rule to the MyVariant records of the
    # OLD variants must reproduce the old labels (code validation of the replica)
    m = oldmis.merge(cv[["hgvs_g", "y_v5rcv", "y"]], on="hgvs_g", how="inner", suffixes=("_old", "_new"))
    res["v5_rule_replica_agreement_with_old_labels_pct"] = r3(100 * (m.y_old == m.y_v5rcv).mean(), 2)
    xt = pd.crosstab(m.y_old, m.y_new.fillna("absent")).rename_axis("v5_label", axis=0).rename_axis("v6_aggregate_label", axis=1)
    xt.to_csv(TAB / "v6_diag_label_crosstab_v5_vs_v6aggregate.csv")
    res["v5_P_now_conflicting_or_VUS_in_v6_aggregate"] = int(((m.y_old == "P") & (m.y_new == "VUS")).sum())
    res["v5_B_now_VUS_in_v6_aggregate"] = int(((m.y_old == "B") & (m.y_new == "VUS")).sum())

    # (6) contamination of v5 benign stand-ins: what does full ClinVar say about them?
    ob = oldm[(oldm.label == "B") & (oldm.benign_source == "gnomad_proxy")]
    x = ob.merge(a[["hgvs_g", "y", "clinsig", "in_clinvar"]], on="hgvs_g", how="left")
    res["v5_standins_n"] = int(len(ob))
    res["v5_standins_in_clinvar_now"] = int(x.in_clinvar.fillna(False).sum())
    res["v5_standins_clinvar_class_now"] = x.y_y.fillna("not in ClinVar variant_summary").value_counts().to_dict() if "y_y" in x else x.y.value_counts().to_dict()
    # explicit: stand-ins that full ClinVar calls P or VUS (cannot be benign by v5's own rule)
    cvcls = x["y_y"] if "y_y" in x else x["y"]
    res["v5_standins_that_are_clinvar_P"] = int((cvcls == "P").sum())
    res["v5_standins_that_are_clinvar_VUS"] = int((cvcls == "VUS").sum())

    # (7) rebuilt-cohort accounting under the v5 rules
    (LOGS / "diagnosis_gap.json").write_text(json.dumps(res, indent=1, default=str))
    record(**{f"diag_{k}": v for k, v in res.items()})
    print(json.dumps(res, indent=1, default=str))
    print(t.to_string()); print(gap.to_string())


if __name__ == "__main__":
    main()
