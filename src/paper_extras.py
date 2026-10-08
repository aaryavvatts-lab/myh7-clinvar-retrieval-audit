"""Extra numbers needed by the v6 paper (Stage 2 write). Every number in the paper that is not in
an earlier table comes from here.

Writes (results/tables):
  v6_power_detectable_auroc.csv      minimum true AUROC detectable vs chance (80% power, one sided 5%)
  v6_tie_saturation.csv              pairwise decomposition of the 3D-PPS AUROC (greater / tied / smaller)
  v6_benign_variants.csv             every wider-set benign variant with source, AF, region, coverage
  v6_label_region_summary.csv        P / B / VUS counts and coverage by protein region
  v6_friedman_overlap.csv            overlap of the Friedman 2024 preprint-supplement variants with the v6 cohort
  v6_sensitivity_grid_ci.csv         AUROC (+ residue-cluster CI) for 3D-PPS, 1D, AM, REVEL in the 4 cohorts
  v6_extras_summary.json             scalar values quoted in the text

Run from the project's v6/ directory:  ../.venv/bin/python -m src.paper_extras
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, mannwhitneyu
from src.common import *   # noqa
from src.build_labels import spec_dir

REGIONS = [("motor domain 1-780", 1, 780), ("lever arm 781-843", 781, 843),
           ("S2 844-1200", 844, 1200), ("light meromyosin 1201-1935", 1201, 1935)]
MISSING = [("763-893", 763, 893), ("1549-1770", 1549, 1770)]


def region_of(p):
    for n, a, b in REGIONS:
        if a <= p <= b:
            return n
    return "other"


# ------------------------------------------------------------------ A. power
def hanley_mcneil_se(auc, nP, nB):
    q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (nP - 1) * (q1 - auc ** 2) + (nB - 1) * (q2 - auc ** 2)) / (nP * nB)
    return float(np.sqrt(v))


def sim_power(auc, nP, nB, reps=1500, seed=42):
    rng = np.random.default_rng(seed)
    d = np.sqrt(2) * norm.ppf(auc)          # binormal, unit variances
    hit = 0
    for _ in range(reps):
        x = rng.normal(d, 1, nP); y = rng.normal(0, 1, nB)
        p = mannwhitneyu(x, y, alternative="greater", method="asymptotic").pvalue
        hit += p < 0.05
    return hit / reps


def power_table():
    settings = [("primary 3D-PPS scoreable (wider)", 297, 7), ("primary, tighter AM/REVEL", 391, 27),
                ("primary, wider AM/REVEL", 391, 30), ("v5 label rule + gnomAD v4, 3D-PPS scoreable", 510, 35),
                ("v5 label rule + gnomAD v4, all labelled", 625, 55)]
    rows = []
    for name, nP, nB in settings:
        lo, hi = 0.5, 0.99
        grid = np.round(np.arange(0.52, 0.995, 0.02), 3)
        mdA = np.nan
        for a in grid:
            if sim_power(a, nP, nB) >= 0.8:
                mdA = a; break
        rows.append({"setting": name, "n_P": nP, "n_B": nB,
                     "min_detectable_auroc_80pct_power": mdA,
                     "power_at_auroc_0p6": round(sim_power(0.6, nP, nB), 2),
                     "power_at_auroc_0p7": round(sim_power(0.7, nP, nB), 2),
                     "power_at_auroc_0p8": round(sim_power(0.8, nP, nB), 2),
                     "hanley_mcneil_se_at_0p5": round(hanley_mcneil_se(0.5, nP, nB), 3),
                     "hanley_mcneil_95ci_halfwidth_at_0p9": round(1.96 * hanley_mcneil_se(0.9, nP, nB), 3),
                     "hanley_mcneil_95ci_halfwidth_at_0p5": round(1.96 * hanley_mcneil_se(0.5, nP, nB), 3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B. ties
def tie_table(lab):
    d = lab[lab.pps3d_norm.notna()]
    P = d[d.y == 1].pps3d_norm.values; B = d[d.y == 0].pps3d_norm.values
    gt = (P[:, None] > B[None, :]).mean(); eq = (P[:, None] == B[None, :]).mean(); lt = (P[:, None] < B[None, :]).mean()
    rows = [{"quantity": "scoreable P", "value": len(P)}, {"quantity": "scoreable B", "value": len(B)},
            {"quantity": "P(pathogenic score > benign score)", "value": round(gt, 4)},
            {"quantity": "P(tie)", "value": round(eq, 4)},
            {"quantity": "P(pathogenic score < benign score)", "value": round(lt, 4)},
            {"quantity": "AUROC = P(>) + 0.5 P(tie)", "value": round(gt + .5 * eq, 4)},
            {"quantity": "benign with score exactly 1.0", "value": int((B == 1.0).sum())},
            {"quantity": "pathogenic with score exactly 1.0", "value": int((P == 1.0).sum())},
            {"quantity": "pathogenic with score < 1.0", "value": int((P < 1.0).sum())},
            {"quantity": "mean score pathogenic", "value": round(P.mean(), 3)},
            {"quantity": "mean score benign", "value": round(B.mean(), 3)},
            {"quantity": "median score pathogenic", "value": round(float(np.median(P)), 3)}]
    # bootstrap-by-residue replicate spread: number of distinct AUROC values and their SD
    y = d.y.values; s = d.pps3d_norm.values; g = d.aapos.values
    b = boot_stats(lambda i: auroc(y[i], s[i]), y, g, n=3000)
    rows += [{"quantity": "bootstrap replicates used (of 3000)", "value": len(b)},
             {"quantity": "bootstrap SD of AUROC", "value": round(float(b.std()), 4)}]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C. benign / regions
def benign_tables(lab_all):
    df = lab_all
    d = df[df.label.isin(["P", "B", "VUS"])].copy()
    d["region"] = d.aapos.map(region_of)
    d["in_previously_missing"] = d.aapos.map(lambda p: any(a <= p <= b for _, a, b in MISSING))
    b = d[d.label == "B"].copy()
    b["source"] = np.where(b.benign_source == "clinvar", "ClinVar benign/likely benign", "frequency stand-in")
    cols = ["hgvs_p", "aapos", "aaref", "aaalt", "source", "gnomad_af", "region", "in_previously_missing",
            "n_nb", "n_path_nb", "n_ben_nb", "pps3d_norm", "alphamissense", "revel"]
    bt = b[[c for c in cols if c in b.columns]].sort_values("aapos")
    # region summary (primary labels, wider set)
    rows = []
    for rname, a, c in REGIONS:
        g = d[d.aapos.between(a, c)]
        n_res = c - a + 1
        row = {"region": rname, "residues": n_res}
        for lab in ("P", "B", "VUS"):
            row[f"n_{lab}"] = int((g.label == lab).sum())
        for lab in ("P", "B"):
            gg = g[g.label == lab]
            row[f"{lab}_pct_no_labelled_neighbour"] = round(100 * (gg.n_nb == 0).mean(), 1) if len(gg) else np.nan
        rows.append(row)
    rs = pd.DataFrame(rows)
    return bt, rs, d


# ------------------------------------------------------------------ D. Friedman
def friedman_overlap(d_all):
    f = pd.read_csv("data/external/friedman2024/friedman2024_scores.csv")
    f = f[f.hgvs_p.notna()]
    uniq = f.drop_duplicates("hgvs_p")[["hgvs_p", "aa_position", "ref_aa", "alt_aa"]]
    key = d_all[["hgvs_p", "label", "aapos"]].copy()
    m = uniq.merge(key, on="hgvs_p", how="left")
    res = {"friedman_distinct_substitutions": int(len(uniq)),
           "friedman_residues": sorted(int(x) for x in uniq.aa_position.unique()),
           "friedman_in_v6_cohort": int(m.label.notna().sum() + 0),
           "friedman_in_v6_labelled_P": int((m.label == "P").sum()),
           "friedman_in_v6_labelled_B": int((m.label == "B").sum()),
           "friedman_in_v6_VUS": int((m.label == "VUS").sum())}
    # also: any v6 cohort row at those residues
    at = d_all[d_all.aapos.isin(uniq.aa_position.unique())]
    res["v6_variants_at_those_5_residues"] = int(len(at))
    res["v6_labelled_P_or_B_at_those_5_residues"] = int(at.label.isin(["P", "B"]).sum())
    return m, res


# ------------------------------------------------------------------ E. sensitivity grid with CIs
GRID = [("full ClinVar aggregate, gnomAD v4 (primary)", "v4", "aggregate"),
        ("full data, v5 label rule, gnomAD v4", "v4", "v5rcv"),
        ("full ClinVar aggregate, gnomAD v2.1.1", "v2", "aggregate"),
        ("full data, v5 label rule, gnomAD v2.1.1", "v2", "v5rcv")]


def sensitivity_grid():
    rows = []
    for name, af, rule in GRID:
        df = load(spec_dir(af, rule))
        for setname in ("wider", "tighter"):
            lab = labelled(df, setname)
            y, g = lab.y.values, lab.aapos.values
            for pred, col in (("3D-PPS", "pps3d_norm"), ("1D control", "pps1d_norm"),
                              ("AlphaMissense", "alphamissense"), ("REVEL", "revel")):
                s = oriented(lab, col).values
                a, lo, hi, n = auroc_ci(y, s, g)
                m = ~np.isnan(s)
                rows.append({"cohort": name, "af_source": af, "label_rule": rule, "set": setname,
                             "predictor": pred, "n": n, "n_P": int(y[m].sum()), "n_B": int((1 - y[m]).sum()),
                             "auroc": r3(a), "ci_low": r3(lo), "ci_high": r3(hi)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ F. gap causes, label shift, n_nb double count
def gap_causes():
    a = pd.read_parquet("data/processed/myh7_all_variants_annotated.parquet")
    cv = a[a.is_missense_clinvar.fillna(False).astype(bool)].copy()
    sym = cv.mv_clinvar_gene_symbol.astype(object).where(cv.mv_clinvar_gene_symbol.notna(), "None").astype(str)
    found = cv[sym == "MYH7"]; other = cv[~sym.isin(["MYH7", "None"])]; none = cv[sym == "None"]
    ceil = float(found.clinvar_variant_id.max())
    reg = cv[cv.aapos_clinvar.between(763, 893) | cv.aapos_clinvar.between(1549, 1770)]
    rs = sym[reg.index]
    res = {"clinvar_missense": int(len(cv)), "found_by_symbol_query": int(len(found)),
           "missed_total": int(len(other) + len(none)), "missed_other_symbol": int(len(other)),
           "missed_other_symbol_LOC126861897": int((sym == "LOC126861897").sum()),
           "missed_other_symbol_LOC126861898": int((sym == "LOC126861898").sum()),
           "missed_other_symbol_MIR208B": int((sym == "MIR208B").sum()),
           "missed_no_myvariant_clinvar_record": int(len(none)),
           "no_record_with_mv_clinvar_id": int(none.mv_clinvar_variant_id.notna().sum()),
           "variationid_ceiling_among_found_and_other": float(max(found.clinvar_variant_id.max(), other.clinvar_variant_id.max())),
           "no_record_above_ceiling": int((none.clinvar_variant_id > ceil).sum()),
           "lost_region_variants": int(len(reg)), "lost_region_symbol_cause": int(rs.isin(["LOC126861897", "LOC126861898"]).sum()),
           "lost_region_snapshot_cause": int((rs == "None").sum())}
    # label-rule shift on the complete data (any-record cohort, gnomAD v4)
    m = pd.read_parquet(spec_dir("v4", "v5rcv") / "myh7_modeling.parquet"); cvm = m[m.source == "clinvar"]
    ag = pd.read_parquet("data/processed/myh7_modeling.parquet")
    p5 = cvm[cvm.y == "P"]; b5 = cvm[cvm.y == "B"]
    cs = a.set_index("hgvs_g").clinsig
    x = p5[p5.y_aggregate == "VUS"]; y = b5[b5.y_aggregate == "VUS"]
    agp = ag[ag.y == "P"]
    new = agp[~agp.hgvs_g.isin(p5.hgvs_g)]
    res.update({"anyrecord_P": int(len(p5)), "anyrecord_P_stay_P": int((p5.y_aggregate == "P").sum()),
                "anyrecord_P_now_VUS": int(len(x)), "anyrecord_P_now_conflicting": int((cs.loc[x.hgvs_g] == "Conflicting").sum()),
                "anyrecord_P_now_uncertain": int((cs.loc[x.hgvs_g] == "Uncertain significance").sum()),
                "anyrecord_B": int(len(b5)), "anyrecord_B_now_VUS": int(len(y)),
                "anyrecord_B_now_conflicting": int((cs.loc[y.hgvs_g] == "Conflicting").sum()),
                "aggregate_P": int(len(agp)), "aggregate_P_not_anyrecord_P": int(len(new)),
                "aggregate_P_not_anyrecord_P_without_snapshot_label": int(new.hgvs_g.isin(cvm[cvm.y == "OTHER"].hgvs_g).sum())})
    return res


def n_nb_double_count():
    from src.build_features import kernels
    K = kernels(); rows = []
    for nm, (af, rule) in {"primary": ("v4", "aggregate"), "any-record, v4": ("v4", "v5rcv"),
                           "aggregate, v2.1.1": ("v2", "aggregate"), "any-record, v2.1.1": ("v2", "v5rcv")}.items():
        df = load(spec_dir(af, rule))
        for setn, col in (("wider", "label"), ("tighter", "label_tight")):
            d = df.copy(); d["lab"] = d[col]
            pm, bm = K.masks(d.loc[d.lab.eq("P"), "aapos"], d.loc[d.lab.eq("B"), "aapos"])
            ri = d.aapos.values.astype(int) - 1
            new = K.A[ri] @ np.maximum(pm, bm); old = K.A[ri] @ pm + K.A[ri] @ bm
            ll = d.lab.isin(["P", "B"]).values
            rows.append({"cohort": nm, "set": setn, "residues_labelled_both_P_and_B": int(((pm > 0) & (bm > 0)).sum()),
                         "variants_n_nb_changes": int((new != old).sum()), "labelled_variants_n_nb_changes": int(((new != old) & ll).sum()),
                         "labelled_variants_ge3_flag_changes": int((((new >= 3) != (old >= 3)) & ll).sum()),
                         "labelled_variants_zero_vs_nonzero_flips": int((((new == 0) != (old == 0)) & ll).sum())})
    return pd.DataFrame(rows)


def main():
    out = {}
    gc = gap_causes(); (TAB.parent / "v6_gap_causes.json").write_text(json.dumps(gc, indent=1)); print(gc)
    dc = n_nb_double_count(); dc.to_csv(TAB / "v6_n_nb_double_count.csv", index=False); print(dc.to_string())
    pt = power_table(); pt.to_csv(TAB / "v6_power_detectable_auroc.csv", index=False)
    print(pt.to_string())

    df = load("data/processed")
    lab = labelled(df, "wider")
    tt = tie_table(lab); tt.to_csv(TAB / "v6_tie_saturation.csv", index=False); print(tt.to_string())

    bt, rs, d_all = benign_tables(df)
    bt.to_csv(TAB / "v6_benign_variants.csv", index=False); rs.to_csv(TAB / "v6_label_region_summary.csv", index=False)
    print(bt.to_string()); print(rs.to_string())
    bsum = {"benign_total": int(len(bt)),
            "benign_in_previously_missing_regions": int(bt.in_previously_missing.sum()),
            "benign_clinvar": int((bt.source.str.startswith("ClinVar")).sum()),
            "benign_standin": int((bt.source == "frequency stand-in").sum()),
            "benign_motor_domain": int((bt.region == "motor domain 1-780").sum()),
            "benign_scoreable": int(bt.pps3d_norm.notna().sum()),
            "benign_residues_distinct": int(bt.aapos.nunique()),
            "benign_with_gnomad_af": int(bt.gnomad_af.notna().sum()),
            "benign_standin_af_min": float(bt[bt.source == "frequency stand-in"].gnomad_af.min()),
            "benign_standin_af_max": float(bt[bt.source == "frequency stand-in"].gnomad_af.max())}
    pm = d_all[d_all.in_previously_missing]
    bsum.update({"previously_missing_labelled_P": int((pm.label == "P").sum()),
                 "previously_missing_labelled_B": int((pm.label == "B").sum()),
                 "previously_missing_VUS": int((pm.label == "VUS").sum())})
    out.update(bsum)

    fm, fres = friedman_overlap(df)
    fm.to_csv(TAB / "v6_friedman_overlap.csv", index=False); out.update(fres); print(fres)

    sg = sensitivity_grid(); sg.to_csv(TAB / "v6_sensitivity_grid_ci.csv", index=False); print(sg.to_string())

    out["power_min_detectable_n_B_7"] = float(pt.iloc[0].min_detectable_auroc_80pct_power)
    out["power_min_detectable_n_B_30"] = float(pt.iloc[2].min_detectable_auroc_80pct_power)
    (TAB.parent / "v6_extras_summary.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
