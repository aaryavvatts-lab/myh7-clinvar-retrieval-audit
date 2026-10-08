"""Generate results/NUMBERS_v7.md: every quantity the manuscript will need, mapped to the file (and row/column) it comes from.
All values are read from results/tables/*.csv|json at run time. Run: python -m src.numbers_v7
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.genes import *   # noqa

ROWS = []


def q(section, label, value, source, where=""):
    if isinstance(value, (float, np.floating)):
        value = f"{value:.4g}"
    ROWS.append((section, label, str(value), source, where))


def T(f):
    return pd.read_csv(TAB / f)


def one(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df.iloc[0]


def main():
    # ---- 1 inclusion / labels / tiers
    inc = T("v7_inclusion.csv")
    q("1 Gene panel", "candidate genes evaluated for inclusion (19 pre-listed + A1 pool passes)", len(inc), "v7_inclusion.csv")
    q("1 Gene panel", "genes passing I1-I4", ", ".join(inc[inc.included.astype(str).str.lower() == "true"].gene), "v7_inclusion.csv", "included==True")
    for _, r in inc.iterrows():
        q("1 Gene panel", f"{r.gene}: P/B (T2 protein changes), included, first failed", f"{r.n_P_T2}/{r.n_B_T2}, {r.included}, {r.failed if isinstance(r.failed, str) else '-'}", "v7_inclusion.csv", f"gene={r.gene}")
    ps = T("v7_pool_screen.csv")
    q("1 Gene panel", "top-100 pool genes failing I1 (before AlphaFold check)", int((ps.first_failed == "I1").sum()), "v7_pool_screen.csv")
    q("1 Gene panel", "top-100 pool genes passing I1 but failing I2", int(ps.first_failed.astype(str).str.startswith("I2").sum()), "v7_pool_screen.csv")
    la = T("v7_label_accounting.csv"); m = la[la.gene == "MYH7"].iloc[0]
    for c in la.columns[1:]:
        q("2 Benign tiers (MYH7)", f"label accounting: {c}", m[c], "v7_label_accounting.csv", "gene=MYH7")
    ts = T("v7_tier_summary_myh7.csv")
    for _, r in ts.iterrows():
        q("2 Benign tiers (MYH7)", f"{r.tier}: P / B / B residues / scoreable B (S0) / B with no labelled neighbour %", f"{r.n_P} / {r.n_B} / {r.B_residues} / {r.n_B_scoreable_S0} / {r.B_no_labelled_neighbour_pct}", "v7_tier_summary_myh7.csv", f"tier={r.tier}")
    af = json.load(open(TAB / "v7_myh7_af_rule_counts.json"))
    for k, v in af.items():
        q("2 Benign tiers (MYH7)", f"AF rule count: {k}", v, "v7_myh7_af_rule_counts.json")
    es = json.load(open(TAB / "v7_erepo_summary.json"))
    for k, v in es.items():
        if k != "access":
            q("2 Benign tiers (MYH7)", f"ClinGen erepo: {k}", v, "v7_erepo_summary.json")
    q("2 Benign tiers (MYH7)", "ClinGen erepo access", es["access"], "v7_erepo_summary.json")
    av = T("v7_erepo_availability.csv")
    q("2 Benign tiers (MYH7)", "candidate genes with any ClinGen erepo VCEP rows", ", ".join(f"{r['HGNC Gene Symbol']}({r.n_rows})" for _, r in av.iterrows()), "v7_erepo_availability.csv")

    # ---- 3 estimand grid
    G = T("v7_estimand_grid.csv")
    for ds in ("T2", "T1", "T3", "T3legacy_wide", "T3legacy_tight", "v6_agg_v4_wider", "v6_v5rcv_v4", "v5_truncated"):
        for _, r in G[G.dataset == ds].iterrows():
            q("3 Estimand grid (MYH7)", f"{ds} | {r.score}: AUROC (95% CI), tie fraction, P(P>B), n, B", f"{r.auroc:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f}); ties {r.tie_fraction:.3f}; P(P>B) {r.p_pair_greater:.3f}; n={r.n_variants}, B={r.n_B}", "v7_estimand_grid.csv", f"dataset={ds}, score={r.score}")
    K = T("v7_estimand_k_sensitivity.csv")
    for _, r in K.iterrows():
        q("3 Estimand grid (MYH7)", f"S1/S2 pseudo-count k={r.k} ({r.score}), T2: AUROC (CI), ties", f"{r.auroc:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f}); ties {r.tie_fraction:.3f}", "v7_estimand_k_sensitivity.csv", f"k={r.k}, score={r.score}")
    P = T("v7_contrasts_paired.csv")
    for _, r in P[P.dataset.isin(["T2", "T1", "T3legacy_tight"])].iterrows():
        q("4 Paired contrasts (MYH7)", f"{r.dataset}: {r.A} minus {r.B}", f"{r.delta:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f}); MDD {r.mdd_80:.3f}; p={r.p_two_sided_boot:.4f}", "v7_contrasts_paired.csv", f"dataset={r.dataset}")
    F3 = T("v7_contrasts_F3.csv")
    for _, r in F3.iterrows():
        q("4 Paired contrasts (MYH7)", f"F3: {r.test}", f"{r.delta:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f}); raw p {r.p_two_sided_boot:.4f}; Holm p {r.p_holm_F3:.3f}", "v7_contrasts_F3.csv", f"test={r.test}")
    for f, lab in (("v7_contrasts_between_sets.csv", "shared variants"), ("v7_contrasts_union_sets.csv", "union (composition + labels)")):
        if (TAB / f).exists():
            for _, r in T(f).iterrows():
                q("4 Paired contrasts (MYH7)", f"{lab}: {r.contrast} [{r.score}]", f"{r.delta:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f}); AUROC A {r.auroc_A}, B {r.auroc_B}", f, f"contrast={r.contrast}, score={r.score}")
    J = T("v7_jackknife_primary.csv")
    for _, r in J.iterrows():
        q("5 Inference (MYH7)", f"jackknife over benign clusters {r.tier}: {r.quantity}", f"{r.estimate:+.3f}; SE {r.jackknife_se}; t({r.df}) CI {r.ci_low_t} to {r.ci_high_t}", "v7_jackknife_primary.csv", f"tier={r.tier}")
    PM = T("v7_perm_tests.csv")
    for _, r in PM.iterrows():
        q("5 Inference (MYH7)", f"permutation {r.tier} | {r.score} | {r.null}: observed, null mean (SD), p, Holm(F1)", f"{r.observed:.3f}; {r.null_mean:.3f} ({r.null_sd:.3f}); p={r.p_one_sided:.4f}; {r.p_holm_F1 if pd.notna(r.p_holm_F1) else '-'}", "v7_perm_tests.csv", f"tier={r.tier}, score={r.score}, null={r.null}")
    pw = T("v7_power_summary_myh7.csv")
    for _, r in pw.iterrows():
        q("5 Inference (MYH7)", f"simulation power {r.tier} | {r.score}: B observed, min detectable true AUROC (80%), B needed for 80% at AUROC 0.7 / 0.8", f"{r.n_B_observed}; {r.min_detectable_true_auroc_80pct}; {r.get('benign_n_needed_for_80pct_power_at_auroc_0.7')} / {r.get('benign_n_needed_for_80pct_power_at_auroc_0.8')}", "v7_power_summary_myh7.csv", f"tier={r.tier}, score={r.score}")
    cv = T("v7_cv_nested_contrasts.csv")
    for _, r in cv.iterrows():
        q("5 Inference (MYH7)", f"nested CV {r.tier}/{r.grouping}: {r.question}", f"{r.delta_mean:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f}); SD over repeats {r.delta_sd_across_repeats}; positive in {r.n_repeats_positive}/20; MDD {r.mdd_80}; TOST p {r['p_tost_0.02']:.4f}", "v7_cv_nested_contrasts.csv", f"tier={r.tier}, grouping={r.grouping}")
    cs = T("v7_cv_nested_summary.csv")
    for _, r in cs.iterrows():
        q("5 Inference (MYH7)", f"nested CV {r.tier}/{r.grouping}: AUROC of {r.model}", f"{r.auroc_mean:.3f} (SD over 20 repeats {r.auroc_sd_across_repeats})", "v7_cv_nested_summary.csv", f"tier={r.tier}, grouping={r.grouping}")
    ls = T("v7_cv_legacy_single_split.csv")
    for _, r in ls.iterrows():
        q("5 Inference (MYH7)", f"legacy single-split {r.tier}: {r.question}", f"{r.delta:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f})", "v7_cv_legacy_single_split.csv", f"tier={r.tier}")
    ci = T("v7_coverage_inference_myh7.csv")
    for _, r in ci.iterrows():
        q("5 Inference (MYH7)", f"coverage association {r.tier}, >= {r.min_neighbours} neighbours: exact OR (CI), Haldane OR (CI), wild cluster p, CR2 LPM coef (CI) p",
          f"{r.or_conditional_mle:.2f} ({r.or_exact_ci_low:.2f} to {r.or_exact_ci_high:.2f}); {r.or_haldane:.2f} ({r.or_haldane_ci_low:.2f} to {r.or_haldane_ci_high:.2f}); {r.wild_cluster_p:.4f}; {r.lpm_coef:.3f} ({r.lpm_ci_low:.3f} to {r.lpm_ci_high:.3f}) p={r.lpm_p:.4f}; B covered {r.B_covered}/{r.B_covered + r.B_uncovered}",
          "v7_coverage_inference_myh7.csv", f"tier={r.tier}, min_neighbours={r.min_neighbours}")
    cr = T("v7_pm1_call_rates.csv")
    for _, r in cr.iterrows():
        q("5 Inference (MYH7)", f"S0 rule call rate {r.tier}, S0>={r.threshold}, floor {r.support_floor}, class {r['class']}", f"{r.n_called}/{r.n} = {r.rate:.3f} (exact CI {r.ci_low:.3f} to {r.ci_high:.3f}); LR+ {r.LR_plus}", "v7_pm1_call_rates.csv", f"tier={r.tier}")
    sw = T("v7_param_sweep_myh7.csv")
    for _, r in sw[sw.score.isin(["S1", "S0_neutral"])].iterrows():
        q("5 Inference (MYH7)", f"parameter sweep {r.param}={r.value}{' (default)' if r.is_default else ''}, {r.score}", f"{r.auroc:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f})", "v7_param_sweep_myh7.csv", f"param={r.param}")
    afs = T("v7_af_strata_myh7.csv")
    for _, r in afs.iterrows():
        q("2 Benign tiers (MYH7)", f"AF stratum: {r['class']} | {r.af_bin}", f"{r.n} ({100 * r.share_of_class:.1f}% of class)", "v7_af_strata_myh7.csv")

    # ---- 6 retrieval audit
    ra = T("v7_retrieval_audit_gene.csv")
    for _, r in ra[ra.in_panel_A].iterrows():
        q("6 Retrieval audit", f"{r.gene}: ClinVar missense, returned, missed (%), symbol-field, snapshot-missing (newer), other, longest >=90%-loss run",
          f"{r.n_clinvar_missense}, {r.n_returned}, {r.n_missed} ({r.loss_pct:.1f}%), {r.n_symbol_field}, {r.n_snapshot_missing} ({r.n_snapshot_missing_newer}), {r.n_other}, {r.longest_run_residues if isinstance(r.longest_run_residues, str) else '-'}", "v7_retrieval_audit_gene.csv", f"gene={r.gene}")
    rj = json.load(open(TAB / "v7_retrieval_audit_summary.json"))
    for k, v in rj.items():
        q("6 Retrieval audit", f"summary: {k}", json.dumps(v) if isinstance(v, dict) else v, "v7_retrieval_audit_summary.json")

    # ---- 7 multi-gene
    gs = T("v7_gene_summary.csv")
    for _, r in gs[gs.tier == "T2"].iterrows():
        q("7 Multi-gene (T2)", f"{r.gene}: P, B, B residues, scoreable B(S0), S0 tie fraction, benign no-neighbour, pathogenic no-neighbour",
          f"{r.n_P}, {r.n_B}, {r.B_residues}, {r.n_B_scoreable_S0}, {r.S0_tie_fraction}, {r.benign_no_neighbour}, {r.pathogenic_no_neighbour}", "v7_gene_summary.csv", f"gene={r.gene}, tier=T2")
        q("7 Multi-gene (T2)", f"{r.gene}: AUROC S1 / S2 / S0 neutral / count / 1D S1 / AM / REVEL",
          f"{r.auroc_S1} / {r.auroc_S2} / {r.auroc_S0_neutral} / {r.auroc_n_nb} / {r.auroc_S1_1d} / {r.auroc_AlphaMissense} / {r.auroc_REVEL}", "v7_gene_summary.csv", f"gene={r.gene}, tier=T2")
        q("7 Multi-gene (T2)", f"{r.gene}: S1 minus AM, S1 minus count (paired)", f"{r.delta_S1_minus_o_alphamissense}; {r.delta_S1_minus_n_nb}", "v7_gene_summary.csv / v7_gene_pairs.csv", f"gene={r.gene}")
        q("7 Multi-gene (T2)", f"{r.gene}: null-2 p (S1), Holm F4; nested M3-M1 (CI), MDD, Holm F6", f"{r.S1_null2_p}, {r.S1_null2_p_holm_F4:.3f}; {r.cv_M3_minus_M1} ({r.cv_M3_minus_M1_ci}), {r.cv_M3_minus_M1_mdd}, {r.cv_M3_minus_M1_p_holm_F6:.3f}", "v7_gene_summary.csv", f"gene={r.gene}")
    mt = T("v7_meta_pooled.csv")
    for _, r in mt.iterrows():
        q("7 Multi-gene (T2)", f"meta [{r.set}] {r.A} minus {r.B}: k, pooled (HK CI), tau, I2, Q p, PI", f"{int(r.k)}, {r.pooled:+.3f} ({r.ci_low_hk:+.3f} to {r.ci_high_hk:+.3f}), {r.tau:.3f}, {r.I2_pct:.0f}%, {r.Q_p:.3f}, {r.pred_low:+.3f} to {r.pred_high:+.3f}", "v7_meta_pooled.csv", f"set={r.set}")
    mr = T("v7_meta_regression.csv")
    for _, r in mr.iterrows():
        q("7 Multi-gene (T2)", f"meta-regression S1 minus {r.B} on {r.moderator}", f"slope {r.slope:+.3f} (SE {r.slope_se:.3f}), p={r.p_wls:.3f}; Spearman {r.spearman_rho:+.2f} (p={r.spearman_p:.3f})", "v7_meta_regression.csv")
    asp = T("v7_across_gene_spearman.csv")
    for _, r in asp.iterrows():
        q("7 Multi-gene (T2)", f"across-gene Spearman {r.outcome} ~ {r.moderator}", f"rho {r.spearman_rho:+.2f}, p={r.p:.3f}, k={int(r.k)}", "v7_across_gene_spearman.csv")

    # ---- 8 reproducibility
    tf = Path("logs/run_all_timings.json")
    if tf.exists():
        for k, v in json.load(open(tf)).items():
            q("8 Reproducibility", f"run_all timing: {k}", f"{v} s", "logs/run_all_timings.json")
    for f in ("logs/prereg_sha256.txt", "logs/prereg_sha256_after_A1.txt", "logs/prereg_sha256_after_D1-D5.txt"):
        if Path(f).exists():
            q("8 Reproducibility", f"PREREG hash ({f})", Path(f).read_text().split()[0], f)
    out = ["# NUMBERS_v7: every quantity the manuscript needs, with its source file", "",
           "Generated by src/numbers_v7.py from results/tables (and logs/). Paths are relative to v7/. `where` gives the filter that selects the row/column. Nothing here is typed by hand.", ""]
    sec = None
    for s, label, v, src, where in ROWS:
        if s != sec:
            out += ["", f"## {s}", "", "| quantity | value | source | where |", "|---|---|---|---|"]; sec = s
        out.append(f"| {label} | {v} | {src} | {where} |")
    (Path("results") / "NUMBERS_v7.md").write_text("\n".join(out) + "\n")
    print(f"NUMBERS_v7.md: {len(ROWS)} quantities")


if __name__ == "__main__":
    main()
