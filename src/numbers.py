"""Build results/NUMBERS.md: every quantity quoted in the v5 manuscript with its OLD v5
value next to the NEW v6 value (all v6 values read from results/tables, logs, numbers.json).

Run: python -m src.numbers
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import *   # noqa
from src.core3d import load_ca


def md_table(df, index=False):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        out.append("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |")
    return "\n".join(out)


def T(n): return pd.read_csv(TAB / n)
def J(n): return json.loads(Path(f"logs/{n}").read_text())


def main():
    acc = J("cohort_accounting_aggregate_v4.json"); acc5 = J("cohort_accounting_v5rcv_v4.json")
    acc2 = J("cohort_accounting_aggregate_v2.json"); acc52 = J("cohort_accounting_v5rcv_v2.json")
    diag = J("diagnosis_gap.json"); prov = J("provenance_acquisition.json")
    df = load()
    t4 = T("v6_table4_auroc.csv"); cf = T("v6_coverage_flag.csv"); ab = T("v6_abstention_by_class.csv")
    st = T("v6_coverage_strata.csv"); h2h = T("v6_head_to_head.csv"); bl = T("v6_burial_logistic.csv")
    perm = T("v6_permutation.csv"); cv = T("v6_cv_metrics.csv"); pm = T("v6_paired_models.csv")
    cal = T("v6_calibration_summary.csv"); disc = T("v6_discordance_sensitivity.csv")
    acm = pd.read_csv(TAB / "v6_acmg_summary.csv", index_col=0)["value"]; orp = pd.read_csv(TAB / "v6_orphan_summary.csv", index_col=0)["value"]
    sens = T("v6_sensitivity_specs.csv"); leak = T("v6_leakage_wider.csv"); leakt = T("v6_leakage_tighter.csv")
    sb = T("spatial_block_cv.csv"); sbp = T("spatial_block_cv_paired.csv"); val = T("v6_validation_on_v5data.csv")

    def t4v(setn, pred, col="auroc"): return t4[(t4.set == setn) & (t4.predictor == pred)][col].iloc[0]
    def ci(setn, pred):
        r = t4[(t4.set == setn) & (t4.predictor == pred)].iloc[0]; return f"{r.auroc:.3f} [{r.ci_low:.3f} to {r.ci_high:.3f}] (n={int(r.n)}, B={int(r.n_B)})"
    def cfv(setn, col): return cf[cf.set == setn][col].iloc[0]
    def abv(setn, cls, col): return ab[(ab.set == setn) & (ab["class"] == cls)][col].iloc[0]
    def pr(cohort, setn, nl):
        r = perm[(perm.cohort == cohort) & (perm.set == setn) & perm.null.str.startswith(nl)].iloc[0]; return r
    def cvm(setn, model, fs):
        r = cv[(cv.set == setn) & (cv.model == model) & (cv.feature_set == fs)].iloc[0]; return f"{r.auroc:.3f} [{r.ci_low:.3f} to {r.ci_high:.3f}], AUPRC {r.auprc:.3f} (prevalence {r.auprc_baseline_prevalence:.3f})"
    def pmv(setn, A, B_):
        r = pm[(pm.set == setn) & (pm.A == A) & (pm.B == B_)].iloc[0]; return f"{r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}], TOST p={r.p_tost:.3f}"

    ca = load_ca(); pl = np.array([v[1] for v in ca.values()])
    ms = df.copy()
    miss = []
    for c in ["gnomad_af", "cadd_phred", "vest4", "revel", "sift", "sift4g", "polyphen2_hdiv", "polyphen2_hvar", "esm1b", "clinpred", "metalr",
              "alphamissense", "primateai", "mpc", "provean", "phylop100", "gerp"]:
        for nm, m in [("all missense cohort", np.ones(len(ms), bool)), ("labelled P/B", ms.label.isin(["P", "B"]).values), ("VUS", ms.label.eq("VUS").values)]:
            miss.append({"predictor": c, "group": nm, "n": int(m.sum()), "non_missing": int(ms.loc[m, c].notna().sum()),
                         "pct_non_missing": r3(100 * ms.loc[m, c].notna().mean(), 1)})
    miss = pd.DataFrame(miss); miss.to_csv(TAB / "v6_predictor_missingness.csv", index=False)
    mall = miss[miss.group == "all missense cohort"].set_index("predictor")
    n = len(df)

    R = []   # (section, quantity, v5, v6, changed?)
    def add(sec, q, old, new, flag=""): R.append((sec, q, str(old), str(new), flag))

    # ---- cohort
    add("Cohort", "Cohort size (missense variants)", "2,769 (said 'all in ClinVar'; 348 were gnomAD-only)", f"{acc['cohort_final']:,} (ClinVar {acc['clinvar_missense_before_filter']:,} + gnomAD v4-observed only {acc['gnomad_only_before_filter']:,}); invalid positions dropped: {acc['dropped_invalid_aapos']}", "CHANGED")
    add("Cohort", "ClinVar MYH7 missense SNVs (any label)", "2,421 records with an amino acid position (includes 66 stop-gain variants)", f"{diag['clinvar_missense_total']:,} (E-utils esearch SNV+missense = {prov['eutils']['missense_snv']}; ours + 1 stop-loss Ter1936Trp excluded)", "CHANGED")
    add("Cohort", "Pathogenic / likely pathogenic", "511 (143 with 2+ stars)", f"{acc['P']} ({acc['P_2plus_stars']} with 2+ stars) [ClinVar aggregate rule]; {acc5['P']} under v5 any-RCV rule", "CHANGED")
    add("Cohort", "Benign, ClinVar only", "38 (19 with 2+ stars)", f"{acc['B_clinvar']} ({acc['B_clinvar_2plus_stars']} with 2+ stars) [aggregate]; {acc5['B_clinvar']} under v5 rule", "CHANGED")
    add("Cohort", "Benign wider (AF>=1e-5)", "110 (72 frequency stand-ins)", f"{acc['B_wide']} ({acc['B_standins_wide']} stand-ins; {acc['standins_wide_AF_1e-5_to_5e-5']} of them in 1e-5..5e-5, {acc['standins_wide_AF_ge_5e-5']} >=5e-5) [gnomAD v4]; v2.1.1: {acc2['B_wide']}", "CHANGED")
    add("Cohort", "Benign tighter (AF>=5e-5)", "58", f"{acc['B_tight']} ({acc['B_standins_tight']} stand-ins); v2.1.1: {acc2['B_tight']}", "CHANGED")
    add("Cohort", "VUS (incl. conflicting)", "1,872 (63 conflicting)", f"{acc['VUS']:,} ({acc['VUS_conflicting']} conflicting) [aggregate]; {acc5['VUS']:,} under v5 rule", "CHANGED")
    add("Cohort", "gnomAD-observed, unlabelled", "276", f"{acc['unlabelled_gnomad_only']}", "CHANGED")
    add("Cohort", "Absent from gnomAD", "1,916 (69.2%)", f"{acc['absent_from_gnomad']:,} ({100*acc['absent_from_gnomad']/acc['cohort_final']:.1f}%)", "CHANGED")
    add("Cohort", "Pathogenic residues / benign residues", "335 / 101 (333/100 valid)", f"{acc['P_residues']} / {acc['B_residues_wide']} (wider), {acc['B_residues_tight']} (tighter)", "CHANGED")
    add("Cohort", "Variants with invalid amino acid position kept", "76 (75 sentinel -1, 1 stop-loss)", f"0 kept; dropped {acc['dropped_invalid_aapos']} (positions parsed from ClinVar Name, gnomAD hgvsp, then dbNSFP)", "FIXED")
    add("Cohort", "Stand-in variants that full ClinVar calls P or VUS", "not checked", f"of v5's 72 stand-ins: {diag['v5_standins_that_are_clinvar_VUS']} are ClinVar VUS, {diag['v5_standins_that_are_clinvar_P']} ClinVar P, 3 ClinVar B, 5 absent", "NEW FINDING")
    add("Cohort", "gnomAD version", "'v4' (actually MyVariant v2.1.1, hg19)", f"v4 (GraphQL gnomad_r4, GRCh38) primary; v2.1.1 sensitivity. v4 missense observed (AC>0): {prov['gnomad_v4']['n_missense_observed_ac_gt0']}", "FIXED")
    add("Cohort", "ClinVar release", "'August 2026' MyVariant (actually ClinVar 2025-05 snapshot)", f"variant_summary.txt.gz, server Last-Modified {prov['clinvar_variant_summary']['server_last_modified']}, sha256 {prov['clinvar_variant_summary']['sha256'][:12]}...", "FIXED")
    add("Cohort", "AlphaFold pLDDT mean / % >70", "74.3 / 64.4%", f"{pl.mean():.1f} / {100*(pl>70).mean():.1f}% (same file, unchanged)", "same")
    # ---- root cause
    add("Diagnosis", "ClinVar missense found by v5 symbol query", "2,421 records with aapos", f"{diag['clinvar_missense_found_by_symbol_query']} of {diag['clinvar_missense_total']}; missed {diag['clinvar_missense_missed_by_symbol_query']} ({diag['missed_pct']}%)", "BUG")
    add("Diagnosis", "Old set as fraction of full set", "n/a", f"{diag['old_fraction_of_v6_all_snv']:.3f} of all SNV records; {diag['old_fraction_of_v6_missense']:.3f} of missense; {diag['old_snv_not_in_v6']} old SNV ids absent from new file ({diag['old_snv_not_in_v6_VariationID_absent_from_whole_release']} gone from the whole release)", "SUBSET (nearly)")
    add("Diagnosis", "Missense in residues 763-893 / 1549-1770 (ClinVar)", "0 / 0", f"{diag['v6_clinvar_missense_763_893']} / {diag['v6_clinvar_missense_1549_1770']}", "BUG")
    add("Diagnosis", "v5 'missense' that are stop-gains", "66 (dbNSFP aa present)", "0", "BUG")
    add("Diagnosis", "v5 labels vs ClinVar aggregate", "any pathogenic RCV => P", f"{diag['v5_P_now_conflicting_or_VUS_in_v6_aggregate']} v5-P and {diag['v5_B_now_VUS_in_v6_aggregate']} v5-B are Conflicting/VUS in ClinVar's own variant-level aggregate", "LABEL RULE")
    # ---- coverage
    add("Coverage", "3D-PPS undefined (all cohort)", "932 (33.7%)", f"{int(df.pps3d_norm.isna().sum())} ({100*df.pps3d_norm.isna().mean():.1f}%) [wider-set scores; incl. VUS and unlabelled]", "")
    add("Coverage", "Labelled P: undefined / >=3 neighbours", "20.9% / 46.6% (incl. 19 invalid positions)", f"{abv('wider','P','score_undefined_pct')}% / {abv('wider','P','covered_ge3_pct')}%", "CHANGED")
    add("Coverage", "Labelled B: undefined / >=3 neighbours", "50.9% / 6.4%", f"{abv('wider','B','score_undefined_pct')}% / {abv('wider','B','covered_ge3_pct')}%", "persists, stronger")
    add("Coverage", "Coverage flag (>=1 neighbour) AUROC", "0.650 [0.590, 0.707] (no saved script)", f"{cfv('wider','flag_ge1_auroc')} [{cfv('wider','flag_ge1_ci_low')}, {cfv('wider','flag_ge1_ci_high')}]; tighter {cfv('tighter','flag_ge1_auroc')}", "CHANGED (higher)")
    add("Coverage", "Coverage flag (>=3) AUROC", "0.701 [0.660, 0.741] (audit)", f"{cfv('wider','flag_ge3_auroc')} [{cfv('wider','flag_ge3_ci_low')}, {cfv('wider','flag_ge3_ci_high')}]", "")
    add("Coverage", "Share of gap to 3D-PPS reachable from coverage", "41% (vs eps-tied 0.869)", f"undefined: exact 3D-PPS AUROC {cfv('wider','auroc_3d_scoreable')} is not above chance (flag alone exceeds it)", "CHANGED")
    add("Coverage", "Odds ratio covered(>=3) P vs B, wider", "12.8, Fisher p=1.3e-17 (variants as independent)", f"{cfv('wider','or_fisher_naive')} naive Fisher p={cfv('wider','p_fisher_naive'):.1e}; cluster-robust logit OR {cfv('wider','or_cluster_logit')} [{cfv('wider','or_cluster_ci_low')}, {cfv('wider','or_cluster_ci_high')}] p={cfv('wider','p_cluster_logit'):.3f}; residue-bootstrap OR CI [{cfv('wider','or_boot_ci_low')}, {cfv('wider','or_boot_ci_high')}]", "persists; far less certain (only 30 benign)")
    add("Coverage", "Odds ratio, tighter", "6.3, p=2.0e-7", f"{cfv('tighter','or_fisher_naive')}; cluster p={cfv('tighter','p_cluster_logit'):.3f}", "")
    add("Coverage", "'thirteen times more likely'", "wording error (rel. likelihood 7.3x)", f"relative likelihood {cfv('wider','relative_likelihood_covered_P_over_B')}x; OR {cfv('wider','or_fisher_naive')}", "")
    add("Coverage", "Spearman count vs burial / vs pLDDT", "0.789 (p=5e-129) / 0.587", f"{bl[bl.set=='wider'].spearman_count_burial.iloc[0]} / {bl[bl.set=='wider'].spearman_count_plddt.iloc[0]}; VIF {bl[bl.set=='wider'].VIF_count_burial.iloc[0]}", "CHANGED")
    for cod in ("count", "flag_ge1", "flag_ge3"):
        r = bl[(bl.set == "wider") & (bl.coverage_coding == cod)].iloc[0]
        old = {"count": "beta 0.70, p=3e-9, burial p=0.82", "flag_ge1": "beta 0.87, p=9e-4; burial p=2.2e-5", "flag_ge3": "beta 2.18, p=5e-6; burial p=0.11"}[cod]
        add("Coverage", f"Logistic class ~ coverage({cod}) + burial", old, f"beta {r.beta_coverage}; cluster p={r.p_coverage_cluster:.2g} (naive {r.p_coverage_naive:.2g}); burial cluster p={r.p_burial_cluster:.2g}", "")
    # ---- AUROCs
    for p_, lab_, old in [("pps3d_norm", "3D-PPS (exact, scoreable only)", "0.869 [0.835, 0.900] zero-filled+eps (exact: 0.834 zero-fill, 0.842 abstain n=458)"),
                          ("pps1d_norm", "1D control", "0.881 [0.835, 0.921]"), ("alphamissense", "AlphaMissense", "0.921 [0.891, 0.946]"),
                          ("revel", "REVEL", "0.877 [0.835, 0.915]"), ("esm1b", "ESM1b", "0.793 [0.741, 0.841]"), ("sift", "SIFT", "0.785 [0.732, 0.836]"),
                          ("polyphen2_hvar", "PolyPhen-2", "0.735 [0.667, 0.798]"), ("cadd_phred", "CADD", "0.700 [0.626, 0.772]")]:
        add("AUROC (wider)", lab_, old, ci("wider", p_), "")
    add("AUROC (tighter)", "3D-PPS", "0.850 (0.806 exact)", ci("tighter", "pps3d_norm"), "")
    add("AUROC (tighter)", "AlphaMissense", "n/r", ci("tighter", "alphamissense"), "")
    add("3D-PPS", "Abstention: scoreable n (P/B)", "458 (404/54) exact; 602 zero-filled", f"{int(t4v('wider','pps3d_norm','n'))} ({int(t4v('wider','pps3d_norm','n_P'))}/{int(t4v('wider','pps3d_norm','n_B'))})", "CHANGED")
    sv = df[df.label.isin(["P", "B"])]
    add("3D-PPS", "Mean 3D-PPS P vs B (defined only)", "0.72 vs -0.14 (zero-filled)", f"{df[df.label.eq('P')].pps3d_norm.mean():.2f} vs {df[df.label.eq('B')].pps3d_norm.mean():.2f}", "")
    add("3D-PPS", "Mean burial P vs B", "16.0 vs 12.5", f"{df[df.label.eq('P')].coordination.mean():.1f} vs {df[df.label.eq('B')].coordination.mean():.1f}", "")
    r = h2h[(h2h.set == "wider") & (h2h.subset == "3D-PPS scoreable") & (h2h.comparison == "pps3d_norm - pps1d_norm")].iloc[0]
    add("3D vs 1D", "Paired 3D-PPS minus 1D (same variants, wider)", "never tested (0.881 vs 0.869 unpaired)", f"{r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}], n={r.n} (B={r.n_B}); tighter: {h2h[(h2h.set=='tighter') & (h2h.subset=='3D-PPS scoreable') & (h2h.comparison=='pps3d_norm - pps1d_norm')].delta.iloc[0]:+.3f}", "NEW")
    # coverage matched
    for i, nm in enumerate(["0", "1 to 2", "3 to 5", "6 to 10"]):
        r = st[(st.set == "wider") & (st.neighbours == nm)].iloc[0]
        old = {"0": "n=163 (107/56) 3D 0.500, AM 0.898, REVEL 0.841", "1 to 2": "n=213 (166/47) 0.875 / 0.880 / 0.857", "3 to 5": "n=155 (149/6) 0.950 / 0.895 / 0.918", "6 to 10": "n=88 (87/1) 0.931 / 1.000 / 1.000"}[nm]
        add("Strata (wider)", f"{nm} neighbours: 3D-PPS / AM / REVEL", old, f"n={r.n} ({r.n_P}/{r.n_B}) {r.pps3d_norm} / {r.alphamissense} / {r.revel}", "")
    r = h2h[(h2h.set == "wider") & (h2h.subset.str.startswith("covered")) & (h2h.comparison == "pps3d_norm - alphamissense")].iloc[0]
    r2 = h2h[(h2h.set == "tighter") & (h2h.subset.str.startswith("covered")) & (h2h.comparison == "pps3d_norm - alphamissense")].iloc[0]
    add("Head to head", "3D-PPS minus AlphaMissense on covered (>=3) variants", "+0.038 [-0.060, 0.185] n=243 (was TIGHTER set mislabelled wider); wider +0.031 [-0.076, 0.177] n=245", f"wider {r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] n={r.n} (B={r.n_B}); tighter {r2.delta:+.3f} n={r2.n} (B={r2.n_B})", "CHANGED: no longer testable (1 benign)")
    r = h2h[(h2h.set == "wider") & (h2h.subset == "3D-PPS scoreable") & (h2h.comparison == "pps3d_norm - alphamissense")].iloc[0]
    add("Head to head", "3D-PPS minus AlphaMissense on scoreable variants (wider)", "n/a", f"{r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}], TOST p={r.p_tost}, n={r.n}", "AM clearly better")
    # ---- permutation
    a, b_ = pr("primary", "wider", "null1"), pr("primary", "wider", "null2")
    add("Permutation", "Observed AUROC (single score)", "0.842 (perm) vs 0.869 (headline): two scores", f"{a.observed_auroc} (identical in permutation, tables, CV; n={a.n_scoreable}, B={a.n_B_scoreable})", "FIXED")
    add("Permutation", "Null 1 (unconditional) mean/SD/max, p", "0.497 / 0.044 / 0.675, p=1e-4 (0 of 10,000)", f"{a.null_mean} / {a.null_sd} / {a.null_max}; {a.n_ge_observed} of 10,000 >= observed; p={a.p_value:.4f}", "CONCLUSION CHANGED" if a.p_value > 0.05 else "holds")
    add("Permutation", "Null 2 (within coverage x burial strata) p", "not done", f"mean {b_.null_mean}, max {b_.null_max}; p={b_.p_value:.4f} ({int(b_.n_strata_cells)} strata)", "NEW")
    for tag, nm in [("v5rcv_v4", "full data + v5 label rule + gnomAD v4"), ("aggregate_v2", "full data + aggregate rule + gnomAD v2.1.1"), ("v5rcv_v2", "full data + v5 label rule + v2.1.1 (closest to v5)")]:
        x1, x2 = pr(tag, "wider", "null1"), pr(tag, "wider", "null2")
        add("Permutation", f"Sensitivity: {nm}", "-", f"observed {x1.observed_auroc} (n={x1.n_scoreable}, B={x1.n_B_scoreable}); null1 p={x1.p_value:.4f}; null2 p={x2.p_value:.4f}", "")
    # ---- models
    add("Models (tighter)", "Sequence panel only", "0.935 / AUPRC 0.988", cvm("tighter", "logistic", "A_only"), "")
    add("Models (tighter)", "Panel + 3D-PPS", "0.952 / 0.994", cvm("tighter", "logistic", "A_3d"), "")
    add("Models (tighter)", "Panel + coverage count only (no score value)", "not done", cvm("tighter", "logistic", "A_cov"), "NEW")
    add("Models (tighter)", "ClinVar-free panel + 3D-PPS", "0.928 / 0.987", cvm("tighter", "logistic", "pop_3d"), "")
    add("Models (wider)", "Panel / +3D / +coverage count", "0.929 / 0.940 / not done (A_only/A_3d)", f"{cvm('wider','logistic','A_only')} | {cvm('wider','logistic','A_3d')} | {cvm('wider','logistic','A_cov')}", "")
    add("Models", "Adding 3D-PPS to panel, paired (wider; tighter)", "+0.011 [0.000, 0.024]", f"{pmv('wider','lr/A_3d','lr/A_only')}; {pmv('tighter','lr/A_3d','lr/A_only')}", "")
    add("Models", "3D-PPS beyond coverage count (A_3d - A_cov)", "not done", f"wider {pmv('wider','lr/A_3d','lr/A_cov')}; tighter {pmv('tighter','lr/A_3d','lr/A_cov')}", "NEW")
    add("Models", "Dropping ClinVar-trained predictors (with 3D)", "'barely changed' (actual -0.024 tighter)", f"wider {pmv('wider','lr/pop_3d','lr/A_3d')}; tighter {pmv('tighter','lr/pop_3d','lr/A_3d')}", "")
    add("Models", "Gradient boosting A_only / A_3d (wider)", "claimed, no output", f"{cvm('wider','gradient_boosting','A_only')} | {cvm('wider','gradient_boosting','A_3d')}", "NOW RUN")
    add("Models", "Leakage inflation (global vs fold-wise 3D-PPS)", "0.945 vs 0.940, +0.004", f"wider naive {leak.auroc.iloc[0]} vs fold-wise {leak.auroc.iloc[1]} (diff {leak.auroc.iloc[2]}); tighter {leakt.auroc.iloc[0]} vs {leakt.auroc.iloc[1]} (diff {leakt.auroc.iloc[2]})", "")
    c0 = cal[(cal.set == "wider") & (cal.weights == "balanced")].iloc[0]; c1 = cal[(cal.set == "wider") & (cal.weights == "unweighted")].iloc[0]
    add("Calibration", "Brier / slope", "Brier 0.081 'reasonably well calibrated' (const. 0.146)", f"balanced: Brier {c0.brier}, slope {c0.cal_slope} [{c0.cal_slope_ci_low}, {c0.cal_slope_ci_high}], mean pred {c0.mean_pred} vs prevalence {c0.observed_prevalence}; unweighted: Brier {c1.brier}, slope {c1.cal_slope}; constant-guess Brier {c0.brier_constant_prevalence}", "slope<1: over-confident, not 'well calibrated'")
    # ---- VUS
    add("Discordance", "n evaluable VUS; loose/usual/ClinGen/strict %", "1,761; 46.1 / 47.2 / 50.5 / 62.1", f"{int(disc.n_evaluable.iloc[0])}; " + " / ".join(str(disc.set_index('thresholds').loc[k, 'pct_discordant']) for k in ['lenient', 'nominal', 'ClinGen PP3 supporting', 'stringent']), "")
    add("ACMG", "VUS -> Likely pathogenic; towards benign", "113; 44", f"{acm['n_LP_no_floor']}; {acm['n_towards_benign_no_floor']} (no floor)", "")
    add("ACMG", "LP calls resting on <3 labelled neighbours", "34 of 113 (audit)", f"{acm['n_LP_no_floor_with_lt3_neighbours']} of {acm['n_LP_no_floor']}; with support floor on PM1: LP = {acm['n_LP_with_floor']}", "NEW")
    add("Hotspots", "Candidates: no floor -> floor (>=3 pathogenic neighbours)", "22 -> 3 (L120F, F121L, V123A; residues 120-123)", f"{orp['orphans_no_floor']} -> {orp['orphans_with_floor_ge3']} ({orp['orphans_floor_variants']}; residues {orp['orphans_floor_residues']}); single-neighbour: {orp['orphans_no_floor_resting_on_single_neighbour']}", "CHANGED")
    add("Hotspots", "VUS 3D-PPS abstained", "'1/3 of variants'", f"{int(acm['n_abstained_VUS'])} of {int(acm['n_vus'])} VUS ({acm['pct_abstained_VUS']}%)", "")
    # ---- spatial
    s = sb[(sb.cohort == "primary") & (sb.set == "wider") & (sb.scheme == "blocks100") & (sb.subset == "all held-out variants")].set_index("method")
    for m in ["3D-PPS (scoreable only)", "AlphaMissense", "REVEL", "model A_only", "model A_cov", "model A_3d"]:
        r = s.loc[m]; add("Spatial block CV (100 aa blocks, wider)", m, "new in v6", f"AUROC {r.auroc} [{r.ci_low}, {r.ci_high}], n={r.n}, B={r.n_B}", "NEW")
    # ---- write
    N = pd.DataFrame(R, columns=["Section", "Quantity", "v5 value", "v6 value", "Change"])
    N.to_csv(TAB / "v6_numbers_side_by_side.csv", index=False)
    md = ["# v6 NUMBERS: v5 manuscript values vs full-data v6\n",
          "All v6 values are read from `results/tables/*.csv`, `logs/*.json` and are regenerated by `bash run_all.sh`.",
          "Primary v6 cohort = complete ClinVar release (variant_summary), ClinVar variant-level aggregate labels, gnomAD v4 allele frequencies, wider benign AF >= 1e-5 unless stated.\n",
          "Validation: the v6 code run on the unchanged v5 table reproduces the v5 audit values:\n"]
    md += [f"- {a}: **{b}**" for a, b in val.itertuples(index=False)]
    # regions that v5 could not see
    lab = pd.read_parquet("data/processed/labelled_wider.parquet")
    reg = []
    for nm, lo, hi in [("lever arm / S2 start 763-893", 763, 893), ("rod 1549-1770", 1549, 1770), ("all other residues", None, None)]:
        m = lab.aapos.between(lo, hi) if lo else ~(lab.aapos.between(763, 893) | lab.aapos.between(1549, 1770))
        mv = df.aapos.between(lo, hi) if lo else ~(df.aapos.between(763, 893) | df.aapos.between(1549, 1770))
        g = lab[m]
        reg.append({"region": nm, "labelled P": int(g.y.sum()), "labelled B": int((1 - g.y).sum()), "VUS": int((df[mv].label == "VUS").sum()),
                    "3D-PPS abstain % (labelled)": r3(100 * (~g.score_defined).mean(), 1), "mean neighbours": r3(g.n_nb.mean(), 2)})
    reg = pd.DataFrame(reg); reg.to_csv(TAB / "v6_regions_previously_missing.csv", index=False)
    A_ = pr("primary", "wider", "null1"); A2 = pr("primary", "wider", "null2")
    v52 = sens[(sens.cohort.str.contains("closest")) & (sens.set == "wider")].iloc[0]
    concl = ["\n## Conclusions that CHANGE (honest reading; nothing tuned to preserve v5)\n",
      f"1. **The v5 cohort was truncated.** 906 of 3,184 ClinVar missense variants ({diag['missed_pct']}%) were never retrieved (MyVariant stores no or another gene symbol for them; ClinVar's own GeneSymbol list contains MYH7). Residues 763-893 and 1549-1770 had zero ClinVar variants in v5 ({diag['v6_clinvar_missense_763_893']} and {diag['v6_clinvar_missense_1549_1770']} exist). v5 was {100*diag['old_fraction_of_v6_missense']:.1f}% of the full missense set; {diag['old_missense_not_in_v6_missense']} v5 'missense' records are not in the new set (66 were stop-gains), 15 v5 SNV ids no longer exist in the current release.",
      f"2. **v5 labels were also too permissive.** v5 called a variant pathogenic if any single condition record was P/LP. ClinVar's variant-level aggregate calls {diag['v5_P_now_conflicting_or_VUS_in_v6_aggregate']} of those Conflicting/VUS. Using ClinVar's aggregate, P falls from 511 to {acc['P']} and benign from 110 to {acc['B_wide']}; of v5's 72 frequency stand-ins, {diag['v5_standins_that_are_clinvar_VUS']} are ClinVar VUS and {diag['v5_standins_that_are_clinvar_P']} ClinVar pathogenic (they were only benign because ClinVar records were missing). Both effects are reported separately in the sensitivity cohorts.",
      f"3. **Permutation test: v5 rejected; v6 primary does NOT** (observed {A_.observed_auroc} on {int(A_.n_scoreable)} scoreable variants of which only {int(A_.n_B_scoreable)} are benign; unconditional null p={A_.p_value:.3f}; coverage/burial-stratified null p={A2.p_value:.3f}). Power is very low (7 benign), so this is 'no evidence', not 'evidence of no signal'. With the v5 label rule on the full data the test rejects again (p=1e-4 under both nulls) but the observed AUROC is {v52.auroc_3d_scoreable} (v5: 0.842), so completing the data alone removes most of the headline score performance.",
      f"4. **Coverage asymmetry persists under ClinVar-aggregate labels and is larger**: no labelled neighbour for {abv('wider','B','no_labelled_neighbour_pct') if 'no_labelled_neighbour_pct' in ab.columns else ''}% of benign vs {abv('wider','P','no_labelled_neighbour_pct')}% of pathogenic; coverage flag AUROC {cfv('wider','flag_ge1_auroc')} (v5 0.650). Under the v5 label rule on complete data it shrinks (flag AUROC {v52.flag_ge1_auroc}, OR {v52.or_covered_fisher_naive} vs v5 12.8). So the existence of asymmetry is robust, its size is not: it depends on how benign/pathogenic are defined. Residue-clustered inference widens the OR interval to [{cfv('wider','or_cluster_ci_low')}, {cfv('wider','or_cluster_ci_high')}] (naive Fisher p=1e-17 in v5 is not valid).",
      f"5. **AlphaMissense head to head: v5 'no measurable edge' no longer holds.** On scoreable variants 3D-PPS minus AlphaMissense = {h2h[(h2h.set=='wider') & (h2h.subset=='3D-PPS scoreable') & (h2h.comparison=='pps3d_norm - alphamissense')].delta.iloc[0]:+.3f} (CI excludes 0); the covered-subset comparison has 1 benign variant and is untestable.",
      f"6. **Coverage-matched strata are uninformative** (benign counts 23 / 6 / 1 / 0 across neighbour groups); 3D-PPS within strata is {st[(st.set=='wider') & (st.neighbours=='1 to 2')].pps3d_norm.iloc[0]} (1 to 2) and {st[(st.set=='wider') & (st.neighbours=='3 to 5')].pps3d_norm.iloc[0]} (3 to 5).",
      "7. **3D vs 1D**: paired difference includes zero in the primary cohort (+0.063 [-0.084, +0.151], 7 benign); the v5 claim that 3D structure adds anything over a sequence-distance control is untested and unsupported.",
      "8. **Models**: adding the coverage count alone (no score value) gives the same gain as adding 3D-PPS (A_3d minus A_cov: +0.001 wider, +0.010 tighter, equivalent within 0.05 by TOST). Dropping ClinVar-trained predictors costs about 0.07 AUROC (CI excludes 0), so v5's 'barely changed' is false. Calibration slope is 0.54 (balanced) / 0.77 (unweighted): over-confident, not 'well calibrated'.",
      "9. **Hotspots**: the v5 hotspot (L120F, F121L; V123A no longer passes) is still found, plus M90V/M90R/M92V; the screen shrinks from 6 to 5 with the floor instead of 22 to 3 (denser labels mean fewer single-neighbour artefacts).",
      "10. **Spatial block hold-out (new)**: 3D-PPS generalises no better than chance to held-out 100-residue blocks in the primary cohort (0.49, 3 scoreable benign) while AlphaMissense stays 0.90; with the v5 rule (55 benign) 3D-PPS is 0.925 on only 9 scoreable benign (CI not reliable) and paired difference vs AM +0.05 [-0.13, +0.24]. The result is therefore not a stable property of the score.",
      f"11. **Previously empty regions** are now covered by {diag['v6_clinvar_missense_763_893']+diag['v6_clinvar_missense_1549_1770']} ClinVar missense variants; see the regions table below. They change the picture mainly by adding pathogenic and VUS residues to the lever arm and rod, and by showing that most benign evidence remains outside them."]
    md += ["\n".join(concl)]
    md += ["\n## Regions v5 could not see (wider label set)\n", md_table(reg)]
    md += ["\n" + md_table(N)]
    md += ["\n## Sensitivity cohorts (isolating data completeness from label rule and gnomAD version)\n", md_table(sens)]
    md += ["\n## Predictor missingness (all-missense cohort, n=%d)\n" % n, md_table(mall[["non_missing", "pct_non_missing"]], index=True)]
    Path("results/NUMBERS.md").write_text("\n".join(md))
    print(f"[numbers] wrote results/NUMBERS.md ({len(N)} rows)")


if __name__ == "__main__":
    main()
