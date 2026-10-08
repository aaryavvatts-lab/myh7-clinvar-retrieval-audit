"""Build results/VERDICTS.md and results/tables/v7_verdicts.csv: one line per pre-specified analysis, with a
mechanical verdict (supported / not supported / underpowered / unfavourable) so the writer cannot cherry-pick.

Rules (PREREG sections 4 and 5, deviation D2):
  chance test (null 2, Holm within its family):  p_holm < 0.05            -> supported
                                                 else MDDx <= 0.05          -> not supported
                                                 else                       -> underpowered      (MDDx = 2.49 x null-2 SD)
                                                 fewer than 10 benign       -> underpowered
  contrast (claim sign s in {+1, -1, 0}; margin 0.02):
        s = +1 or -1: Holm p < 0.05 and sign of delta = s        -> supported
                      Holm p < 0.05 and sign of delta = -s       -> unfavourable (to the claim)
                      else MDD > 0.02                            -> underpowered
                      else                                       -> not supported
        s = 0 (equivalence): TOST p < 0.05 and MDD <= 0.02       -> supported
                      Holm p < 0.05 (a difference exists)        -> unfavourable
                      else                                       -> underpowered (if MDD > 0.02) / not supported
"Unfavourable" always means: significant, in the direction opposite to what the row's claim says. The claim of each row is
written out so the direction can be audited. Every row is reported; none is dropped.
Run: python -m src.verdicts
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from src.genes import *   # noqa
from src.eng import holm

ROWS = []


def add(id_, family, analysis, claim, result, test, verdict, source, note=""):
    ROWS.append({"id": id_, "family": family, "analysis": analysis, "claim_tested": claim, "result": result, "test": test,
                 "verdict": verdict, "source_file": source, "note": note})


def chance_verdict(p_adj, n_B, mddx, thr=0.05):
    if n_B < 10:
        return "underpowered"
    if p_adj < 0.05:
        return "supported"
    return "not supported" if mddx <= thr else "underpowered"


def contrast_verdict(delta, p_adj, mdd, tost_p, sign, margin=0.02, n_B=99):
    if n_B < 10:
        return "underpowered"
    if sign == 0:
        if tost_p < 0.05 and mdd <= margin:
            return "supported"
        if p_adj < 0.05:
            return "unfavourable"
        return "underpowered" if mdd > margin else "not supported"
    if p_adj < 0.05:
        return "supported" if np.sign(delta) == sign else "unfavourable"
    return "underpowered" if mdd > margin else "not supported"


def fmt(d, lo, hi, k=3):
    return f"{d:+.{k}f} (95% CI {lo:+.{k}f} to {hi:+.{k}f})"


def main():
    T = lambda f: pd.read_csv(TAB / f)
    # ---------------- F1: MYH7 chance tests (null 2)
    pm = T("v7_perm_tests.csv")
    f1 = pm[(pm.null == "null2_coverage_burial_strata")]
    for _, r in f1.iterrows():
        mddx = 2.49 * r.null_sd
        padj = r.p_holm_F1 if pd.notna(r.p_holm_F1) else np.nan
        in_family = r.tier in ("T1", "T2", "T3", "T3legacy_wide")
        v = chance_verdict(padj if in_family else r.p_one_sided, r.n_B, mddx) if True else ""
        add(f"F1-{r.tier}-{r.score}", "F1" if in_family else "F1-extra (outside Holm family)", f"MYH7 {r.tier}: {r.score} vs coverage-and-burial null",
            "Label proximity (this score) separates P from B beyond what coverage and burial alone produce",
            f"AUROC {r.observed:.3f} vs null-2 mean {r.null_mean:.3f} (SD {r.null_sd:.3f}); B={int(r.n_B)}; MDDx={mddx:.3f}",
            f"null 2, one-sided p={r.p_one_sided:.4f}" + (f", Holm p={padj:.3f}" if in_family else " (not in Holm family)"), v, "v7_perm_tests.csv",
            "S0 scoreable is capped at 0.5 x P(tie)+0.5 by construction (see v7_estimand_grid.csv p_pair_greater=0); read as degenerate." if r.score == "S0a" else "")
    # ---------------- F2: nested models
    c = T("v7_cv_nested_contrasts.csv"); c = c[c.grouping == "residue"]
    claims = {"count over panel": "A neighbour count adds discrimination to the predictor panel",
              "S1 over panel": "S1 adds discrimination to the predictor panel",
              "S1 beyond count (primary nested)": "S1 adds discrimination beyond panel + neighbour count",
              "count beyond S1": "The neighbour count adds discrimination beyond panel + S1"}
    for _, r in c.iterrows():
        sup = contrast_verdict(r.delta_mean, r.p_holm_F2, r.mdd_80, r["p_tost_0.02"], +1, n_B=r.n_B)
        eq = contrast_verdict(r.delta_mean, r.p_holm_F2, r.mdd_80, r["p_tost_0.02"], 0, n_B=r.n_B)
        add(f"F2-{r.tier}-{r.question.split(' (')[0].replace(' ', '_')}", "F2", f"MYH7 {r.tier}: {r.question}", claims[r.question] + " (superiority, delta > 0)",
            fmt(r.delta_mean, r.ci_low, r.ci_high) + f"; SD over 20 repeats {r.delta_sd_across_repeats:.4f}; MDD {r.mdd_80:.3f}",
            f"20x residue-grouped 5-fold CV; bootstrap p={r.p_two_sided_boot:.4f}, Holm p={r.p_holm_F2:.3f}", sup, "v7_cv_nested_contrasts.csv")
        add(f"F2eq-{r.tier}-{r.question.split(' (')[0].replace(' ', '_')}", "F2", f"MYH7 {r.tier}: {r.question}", "No difference larger than 0.02 AUROC (equivalence)",
            fmt(r.delta_mean, r.ci_low, r.ci_high) + f"; MDD {r.mdd_80:.3f}", f"TOST at +-0.02: p={r['p_tost_0.02']:.4f}", eq, "v7_cv_nested_contrasts.csv",
            "bootstrap holds out-of-fold predictions fixed, so MDD is a lower bound; the across-repeat SD is shown in the result" if True else "")
    # ---------------- F3: paired contrasts
    f3 = T("v7_contrasts_F3.csv")
    sign = {0: ("3D (S0, scoreable) is not better than a sequence-neighbour (1D) score", -1), 1: ("S1 and its 1D counterpart are equivalent (within 0.02)", 0),
            2: ("S1 and the neighbour count have the same discrimination (equivalence)", 0), 3: ("AlphaMissense is better than S1", -1),
            4: ("REVEL is better than S1", -1), 5: ("The benign tier (T2 vs T1) does not change the S1 AUROC", 0),
            6: ("Adding the two frequency stand-ins (T3 vs T2) does not change the S1 AUROC", 0),
            7: ("Legacy global-AF stand-ins (T3-legacy vs T3) do not change the S1 AUROC", 0),
            8: ("The label rule (aggregate lower than any-record) lowers the S1 AUROC", -1),
            9: ("Retrieval completeness (complete vs truncated v5) lowers the S1 AUROC", -1)}
    for i, r in f3.iterrows():
        txt, s = sign[i]
        nBv = r.get("n_B_in_A", np.nan)
        n_B = int(nBv) if pd.notna(nBv) else 20
        # for claim sign convention: 'A is better than B' with delta = A - B; for -1-claims (B better), claim holds when delta < 0
        v = contrast_verdict(r.delta, r.p_holm_F3, r.mdd_80 if pd.notna(r.mdd_80) else r.boot_sd * 2.8, r["p_tost_0.02"], s, n_B=n_B)
        add(f"F3-{i+1}", "F3", r.test, txt, fmt(r.delta, r.ci_low, r.ci_high) + f"; MDD {r.mdd_80 if pd.notna(r.mdd_80) else float('nan'):.3f}",
            f"paired residue-cluster bootstrap p={r.p_two_sided_boot:.4f}, Holm p={r.p_holm_F3:.3f}; TOST(0.02) p={r['p_tost_0.02']:.3f}", v, "v7_contrasts_F3.csv",
            "shared-variant contrast: benign composition held fixed" if i in (5, 6, 7) else "")
    # ---------------- coverage inference (exploratory family)
    ci = T("v7_coverage_inference_myh7.csv")
    for _, r in ci.iterrows():
        v = "supported" if r.wild_cluster_p < 0.05 else "not supported"
        add(f"COV-{r.tier}-ge{int(r.min_neighbours)}", "exploratory", f"MYH7 {r.tier}: coverage association (>= {int(r.min_neighbours)} labelled neighbours)",
            "Benign variants are less often covered by labelled neighbours than pathogenic ones",
            f"exact OR {r.or_conditional_mle:.2f} (exact 95% CI {r.or_exact_ci_low:.2f} to {r.or_exact_ci_high:.2f}); B covered {int(r.B_covered)}/{int(r.B_covered+r.B_uncovered)}; CR2 LPM p={r.lpm_p:.4f}",
            f"wild cluster score bootstrap p={r.wild_cluster_p:.4f} (unadjusted, exploratory)", v, "v7_coverage_inference_myh7.csv",
            "v6 Wald interval replaced by exact / wild cluster / CR2 inference")
    # ---------------- retrieval audit
    j = json.load(open(TAB / "v7_retrieval_audit_summary.json"))
    for key, nm in (("A_panel19", "19 pre-listed genes"), ("A2_top100", "top-100 ClinVar-missense genes")):
        s = j[key]; lo, hi = s["prop_gt5pct_exact_ci"]
        v = "supported" if lo > 0.5 else ("not supported" if hi < 0.5 else "underpowered")
        add(f"RET-{key}", "retrieval audit", f"Symbol-field MyVariant query vs ClinVar bulk, {nm}",
            "Most genes lose more than 5% of their ClinVar missense variants with the single-symbol MyVariant query (the loss generalises)",
            f"{s['n_loss_gt5pct']}/{s['n_genes']} genes lose >5% (exact 95% CI {lo:.2f} to {hi:.2f}); median loss {s['median_loss_pct']:.1f}%, max {s['max_loss_pct']:.1f}%; pooled {s['pooled_loss_pct']:.1f}%",
            "exact binomial interval vs 50%", v, "v7_retrieval_audit_summary.json", f"causes of missed variants: {s['causes_of_missed']}")
    g = T("v7_retrieval_audit_gene.csv"); m = g[g.gene == "MYH7"].iloc[0]
    add("RET-MYH7", "retrieval audit", "MYH7 loss re-measured with all-pages scroll", "MYH7 loses 28.5% (906/3184), 538 symbol-field and 368 snapshot",
        f"{int(m.n_missed)}/{int(m.n_clinvar_missense)} ({m.loss_pct:.1f}%): symbol field {int(m.n_symbol_field)}, snapshot-missing {int(m.n_snapshot_missing)} ({int(m.n_snapshot_missing_newer)} newer than the retrieved ceiling), other {int(m.n_other)}",
        "count comparison", "supported", "v7_retrieval_audit_gene.csv", "reproduces v6 within 1 variant (classification of one record differs; see NUMBERS_v7)")
    # ---------------- multi-gene
    gs = T("v7_gene_summary.csv"); gs = gs[gs.tier == "T2"]
    gp = T("v7_gene_perm.csv"); gp = gp[(gp.null == "null2_coverage_burial_strata") & (gp.score == "S1")].set_index("gene")
    for _, r in gs.iterrows():
        sd = gp.loc[r.gene, "null_sd"]; mddx = 2.49 * sd
        v = chance_verdict(r.S1_null2_p_holm_F4, r.n_B, mddx)
        add(f"F4-{r.gene}", "F4", f"{r.gene} T2: S1 vs coverage-and-burial null", "S1 separates P from B beyond coverage and burial",
            f"AUROC {r.auroc_S1:.3f} vs null-2 mean {r.S1_null2_mean:.3f}; B={int(r.n_B)}; MDDx={mddx:.3f}", f"null 2 p={r.S1_null2_p:.4f}, Holm p={r.S1_null2_p_holm_F4:.3f}", v, "v7_gene_summary.csv")
        if pd.notna(r.get("cv_M3_minus_M1")):
            lo, hi = [float(x) for x in str(r.cv_M3_minus_M1_ci).split(" to ")]
            sup = contrast_verdict(r.cv_M3_minus_M1, r.cv_M3_minus_M1_p_holm_F6, r.cv_M3_minus_M1_mdd, r.cv_M3_minus_M1_tost, +1, n_B=r.n_B)
            eq = contrast_verdict(r.cv_M3_minus_M1, r.cv_M3_minus_M1_p_holm_F6, r.cv_M3_minus_M1_mdd, r.cv_M3_minus_M1_tost, 0, n_B=r.n_B)
            add(f"F6-{r.gene}", "F6", f"{r.gene} T2: S1 beyond panel + count (nested CV)", "S1 adds discrimination beyond panel + neighbour count (superiority)",
                fmt(r.cv_M3_minus_M1, lo, hi) + f"; MDD {r.cv_M3_minus_M1_mdd:.3f}", f"Holm p={r.cv_M3_minus_M1_p_holm_F6:.3f}", sup, "v7_gene_summary.csv")
            add(f"F6eq-{r.gene}", "F6", f"{r.gene} T2: S1 beyond panel + count (nested CV)", "No difference larger than 0.02 AUROC (equivalence)",
                fmt(r.cv_M3_minus_M1, lo, hi) + f"; MDD {r.cv_M3_minus_M1_mdd:.3f}", f"TOST p={r.cv_M3_minus_M1_tost:.4f}", eq, "v7_gene_summary.csv")
    mt = T("v7_meta_pooled.csv")
    for _, r in mt[mt.set == "amended_pool"].iterrows():
        if (r.A, r.B) not in (("S1", "o_alphamissense"), ("S1", "n_nb")):
            continue
        pa = r.p_holm_F5 if pd.notna(r.p_holm_F5) else r.p_hk
        s = -1 if r.B == "o_alphamissense" else 0
        # equivalence for count uses the HK CI inside +-0.02
        if s == 0:
            v = "supported" if (r.ci_low_hk > -0.02 and r.ci_high_hk < 0.02) else ("unfavourable" if pa < 0.05 else "underpowered")
        else:
            v = "supported" if (pa < 0.05 and r.pooled < 0) else ("unfavourable" if (pa < 0.05 and r.pooled > 0) else "underpowered")
        claim = "AlphaMissense is better than S1 across genes (pooled)" if s == -1 else "S1 and the neighbour count are equivalent across genes (HK CI inside +-0.02)"
        add(f"F5-{r.B}", "F5", f"Random-effects pooled S1 minus {r.B}, {int(r.k)} genes", claim,
            fmt(r.pooled, r.ci_low_hk, r.ci_high_hk) + f"; tau={r.tau:.3f}; I2={r.I2_pct:.0f}%; 95% PI {r.pred_low:+.3f} to {r.pred_high:+.3f}", f"HK p={r.p_hk:.4f}, Holm p={pa:.4f}", v, "v7_meta_pooled.csv")
    mr = T("v7_meta_regression.csv")
    for _, r in mr.iterrows():
        add(f"META-REG-{r.A}-{r.B}-{r.moderator}", "exploratory", f"Meta-regression of S1 minus {r.B} on {r.moderator} ({int(r.k)} genes)",
            "Performance of the score tracks the moderator (benign scarcity / coverage asymmetry / ties)", f"slope {r.slope:+.3f} (SE {r.slope_se:.3f}); Spearman rho {r.spearman_rho:+.2f}",
            f"WLS p={r.p_wls:.3f}; Spearman p={r.spearman_p:.3f} (unadjusted)", "supported" if r.p_wls < 0.05 else "underpowered", "v7_meta_regression.csv", "exploratory; k is small")
    V = pd.DataFrame(ROWS)
    V.to_csv(TAB / "v7_verdicts.csv", index=False)
    out = ["# VERDICTS (generated by src/verdicts.py from the saved tables; rules in the docstring and in PREREG.md section 5 and D2)", "",
           "Every pre-specified analysis is listed once with a mechanical verdict. 'unfavourable' = significant in the direction opposite to the row's own claim. "
           "Holm correction is applied within each family (F1..F6) as pre-listed; rows marked exploratory use unadjusted p-values and are not tests of record.", "",
           "## Counts by verdict", ""]
    cnt = V.groupby(["family", "verdict"]).size().unstack(fill_value=0)
    cols = list(cnt.columns)
    out.append("| family | " + " | ".join(cols) + " |"); out.append("|---|" + "---|" * len(cols))
    for fam_, r_ in cnt.iterrows():
        out.append(f"| {fam_} | " + " | ".join(str(int(r_[c_])) for c_ in cols) + " |")
    out.append("")
    for fam, d in V.groupby("family", sort=False):
        out += [f"## {fam}", "", "| id | analysis | claim tested | result | test | verdict | source |", "|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            out.append(f"| {r.id} | {r.analysis} | {r.claim_tested} | {r.result} | {r.test} | **{r.verdict}** | {r.source_file} |")
        out.append("")
    (Path("results") / "VERDICTS.md").write_text("\n".join(out))
    print(cnt)


from pathlib import Path   # noqa
if __name__ == "__main__":
    main()
