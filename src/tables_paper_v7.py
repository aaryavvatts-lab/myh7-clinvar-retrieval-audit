"""Pipe tables for the v7 manuscript and supplement, generated from results/tables/*.csv|json so that no number is typed by hand.
Used by src/build_paper_v7.py through the tokens {{TABLE:name}}.  A first line '%%W w1 w2 ...' sets relative column widths.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

TAB = Path("results/tables")
rd = lambda n: pd.read_csv(TAB / n)


def f(x, k=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{k}f}"


def sgn(x, k=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{k}f}"


def pf(x, k=4):
    """p-value; a value at the resampling floor 1/10,001 is written as such."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return "0.0001 (floor)" if x <= 1.0001 / 10001 else f"{x:.{k}f}"


def pct(x, k=1):
    return f"{100 * x:.{k}f}"


def md(head, rows, align=None, widths=None):
    align = align or ["---"] + ["---:"] * (len(head) - 1)
    out = ([f"%%W {' '.join(str(w) for w in widths)}"] if widths else [])
    out += ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ main text
def retrieval_summary():
    s = json.loads((TAB / "v7_retrieval_audit_summary.json").read_text())
    rows = []
    for key, nm in (("A_panel19", "19 candidate genes"), ("A2_top100", "100 genes with most ClinVar missense SNVs")):
        a = s[key]; c = a["causes_of_missed"]; lo, hi = a["prop_gt5pct_exact_ci"]
        rows.append([nm, f"{a['n_loss_gt5pct']} of {a['n_genes']} ({lo:.2f} to {hi:.2f})", f"{a['median_loss_pct']:.1f}",
                     f"{a['pooled_loss_pct']:.1f} ({a['total_missed']:,} of {a['total_missense']:,})",
                     f"{c['n_symbol_field']:,} ({100 * c['n_symbol_field'] / a['total_missed']:.1f}%)",
                     f"{c['n_snapshot_missing']:,} ({100 * c['n_snapshot_missing'] / a['total_missed']:.1f}%)"])
    head = ["Gene set", "Genes losing more than 5% (exact 95% CI of share)", "Median loss per gene (%)", "Pooled loss (%)",
            "Lost: other gene symbol stored", "Lost: absent from the snapshot"]
    return md(head, rows, ["---", "---:", "---:", "---:", "---:", "---:"], [1.5, 1.4, 1, 1.5, 1.2, 1.3])


def genes_main():
    g = rd("v7_gene_summary.csv"); g = g[g.tier == "T2"].copy()
    myh = g[g.gene == "MYH7"]; rest = g[g.gene != "MYH7"].sort_values("n_B")
    g = pd.concat([myh, rest])
    rows = []
    for _, r in g.iterrows():
        rows.append([r.gene, int(r.n_P), int(r.n_B), int(r.n_B_scoreable_S0), f(r.S0_tie_fraction, 2), pct(r.benign_no_neighbour, 0),
                     pct(r.pathogenic_no_neighbour, 0), f(r.auroc_S1, 2), f(r.auroc_n_nb, 2), f(r.auroc_AlphaMissense, 2)])
    head = ["Gene", "P", "B", "Scoreable B (S0)", "S0 tie fraction", "B with no neighbour (%)", "P with no neighbour (%)",
            "AUROC S1", "AUROC count", "AUROC AlphaMissense"]
    return md(head, rows, ["---"] + ["---:"] * 9, [1.1, 0.8, 0.8, 1.1, 1.0, 1.2, 1.2, 0.9, 1.0, 1.3])


def verdict_counts():
    v = rd("v7_verdicts.csv")
    desc = {"F1": "MYH7: score against the coverage-and-burial null (4 scores, 4 benign sets)",
            "F2": "MYH7: nested models (4 contrasts, 3 benign sets; superiority and equivalence)",
            "F3": "MYH7: paired contrasts between scores and between benign sets",
            "F4": "Ten genes: S1 against the coverage-and-burial null",
            "F5": "Ten genes: pooled S1 minus AlphaMissense and minus neighbour count",
            "F6": "Ten genes: S1 beyond panel plus count (nested; superiority and equivalence)"}
    rows = []
    for fam in ("F1", "F2", "F3", "F4", "F5", "F6"):
        c = v[v.family == fam].verdict.value_counts()
        rows.append([fam, desc[fam]] + [int(c.get(k, 0)) for k in ("supported", "not supported", "underpowered", "unfavourable")])
    return md(["Family", "Question", "Supported", "Not supported", "Underpowered", "Unfavourable"], rows,
              ["---", "---", "---:", "---:", "---:", "---:"], [0.6, 3.2, 0.9, 0.9, 1.0, 1.0])


# ------------------------------------------------------------------ supplement
def s_inclusion():
    d = rd("v7_inclusion.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append([r.gene, int(r.n_P_T2), int(r.n_B_T2), f(r.seq_match_frac, 2), "yes" if r.I1 else "no", "yes" if r.I2 else "no",
                     "yes" if r.I3 else "no", "yes" if r.I4 else "no", "yes" if r.included else "no",
                     "" if pd.isna(r.failed) else str(r.failed)])
    return md(["Gene", "P", "B", "Residue match", "I1", "I2", "I3", "I4", "Included", "First failed"], rows,
              ["---"] + ["---:"] * 4 + ["---"] * 5, [1, 0.7, 0.7, 1, 0.6, 0.6, 0.6, 0.6, 0.9, 2.6])


def s_retrieval_panel():
    d = rd("v7_retrieval_audit_gene.csv"); d = d[d.in_panel_A].sort_values("loss_pct", ascending=False)
    rows = [[r.gene, f"{int(r.n_clinvar_missense):,}", f"{int(r.n_missed):,}", f(r.loss_pct, 1), int(r.n_symbol_field), int(r.n_snapshot_missing),
             int(r.n_other), "none" if pd.isna(r.longest_run_residues) else str(r.longest_run_residues)] for _, r in d.iterrows()]
    return md(["Gene", "ClinVar missense SNVs", "Not returned", "Loss (%)", "Other symbol stored", "Absent from snapshot", "Other",
               "Longest run of 10%-bins with 90% loss or more (residues)"], rows, ["---"] + ["---:"] * 6 + ["---"], [1, 1.3, 1.1, 0.9, 1.1, 1.2, 0.7, 2])


def s_estimand():
    g = rd("v7_estimand_grid.csv")
    names = [("T2", "T2 any ClinVar benign (primary)"), ("T1", "T1 curated benign"), ("T3", "T3 with population stand-ins"),
             ("T3legacy_wide", "Global-frequency stand-ins, AF 1e-5 or more"), ("T3legacy_tight", "Global-frequency stand-ins, AF 5e-5 or more")]
    cols = [("S0_scoreable", "S0 scoreable"), ("S0_neutral", "S0 neutral"), ("S0_lowest", "S0 lowest-fill"), ("S1", "S1"), ("raw_diff", "Raw difference d_P - d_B"),
            ("n_nb", "Count"), ("AlphaMissense", "AlphaMissense")]
    rows = []
    for ds, nm in names:
        sub = g[g.dataset == ds]; r = [nm, int(sub[sub.score == "S1"].n_B.iloc[0])]
        for sc, _ in cols:
            x = sub[sub.score == sc].iloc[0]
            r.append(f"{x.auroc:.3f} ({x.ci_low:.3f} to {x.ci_high:.3f}); ties {x.tie_fraction:.2f}")
        rows.append(r)
    return md(["Benign set", "B"] + [c[1] for c in cols], rows, ["---", "---:"] + ["---:"] * 7, [1.9, 0.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])


def verdict_family(fam):
    def _t():
        v = rd("v7_verdicts.csv"); v = v[v.family == fam]
        import re
        fl = lambda t: re.sub(r"(?<!HK )p=0\.0003(?!\d)", "p=0.0003 (floor)", re.sub(r"(?<!HK )p=0\.0001(?!\d)", "p=0.0001 (floor)", str(t)))
        rows = [[r.analysis, fl(r.result), fl(r.test), r.verdict] for _, r in v.iterrows()]
        return md(["Analysis", "Result", "Test", "Verdict"], rows, ["---"] * 4, [2, 3, 2.2, 1])
    return _t


def s_tiers():
    t = rd("v7_tier_summary_myh7.csv")
    nm = {"T1": "T1 curated", "T2": "T2 any ClinVar benign", "T3": "T3 with population stand-ins", "T3legacy_wide": "Global-frequency stand-ins (AF 1e-5 or more)",
          "T3legacy_tight": "Global-frequency stand-ins (AF 5e-5 or more)", "T1both": "T1 with 2-star pathogenic only", "T3_BA1": "T3 at the BA1 frequency"}
    rows = [[nm.get(r.tier, r.tier), int(r.n_P), int(r.n_B), int(r.B_residues), int(r.n_B_scoreable_S0), f(r.B_no_labelled_neighbour_pct, 1),
             f(r.P_no_labelled_neighbour_pct, 1)] for _, r in t.iterrows()]
    return md(["Benign set", "P", "B", "B residues", "Scoreable B (S0)", "B with no neighbour (%)", "P with no neighbour (%)"], rows,
              ["---"] + ["---:"] * 6, [2.4, 0.7, 0.7, 1, 1.1, 1.3, 1.3])


def s_af():
    a = rd("v7_af_strata_myh7.csv")
    cls = list(dict.fromkeys(a["class"])); bins = list(dict.fromkeys(a.af_bin))
    rows = []
    for c in cls:
        sub = a[a["class"] == c]; rows.append([c] + [f"{int(sub[sub.af_bin == b].n.iloc[0])} ({100 * sub[sub.af_bin == b].share_of_class.iloc[0]:.1f}%)" for b in bins])
    return md(["MYH7 protein-change class"] + [f"gnomAD v4 AF {b}" for b in bins], rows, ["---"] + ["---:"] * len(bins), [2.2, 1, 1, 1.1, 1])


def s_nested():
    n = rd("v7_cv_nested_contrasts.csv"); n = n[n.grouping == "residue"]
    nm = {"T1": "T1", "T2": "T2", "T3": "T3"}
    rows = [[nm[r.tier], r.question, f"{sgn(r.delta_mean)} ({sgn(r.ci_low)} to {sgn(r.ci_high)})", f(r.delta_sd_across_repeats, 4), f(r.mdd_80, 3),
             ("0.0003 (floor)" if r["p_tost_0.02"] <= 0.00034 else f(r["p_tost_0.02"], 4)), f"{int(r.n_repeats_positive)}/20"] for _, r in n.iterrows()]
    return md(["Benign set", "Contrast", "Mean difference (95% CI)", "SD over 20 repeats", "MDD", "TOST p at 0.02", "Repeats positive"], rows,
              ["---", "---"] + ["---:"] * 5, [0.9, 2.4, 2.2, 1.1, 0.7, 1, 1])


def s_perm():
    p = rd("v7_perm_tests.csv"); p = p[p.tier == "T2"]
    rows = [[r["null"].replace("null2_coverage_burial_strata", "null 2 (strata)").replace("null1_unconditional", "null 1 (all labels)").replace("null2_residue_level", "null 2 (residue level)"),
             r.score, f(r.observed), f"{f(r.null_mean)} ({f(r.null_sd)})", pf(r.p_one_sided), "n/a" if pd.isna(r.p_holm_F1) else f(r.p_holm_F1, 3)] for _, r in p.iterrows()]
    return md(["Null", "Score", "Observed AUROC", "Null mean (SD)", "p (one-sided)", "Holm p (F1)"], rows, ["---", "---"] + ["---:"] * 4, [2, 0.8, 1.1, 1.4, 1, 1])


def s_gene_extra():
    g = rd("v7_gene_summary.csv"); g = g[g.tier == "T2"]
    pr = rd("v7_gene_pairs.csv"); pr = pr[(pr.A == "S1") & (pr.B == "S1_1d") & (pr.tier == "T2")].set_index("gene")
    rows = [[r.gene, f(r.S1_tie_fraction, 2), f"{sgn(pr.loc[r.gene].delta)} ({sgn(pr.loc[r.gene].ci_low)} to {sgn(pr.loc[r.gene].ci_high)})", pf(r.S1_null2_p), f(r.S1_null2_p_holm_F4, 3), f"{sgn(r.delta_S1_minus_o_alphamissense)}", f"{sgn(r.delta_S1_minus_n_nb)}",
             f(r.cv_auroc_M0, 3), f(r.cv_auroc_M1, 3), f(r.cv_auroc_M3, 3), f"{sgn(r.cv_M3_minus_M1)} ({r.cv_M3_minus_M1_ci})", f(r.cv_M3_minus_M1_mdd, 3),
             f(r.cv_M3_minus_M1_p_holm_F6, 3)] for _, r in g.iterrows()]
    return md(["Gene", "S1 tie fraction", "S1 minus 1D S1 (95% CI)", "Null 2 p (S1)", "Holm p (F4)", "S1 minus AM", "S1 minus count", "CV AUROC panel", "Panel + count", "Panel + count + S1",
               "S1 beyond count", "MDD", "Holm p (F6)"], rows, ["---"] + ["---:"] * 12, [1.3, 0.7, 1.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.5, 0.6, 0.8])


def s_meta():
    m = rd("v7_meta_pooled.csv")
    lab = {"amended_pool": "Ten genes", "prelisted_panel_passing": "Six genes of the original list", "amended_pool_excluding_MYH7": "Ten genes without MYH7"}
    nm = {"o_alphamissense": "AlphaMissense", "n_nb": "neighbour count", "S1_1d": "1D S1", "S1": "S1"}
    rows = [[lab[r.set], f"{r.A} minus {nm.get(r.B, r.B)}", int(r.k), f"{sgn(r.pooled)} ({sgn(r.ci_low_hk)} to {sgn(r.ci_high_hk)})", f(r.tau, 3), f(r.I2_pct, 0),
             f"{sgn(r.pred_low)} to {sgn(r.pred_high)}"] for _, r in m.iterrows()]
    return md(["Gene set", "Contrast", "k", "Pooled difference (Hartung-Knapp 95% CI)", "tau", "I2 (%)", "95% prediction interval"], rows,
              ["---", "---"] + ["---:"] * 5, [1.6, 1.8, 0.4, 2.2, 0.6, 0.6, 1.8])


def s_pm1():
    p = rd("v7_pm1_call_rates.csv"); p = p[p.tier == "T2"]
    rows = []
    for (th, fl), sub in p.groupby(["threshold", "support_floor"]):
        pp = sub[sub["class"] == "P"].iloc[0]; bb = sub[sub["class"] == "B"].iloc[0]
        rows.append([f(th, 1), int(fl), f"{int(pp.n_called)}/{int(pp.n)} = {pp.rate:.3f} ({pp.ci_low:.3f} to {pp.ci_high:.3f})",
                     f"{int(bb.n_called)}/{int(bb.n)} = {bb.rate:.3f} ({bb.ci_low:.3f} to {bb.ci_high:.3f})", f(pp.LR_plus, 2)])
    return md(["S0 threshold", "Support floor", "Pathogenic called (exact 95% CI)", "Benign called (exact 95% CI)", "LR+"], rows,
              ["---:", "---:", "---:", "---:", "---:"], [0.9, 0.9, 2.6, 2.6, 0.7])


def s_sweep():
    s = rd("v7_param_sweep_myh7.csv")
    rows = []
    for (p_, v_), sub in s.groupby(["param", "value"], sort=False):
        r = [p_, f"{v_:g}" + (" (default)" if sub.is_default.iloc[0] else "")]
        for sc in ("S0_neutral", "S1", "n_nb"):
            x = sub[sub.score == sc].iloc[0]; r.append(f"{x.auroc:.3f} ({x.ci_low:.3f} to {x.ci_high:.3f})")
        rows.append(r)
    return md(["Parameter", "Value", "S0 neutral", "S1", "Neighbour count"], rows, ["---", "---:", "---:", "---:", "---:"], [1, 1.3, 1.8, 1.8, 1.8])


def s_k():
    k = rd("v7_estimand_k_sensitivity.csv")
    rows = []
    for kk, sub in k.groupby("k", sort=False):
        s1 = sub[sub.score == "S1"].iloc[0]; s2 = sub[sub.score == "S2"].iloc[0]
        rows.append([f"{kk:g}", f"{s1.auroc:.3f} ({s1.ci_low:.3f} to {s1.ci_high:.3f})", f"{s2.auroc:.3f} ({s2.ci_low:.3f} to {s2.ci_high:.3f})", f(s1.tie_fraction, 3)])
    return md(["Pseudo-count k", "S1", "S2", "Tie fraction"], rows, ["---:", "---:", "---:", "---:"], [1, 2, 2, 1])


def s_earlier_cohort_blocks():
    s = rd("spatial_block_cv.csv")
    s = s[(s.cohort == "primary") & (s.scheme == "blocks100") & (s.set == "wider") & (s.subset == "all held-out variants")]
    rows = [[r.method.replace("model A_only", "Panel only").replace("model A_cov", "Panel + coverage count").replace("model A_3d", "Panel + 3D-PPS (S0)"),
             int(r.n), int(r.n_B), f"{r.auroc:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f})"] for _, r in s.iterrows()]
    return md(["Method", "Variants", "Benign", "AUROC (95% CI)"], rows, ["---", "---:", "---:", "---:"], [3, 1, 1, 2])


def s_calib():
    c = rd("v6_calibration_summary.csv"); c = c[c.set == "wider"]
    rows = [[r.weights, f"{r.cal_slope:.3f} ({r.cal_slope_ci_low:.3f} to {r.cal_slope_ci_high:.3f})", f"{r.brier:.3f}", f"{r.brier_constant_prevalence:.3f}",
             f"{r.mean_pred:.3f}", f"{r.observed_prevalence:.3f}"] for _, r in c.iterrows()]
    return md(["Class weights", "Calibration slope (95% CI)", "Brier score", "Brier score, constant guess", "Mean prediction", "Observed prevalence"], rows,
              ["---"] + ["---:"] * 5, [1.2, 2.2, 1, 1.4, 1, 1.2])


def s_hotspots():
    o = rd("v6_orphans_floor.csv")
    rows = [[r.hgvs_p, int(r.n_path_nb), f(r.alphamissense, 3), f(r.sift, 3), f(r.polyphen2_hvar, 3),
             "absent" if pd.isna(r.gnomad_af) else f"{r.gnomad_af:.1e}"] for _, r in o.iterrows()]
    return md(["Variant", "Pathogenic neighbours", "AlphaMissense", "SIFT", "PolyPhen-2 HVAR", "gnomAD v4 frequency"], rows, ["---"] + ["---:"] * 5,
              [1.3, 1.3, 1.2, 0.8, 1.2, 1.5])


def s_earlier_path():
    g = rd("v7_estimand_grid.csv")
    names = [("v5_truncated", "Truncated retrieval, any-record rule (earlier analysis)"), ("v6_v5rcv_v4", "Complete retrieval, any-record rule, gnomAD v4"),
             ("v6_agg_v4_wider", "Complete retrieval, aggregate rule, gnomAD v4, global-frequency stand-ins")]
    cols = [("S0_scoreable", "S0 scoreable"), ("S0_neutral", "S0 neutral"), ("S1", "S1"), ("n_nb", "Count"), ("AlphaMissense", "AlphaMissense")]
    rows = []
    for ds, nm in names:
        sub = g[g.dataset == ds]; sc0 = sub[sub.score == "S0_scoreable"].iloc[0]
        r = [nm, f"{int(sc0.n_variants)} ({int(sc0.n_P)}/{int(sc0.n_B)})"]
        for sc, _ in cols:
            x = sub[sub.score == sc].iloc[0]; r.append(f"{x.auroc:.3f} ({x.ci_low:.3f} to {x.ci_high:.3f}); ties {x.tie_fraction:.2f}")
        rows.append(r)
    return md(["Dataset", "Scoreable (P/B)"] + [c[1] for c in cols], rows, ["---", "---:"] + ["---:"] * 5, [2.6, 1.1, 1.7, 1.7, 1.7, 1.7, 1.7])


TABLES = {"s_earlier_path": s_earlier_path, "retrieval_summary": retrieval_summary, "genes_main": genes_main, "verdict_counts": verdict_counts,
          "s_inclusion": s_inclusion, "s_retrieval_panel": s_retrieval_panel, "s_estimand": s_estimand,
          "s_f3": verdict_family("F3"), "s_f1": verdict_family("F1"), "s_f2": verdict_family("F2"), "s_f4": verdict_family("F4"),
          "s_f5": verdict_family("F5"), "s_f6": verdict_family("F6"), "s_retrieval_verdicts": verdict_family("retrieval audit"),
          "s_tiers": s_tiers, "s_af": s_af, "s_nested": s_nested, "s_perm": s_perm, "s_gene_extra": s_gene_extra, "s_meta": s_meta,
          "s_pm1": s_pm1, "s_sweep": s_sweep, "s_k": s_k, "s_blocks": s_earlier_cohort_blocks, "s_calib": s_calib, "s_hotspots": s_hotspots}

if __name__ == "__main__":
    for k, fn in TABLES.items():
        print("=====", k); print(fn()[:900])
