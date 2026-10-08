"""v6 coverage / headline analysis (exact 3D-PPS with abstention, cluster-robust inference).

Produces (results/tables): v6_abstention_by_class, v6_table4_auroc, v6_coverage_flag,
v6_coverage_strata, v6_head_to_head, v6_3d_vs_1d, v6_burial_logistic, v6_sensitivity_specs.

Run: python -m src.coverage_v6
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.common import *   # noqa
from src.build_labels import spec_dir

BINS = [-0.1, 0.5, 2.5, 5.5, 10.5, 1e9]
BIN_NAMES = ["0", "1 to 2", "3 to 5", "6 to 10", "over 10"]
TABLE4 = ["pps3d_norm", "pps1d_norm", "alphamissense", "revel", "clinpred", "esm1b", "sift",
          "polyphen2_hvar", "cadd_phred"]


def strata_of(lab):
    return pd.cut(lab["n_nb"], bins=BINS, labels=BIN_NAMES)


def abstention_table(lab, setname):
    rows = []
    for cls, g in [("P", lab[lab.lab.eq("P")]), ("B", lab[lab.lab.eq("B")]), ("all", lab)]:
        rows.append({"set": setname, "class": cls, "n": len(g),
                     "score_undefined_n": int((~g.score_defined).sum()),
                     "score_undefined_pct": r3(100 * (~g.score_defined).mean(), 1),
                     "no_labelled_neighbour_pct": r3(100 * (g.n_nb == 0).mean(), 1),
                     "covered_ge3_pct": r3(100 * (g.n_nb >= 3).mean(), 1),
                     "covered_ge3_gated_pct": r3(100 * (g.n_nb_gated >= 3).mean(), 1)})
    return rows


def table4(lab, setname):
    y, g = lab.y.values, lab.aapos.values
    sc = lab.score_defined.values
    rows = []
    feats = {c: oriented(lab, c).values for c in TABLE4}
    feats["coverage_flag_ge1"] = (lab.n_nb >= 1).astype(float).values
    feats["coverage_flag_ge3"] = (lab.n_nb >= 3).astype(float).values
    for c, s in feats.items():
        a, lo, hi, n = auroc_ci(y, s, g)
        # same-variant comparison: restricted to variants where 3D-PPS is defined
        s2 = np.where(sc, s, np.nan)
        a2, lo2, hi2, n2 = auroc_ci(y, s2, g)
        m = ~np.isnan(s)
        rows.append({"set": setname, "predictor": c, "n": n, "n_P": int(y[m].sum()),
                     "n_B": int((1 - y[m]).sum()), "auroc": r3(a), "ci_low": r3(lo), "ci_high": r3(hi),
                     "n_on_3d_scoreable": n2, "auroc_on_3d_scoreable": r3(a2),
                     "ci_low_on_3d_scoreable": r3(lo2), "ci_high_on_3d_scoreable": r3(hi2)})
    return rows


def coverage_stats(lab, setname):
    y, g = lab.y.values, lab.aapos.values
    out = {"set": setname}
    for k in (1, 3):
        f = (lab.n_nb >= k).astype(float).values
        a, lo, hi, _ = auroc_ci(y, f, g)
        out[f"flag_ge{k}_auroc"] = r3(a); out[f"flag_ge{k}_ci_low"] = r3(lo); out[f"flag_ge{k}_ci_high"] = r3(hi)
    # headline 3D-PPS (scoreable only) and the share of the distance from chance
    s3 = lab.pps3d_norm.values
    a3, lo3, hi3, n3 = auroc_ci(y, s3, g)
    out.update(auroc_3d_scoreable=r3(a3), auroc_3d_ci_low=r3(lo3), auroc_3d_ci_high=r3(hi3), n_scoreable=n3)
    f1 = (lab.n_nb >= 1).astype(float).values
    # share of the gap above chance reachable from coverage alone; undefined when 3D-PPS is not
    # meaningfully above chance (denominator ~0 or negative)
    ok_den = a3 > 0.55
    out["share_of_gap_flag1"] = r3((out["flag_ge1_auroc"] - .5) / (a3 - .5), 3) if ok_den else None
    out["share_of_gap_flag3"] = r3((out["flag_ge3_auroc"] - .5) / (a3 - .5), 3) if ok_den else None
    sc = ~np.isnan(s3)
    def ratio(i, k=1):
        ff = (lab.n_nb.values[i] >= k).astype(float)
        yy = y[i]; ss = s3[i]; m = ~np.isnan(ss)
        if len(np.unique(yy[m])) < 2: return np.nan
        return (auroc(yy, ff) - .5) / (auroc(yy[m], ss[m]) - .5)
    for k in (1, 3):
        if ok_den:
            b = boot_stats(lambda i: ratio(i, k), y, g)
            out[f"share_flag{k}_ci_low"], out[f"share_flag{k}_ci_high"] = r3(np.percentile(b, 2.5)), r3(np.percentile(b, 97.5))
    # odds ratio of being covered (>=3) P vs B: naive Fisher vs residue-clustered logistic
    cov = (lab.n_nb >= 3).astype(int).values
    tab = [[int(((y == 1) & (cov == 1)).sum()), int(((y == 1) & (cov == 0)).sum())],
           [int(((y == 0) & (cov == 1)).sum()), int(((y == 0) & (cov == 0)).sum())]]
    orr, pf = fisher_exact(tab)
    out.update(fisher_table=str(tab), or_fisher_naive=r3(orr, 2), p_fisher_naive=float(pf))
    try:
        fit = sm.Logit(cov, sm.add_constant(y.astype(float))).fit(disp=0, cov_type="cluster",
                                                                   cov_kwds={"groups": g})
        out.update(or_cluster_logit=r3(np.exp(fit.params[1]), 2), p_cluster_logit=float(fit.pvalues[1]),
                   or_cluster_ci_low=r3(np.exp(fit.conf_int()[1][0]), 2), or_cluster_ci_high=r3(np.exp(fit.conf_int()[1][1]), 2))
    except Exception as e:  # noqa
        out["or_cluster_logit_error"] = str(e)
    def orfn(i):
        c, yy = cov[i], y[i]
        a_, b_ = ((yy == 1) & (c == 1)).sum() + .5, ((yy == 1) & (c == 0)).sum() + .5
        c_, d_ = ((yy == 0) & (c == 1)).sum() + .5, ((yy == 0) & (c == 0)).sum() + .5
        return np.log(a_ * d_ / (b_ * c_))
    b = boot_stats(orfn, y, g)
    out["or_boot_ci_low"], out["or_boot_ci_high"] = r3(np.exp(np.percentile(b, 2.5)), 2), r3(np.exp(np.percentile(b, 97.5)), 2)
    out["relative_likelihood_covered_P_over_B"] = r3((cov[y == 1].mean()) / max(cov[y == 0].mean(), 1e-9), 2)
    return out


def strata_table(lab, setname):
    y, g = lab.y.values, lab.aapos.values
    st = strata_of(lab)
    rows = []
    for nm in BIN_NAMES:
        m = (st == nm).values
        yy = y[m]
        row = {"set": setname, "neighbours": nm, "n": int(m.sum()), "n_P": int(yy.sum()), "n_B": int((1 - yy).sum())}
        for c in ("pps3d_norm", "pps1d_norm", "alphamissense", "revel"):
            s = oriented(lab, c).values[m]
            ok = ~np.isnan(s)
            row[c] = r3(auroc(yy[ok], s[ok])) if len(np.unique(yy[ok])) > 1 else None
            row[c + "_n"] = int(ok.sum())
        row["note"] = "too few benign (<10)" if row["n_B"] < 10 else ""
        rows.append(row)
    return rows


def head_to_head(lab, setname):
    y, g = lab.y.values, lab.aapos.values
    rows = []
    pairs = [("pps3d_norm", "alphamissense"), ("pps3d_norm", "revel"), ("pps3d_norm", "pps1d_norm")]
    subsets = {"all labelled (pairwise complete)": np.ones(len(lab), bool),
               "3D-PPS scoreable": lab.score_defined.values,
               "covered (>=3 labelled neighbours)": (lab.n_nb >= 3).values}
    for sub, m in subsets.items():
        for a, b in pairs:
            sA, sB = oriented(lab, a).values[m], oriented(lab, b).values[m]
            if a == "pps3d_norm" and b == "pps1d_norm":
                pass
            r = paired_diff(y[m], sA, sB, g[m])
            yy = y[m][~(np.isnan(sA) | np.isnan(sB))]
            rows.append({"set": setname, "subset": sub, "comparison": f"{a} - {b}", "n": r["n"],
                         "n_P": int(yy.sum()), "n_B": int((1 - yy).sum()),
                         **{k: (r3(v) if isinstance(v, float) else v) for k, v in r.items() if k != "n"}})
    return rows


def burial_logistic(lab, setname):
    y, g = lab.y.values.astype(float), lab.aapos.values
    bur = lab.coordination.values.astype(float)
    cnt = lab.n_nb.values.astype(float)
    ok = np.isfinite(bur)
    out = []
    codings = {"count": cnt, "flag_ge1": (cnt >= 1).astype(float), "flag_ge3": (cnt >= 3).astype(float)}
    for nm, cv in codings.items():
        X = sm.add_constant(np.column_stack([cv[ok], bur[ok]]))
        fn = sm.Logit(y[ok], X).fit(disp=0)
        fc = sm.Logit(y[ok], X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g[ok]})
        out.append({"set": setname, "coverage_coding": nm, "n": int(ok.sum()),
                    "beta_coverage": r3(fc.params[1]), "p_coverage_naive": float(fn.pvalues[1]),
                    "p_coverage_cluster": float(fc.pvalues[1]),
                    "beta_burial": r3(fc.params[2], 4), "p_burial_naive": float(fn.pvalues[2]),
                    "p_burial_cluster": float(fc.pvalues[2])})
    Xv = sm.add_constant(np.column_stack([cnt[ok], bur[ok]]))
    vif = [variance_inflation_factor(Xv, i) for i in (1, 2)]
    rho, prho = spearmanr(cnt[ok], bur[ok])
    rho_pl, _ = spearmanr(cnt[ok], lab.plddt.values[ok])
    for o in out:
        o["VIF_count_burial"] = r3(vif[0], 2); o["spearman_count_burial"] = r3(rho); o["p_spearman_naive"] = float(prho)
        o["spearman_count_plddt"] = r3(rho_pl)
    return out


def sens_headline(df, tag):
    """Light headline for sensitivity cohorts (no bootstrap)."""
    rows = []
    for setname in SETS:
        lab = labelled(df, setname)
        y = lab.y.values
        cov = (lab.n_nb >= 3).astype(int).values
        tab = [[((y == 1) & (cov == 1)).sum(), ((y == 1) & (cov == 0)).sum()],
               [((y == 0) & (cov == 1)).sum(), ((y == 0) & (cov == 0)).sum()]]
        sc = lab.score_defined.values
        rows.append({"cohort": tag, "set": setname, "n_P": int(y.sum()), "n_B": int((1 - y).sum()),
                     "n_VUS": int(df.label.eq("VUS").sum()),
                     "n_scoreable": int(sc.sum()), "abstain_P_pct": r3(100 * (~sc[y == 1]).mean(), 1),
                     "abstain_B_pct": r3(100 * (~sc[y == 0]).mean(), 1),
                     "auroc_3d_scoreable": r3(auroc(y[sc], lab.pps3d_norm.values[sc])),
                     "auroc_alphamissense_all": r3(auroc(y[~lab.alphamissense.isna().values], lab.alphamissense.dropna().values)),
                     "auroc_alphamissense_on_3d_scoreable": r3(auroc(y[sc & ~lab.alphamissense.isna().values], lab.alphamissense.values[sc & ~lab.alphamissense.isna().values])),
                     "flag_ge1_auroc": r3(auroc(y, (lab.n_nb >= 1).astype(float))),
                     "flag_ge3_auroc": r3(auroc(y, (lab.n_nb >= 3).astype(float))),
                     "covered_ge3_P_pct": r3(100 * cov[y == 1].mean(), 1), "covered_ge3_B_pct": r3(100 * cov[y == 0].mean(), 1),
                     "or_covered_fisher_naive": r3(fisher_exact(tab)[0], 2)})
    return rows


def main():
    df = load()
    A, B, C, D, E, F, G = [], [], [], [], [], [], []
    for setname in SETS:
        lab = labelled(df, setname)
        lab.to_parquet(f"data/processed/labelled_{setname}.parquet", index=False)
        A += abstention_table(lab, setname); B += table4(lab, setname)
        C.append(coverage_stats(lab, setname)); D += strata_table(lab, setname)
        E += head_to_head(lab, setname); F += burial_logistic(lab, setname)
        print(f"[coverage] {setname}: P={int(lab.y.sum())} B={int((1-lab.y).sum())} done")
    pd.DataFrame(A).to_csv(TAB / "v6_abstention_by_class.csv", index=False)
    pd.DataFrame(B).to_csv(TAB / "v6_table4_auroc.csv", index=False)
    pd.DataFrame(C).to_csv(TAB / "v6_coverage_flag.csv", index=False)
    pd.DataFrame(D).to_csv(TAB / "v6_coverage_strata.csv", index=False)
    pd.DataFrame(E).to_csv(TAB / "v6_head_to_head.csv", index=False)
    pd.DataFrame(F).to_csv(TAB / "v6_burial_logistic.csv", index=False)
    # sensitivity cohorts
    S = []
    for af, rule, tag in [("v4", "aggregate", "PRIMARY: full ClinVar aggregate labels, gnomAD v4"),
                          ("v4", "v5rcv", "full data, v5 any-RCV label rule, gnomAD v4"),
                          ("v2", "aggregate", "full ClinVar aggregate labels, gnomAD v2.1.1"),
                          ("v2", "v5rcv", "full data, v5 label rule + v2.1.1 (closest to v5)")]:
        S += sens_headline(load(spec_dir(af, rule)), tag)
    pd.DataFrame(S).to_csv(TAB / "v6_sensitivity_specs.csv", index=False)
    print(pd.DataFrame(S).to_string())
    print(pd.DataFrame(A).to_string()); print(pd.DataFrame(C).T.to_string())


if __name__ == "__main__":
    main()
