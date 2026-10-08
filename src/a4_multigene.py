"""v7 analysis 4(b,c): the same 3D-PPS evaluation in every gene that passes the inclusion rule (PREREG I1-I4 + A1),
then the across-gene meta-analysis.

Per gene (tier T2 = primary; T1 where >= 8 benign):
  v7_gene_grid.csv        AUROC (ties 0.5) with residue-cluster CI + tie decomposition for S0 (scoreable / neutral), S1, S2,
                          1D, neighbour count, coverage flag, AlphaMissense, REVEL
  v7_gene_pairs.csv       paired differences S1-AM, S1-count, ... (bootstrap SD feeds the meta-analysis)
  v7_gene_summary.csv     one row per gene: counts, scoreable fractions, coverage asymmetry, ties, null-2 p, jackknife, MDD, nested CV
  v7_gene_cv.csv          nested CV summary + contrasts per gene (F6)
  v7_gene_perm.csv        null 2 (primary) and null 1 for S0a, S0b, S1, S2
  v7_meta_*.csv / json    random-effects pooled differences, heterogeneity, meta-regression
Run: python -m src.a4_multigene
"""
from __future__ import annotations
import json, time
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from src.eng import *   # noqa
from src.inf7 import *   # noqa
from src import perm7
from src.a1_myh7 import SCORE_LIST, exact_ci

GSCORES = [("S0_scoreable", "S0"), ("S0_neutral", "S0_neutral"), ("S1", "S1"), ("S2", "S2"), ("n_nb", "n_nb"), ("flag_ge1", "covered_ge1"),
           ("S0_1d_scoreable", "S0_1d"), ("S1_1d", "S1_1d"), ("AlphaMissense", "o_alphamissense"), ("REVEL", "o_revel")]
GPAIRS = [("S1", "o_alphamissense"), ("S1", "n_nb"), ("S2", "o_alphamissense"), ("S2", "n_nb"), ("S1", "o_revel"), ("S1", "S1_1d"),
          ("S0_neutral", "n_nb"), ("S0_neutral", "o_alphamissense")]


def included_genes():
    inc = pd.read_csv(TAB / "v7_inclusion.csv")
    return inc[inc.included.astype(str).str.lower() == "true"].gene.tolist()


def gene_tier(gene, tier, nboot=N_BOOT, do_cv=True, do_perm=True):
    geom = Geom(pdb_path(gene))
    pt = pd.read_parquet(PROC / gene / "protein_table.parquet")
    ds = tier_dataset(gene, tier, geom, pt)
    y = ds.y.values; g = ds.aapos.values
    cols = {c: ds[c].values for _, c in GSCORES}
    pairs = [(a, b) for a, b in GPAIRS]
    allc = dict(cols); allc.update({c: ds[c].values for c in set(sum(pairs, ()))})
    out = boot_auc(y, g, allc, pairs=pairs, n=nboot)
    grid, prow = [], []
    for nm, col in GSCORES:
        o = out[col]; dec = pair_decomp(y, ds[col].values)
        grid.append({"gene": gene, "tier": tier, "score": nm, "n_variants": int(np.isfinite(ds[col].values).sum()), "n_P": dec["n_P"], "n_B": dec["n_B"],
                     "auroc": r3(o["auroc"]), "ci_low": r3(o["ci_low"]), "ci_high": r3(o["ci_high"]), "boot_sd": r3(o["boot_sd"], 4),
                     "p_pair_greater": r3(dec["p_greater"]), "tie_fraction": r3(dec["p_tie"]), "p_pair_smaller": r3(dec["p_smaller"]),
                     "n_distinct_scores": dec["n_distinct"]})
    for a, b in pairs:
        o = out[(a, b)]
        prow.append({"gene": gene, "tier": tier, "A": a, "B": b, "delta": r3(o["delta"]), "ci_low": r3(o["ci_low"]), "ci_high": r3(o["ci_high"]),
                     "boot_sd": r3(o["boot_sd"], 5), "mdd_80": r3(o["mdd_80"]), "n": o["n"], "p_two_sided_boot": diff_p(o["reps"]),
                     "p_tost_0.02": tost_p(o["reps"], 0.02)})
    # summary stats
    nB = int((y == 0).sum()); nP = int(y.sum())
    sc_ok = ds.S0.notna()
    asym_B = float(((ds.y == 0) & (ds.n_nb == 0)).sum() / max(nB, 1)); asym_P = float(((ds.y == 1) & (ds.n_nb == 0)).sum() / max(nP, 1))
    dec0 = pair_decomp(y, ds.S0.values); dec1 = pair_decomp(y, ds.S1.values)
    summ = {"gene": gene, "tier": tier, "L": geom.n, "n_P": nP, "n_B": nB, "P_residues": int(ds[ds.y == 1].aapos.nunique()), "B_residues": int(ds[ds.y == 0].aapos.nunique()),
            "n_B_scoreable_S0": int((sc_ok & (ds.y == 0)).sum()), "n_P_scoreable_S0": int((sc_ok & (ds.y == 1)).sum()),
            "scoreable_frac_P": r3(float((sc_ok & (ds.y == 1)).sum() / max(nP, 1))), "scoreable_frac_B": r3(float((sc_ok & (ds.y == 0)).sum() / max(nB, 1))),
            "benign_no_neighbour": r3(asym_B), "pathogenic_no_neighbour": r3(asym_P), "coverage_asymmetry_B_minus_P": r3(asym_B - asym_P),
            "benign_to_pathogenic_ratio": r3(nB / max(nP, 1)), "S0_tie_fraction": r3(dec0["p_tie"]), "S0_p_pair_greater": r3(dec0["p_greater"]),
            "S1_tie_fraction": r3(dec1["p_tie"]), "log10_nB": r3(np.log10(max(nB, 1)))}
    for nm, col in GSCORES:
        o = out[col]; summ[f"auroc_{nm}"] = r3(o["auroc"]); summ[f"ci_{nm}"] = f"{r3(o['ci_low'])} to {r3(o['ci_high'])}"
    for a, b in pairs:
        o = out[(a, b)]; summ[f"delta_{a}_minus_{b}"] = r3(o["delta"]); summ[f"sd_{a}_minus_{b}"] = r3(o["boot_sd"], 5)
    # jackknife over benign clusters
    S1 = ds.S1.values; cnt = ds.n_nb.values.astype(float); AM = ds.o_alphamissense.values
    for nm, fn in [("S1", lambda i: auroc(y[i], S1[i])), ("S1_minus_AM", lambda i: auroc(y[i], S1[i]) - auroc(y[i], AM[i])),
                   ("S1_minus_count", lambda i: auroc(y[i], S1[i]) - auroc(y[i], cnt[i]))]:
        th, se, df, lo, hi = jackknife_benign(y, g, fn)
        summ[f"jk_{nm}_se"] = r3(se, 4); summ[f"jk_{nm}_ci"] = f"{r3(lo)} to {r3(hi)}"; summ["jk_df"] = df
    # power MDD under observed tie structure
    for sname, P, B in (("S0_scoreable", ds[(ds.y == 1) & sc_ok].S0.values, ds[(ds.y == 0) & sc_ok].S0.values),
                        ("S1", ds[ds.y == 1].S1.values, ds[ds.y == 0].S1.values)):
        if len(B) >= 2 and len(P) >= 2:
            rows = power_curve(P, len(B), lower_pool(P, B), nP=len(P), nsim=1000)
            summ[f"mdd_true_auroc_{sname}"] = r3(min_detectable(rows))
    # coverage inference
    ci = coverage_inference(ds, 1)
    summ.update({"cov1_or_exact": r3(ci["or_conditional_mle"]), "cov1_or_exact_ci": f"{r3(ci['or_exact_ci_low'])} to {r3(ci['or_exact_ci_high'])}",
                 "cov1_wild_cluster_p": ci["wild_cluster_p"], "cov1_lpm_cr2_p": ci.get("lpm_p")})
    perm_rows = []
    if do_perm:
        res = perm7.run(ds, geom, nperm=perm7.N_PERM, kinds=("strata", "all"))
        for r in perm7.summarize(res):
            perm_rows.append({"gene": gene, "tier": tier, **r, "n_P": nP, "n_B": nB})
        s1p = [r for r in perm_rows if r["null"].startswith("null2") and r["score"] == "S1"][0]
        summ.update(S1_null2_p=s1p["p_one_sided"], S1_null2_mean=r3(s1p["null_mean"]), S1_null1_p=[r for r in perm_rows if r["null"].startswith("null1") and r["score"] == "S1"][0]["p_one_sided"])
    cv_s, cv_c = [], []
    if do_cv:
        oof = nested_cv(ds, geom, repeats=20)
        cv_s, cv_c = cv_summary(ds, oof, tag=f"{gene}|{tier}")
        for r in cv_s + cv_c: r.update(gene=gene, tier=tier)
        m = [r for r in cv_c if r["A"] == "M3_count_S1" and r["B"] == "M1_count"][0]
        summ.update(cv_M3_minus_M1=m["delta_mean"], cv_M3_minus_M1_ci=f"{m['ci_low']} to {m['ci_high']}", cv_M3_minus_M1_mdd=m["mdd_80"],
                    cv_M3_minus_M1_p=m["p_two_sided_boot"], cv_M3_minus_M1_tost=m["p_tost_0.02"], cv_M3_minus_M1_sd_repeats=m["delta_sd_across_repeats"],
                    cv_auroc_M0=[r for r in cv_s if r["model"] == "M0_panel"][0]["auroc_mean"], cv_auroc_M1=[r for r in cv_s if r["model"] == "M1_count"][0]["auroc_mean"],
                    cv_auroc_M3=[r for r in cv_s if r["model"] == "M3_count_S1"][0]["auroc_mean"])
    return grid, prow, summ, perm_rows, cv_s, cv_c


def meta(summary, pairs, label, genes):
    rows = []
    for (A, B) in (("S1", "o_alphamissense"), ("S1", "n_nb"), ("S2", "o_alphamissense"), ("S2", "n_nb"), ("S1", "S1_1d"), ("S0_neutral", "n_nb")):
        d = pairs[(pairs.A == A) & (pairs.B == B) & pairs.gene.isin(genes) & (pairs.tier == "T2")]
        d = d[d.boot_sd > 0]
        if len(d) < 2:
            continue
        r = random_effects(d.delta.values, d.boot_sd.values)
        r.update(set=label, A=A, B=B, genes=",".join(d.gene))
        rows.append(r)
    return rows


def main():
    t0 = time.time()
    genes = included_genes()
    print("included genes:", genes, flush=True)
    G, P, S, PM, CVS, CVC = [], [], [], [], [], []
    for gene in genes:
        for tier in ("T2", "T1"):
            pt = pd.read_parquet(PROC / gene / "protein_table.parquet")
            nB = int((pt[f"lab_{tier}"] == "B").sum())
            if tier == "T1" and nB < 8:
                continue
            gr, pr, sm_, pmr, cs, cc = gene_tier(gene, tier, do_cv=(tier == "T2"), do_perm=(tier == "T2"))
            G += gr; P += pr; S.append(sm_); PM += pmr; CVS += cs; CVC += cc
            print(f"[gene] {gene} {tier} n_B={sm_['n_B']} S1={sm_['auroc_S1']} AM={sm_['auroc_AlphaMissense']} count={sm_['auroc_n_nb']} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(G).to_csv(TAB / "v7_gene_grid.csv", index=False); pd.DataFrame(S).to_csv(TAB / "v7_gene_summary.csv", index=False)
    Gd, Pd_, Sd, PMd = pd.DataFrame(G), pd.DataFrame(P), pd.DataFrame(S), pd.DataFrame(PM)
    # F4: Holm across genes (null 2, S1, T2); F6: Holm across genes (nested M3-M1)
    Sd["S1_null2_p_holm_F4"] = np.nan
    t2 = Sd.tier == "T2"
    Sd.loc[t2, "S1_null2_p_holm_F4"] = holm(Sd.loc[t2, "S1_null2_p"].values)
    CVCd = pd.DataFrame(CVC); CVSd = pd.DataFrame(CVS)
    f6 = CVCd[(CVCd.A == "M3_count_S1") & (CVCd.B == "M1_count")].index
    CVCd["p_holm_F6"] = np.nan
    CVCd.loc[f6, "p_holm_F6"] = holm(CVCd.loc[f6, "p_two_sided_boot"].values)
    Sd["cv_M3_minus_M1_p_holm_F6"] = np.nan
    Sd.loc[t2, "cv_M3_minus_M1_p_holm_F6"] = CVCd.loc[f6].set_index("gene").loc[Sd[t2].gene, "p_holm_F6"].values
    Gd.to_csv(TAB / "v7_gene_grid.csv", index=False); Pd_.to_csv(TAB / "v7_gene_pairs.csv", index=False)
    Sd.to_csv(TAB / "v7_gene_summary.csv", index=False); PMd.to_csv(TAB / "v7_gene_perm.csv", index=False)
    CVSd.to_csv(TAB / "v7_gene_cv.csv", index=False); CVCd.to_csv(TAB / "v7_gene_cv_contrasts.csv", index=False)

    # ---- meta-analysis
    prelisted_pass = [g for g in genes if g in GENES]
    sets = {"amended_pool": genes, "prelisted_panel_passing": prelisted_pass, "amended_pool_excluding_MYH7": [g for g in genes if g != "MYH7"]}
    rows = []
    for lab_, gl in sets.items():
        rows += meta(Sd, Pd_, lab_, gl)
    M = pd.DataFrame(rows)
    M["p_holm_F5"] = np.nan
    f5 = M[(M.set == "amended_pool") & (((M.A == "S1") & (M.B == "o_alphamissense")) | ((M.A == "S1") & (M.B == "n_nb")))].index
    M.loc[f5, "p_holm_F5"] = holm(M.loc[f5, "p_hk"].values)
    M.to_csv(TAB / "v7_meta_pooled.csv", index=False)
    # meta-regression (exploratory): per-gene delta S1-AM and S1-count on log10 benign n, asymmetry, tie fraction
    mr = []
    t2d = Sd[Sd.tier == "T2"].set_index("gene")
    for (A, B) in (("S1", "o_alphamissense"), ("S1", "n_nb")):
        d = Pd_[(Pd_.A == A) & (Pd_.B == B) & (Pd_.tier == "T2") & Pd_.gene.isin(genes)].set_index("gene")
        for mod in ("log10_nB", "coverage_asymmetry_B_minus_P", "S0_tie_fraction", "benign_to_pathogenic_ratio"):
            x = t2d.loc[d.index, mod].astype(float).values; yv = d.delta.values; w = 1 / np.maximum(d.boot_sd.values, 1e-3) ** 2
            if len(x) >= 4 and np.std(x) > 0:
                res = sm.WLS(yv, sm.add_constant(x), weights=w).fit()
                rho, rp = stats.spearmanr(x, yv)
                mr.append({"A": A, "B": B, "moderator": mod, "k": len(x), "slope": float(res.params[1]), "slope_se": float(res.bse[1]), "p_wls": float(res.pvalues[1]),
                           "spearman_rho": float(rho), "spearman_p": float(rp)})
    pd.DataFrame(mr).to_csv(TAB / "v7_meta_regression.csv", index=False)
    # across-gene: does performance track benign scarcity / asymmetry? (auroc S1, auroc S1 - count)
    ac = []
    for yname in ("auroc_S1", "auroc_n_nb", "auroc_AlphaMissense", "auroc_S0_neutral", "delta_S1_minus_n_nb"):
        for mod in ("log10_nB", "coverage_asymmetry_B_minus_P", "S0_tie_fraction", "scoreable_frac_B"):
            x = t2d[mod].astype(float).values; yv = t2d[yname].astype(float).values
            if len(x) >= 4 and np.std(x) > 0 and np.std(yv) > 0:
                rho, p = stats.spearmanr(x, yv); ac.append({"outcome": yname, "moderator": mod, "k": len(x), "spearman_rho": float(rho), "p": float(p)})
    pd.DataFrame(ac).to_csv(TAB / "v7_across_gene_spearman.csv", index=False)
    print(M[["set", "A", "B", "k", "pooled", "ci_low_hk", "ci_high_hk", "tau", "I2_pct"]].to_string())
    print(f"a4 done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
