"""v7 analyses 1 and 2 (roadmap 1.2, 1.3, 2.4, 2.5, 2.6) for the index gene MYH7.

  v7_estimand_grid.csv          every score x every tier/cohort: AUROC (ties 0.5), residue-cluster CI, tie decomposition
  v7_estimand_k_sensitivity.csv S1 pseudo-count sensitivity (not primary)
  v7_contrasts_paired.csv       paired contrasts with CIs (same variants), MDD, TOST at 0.02, Holm within F3
  v7_perm_tests.csv             null 2 (primary), null 1, residue-level null for each tier x score; Holm within F1
  v7_jackknife_primary.csv      delete-one-benign-cluster jackknife CIs for the primary contrasts
  v7_pm1_call_rates.csv         rule call rates (S0 >= 0.5 / 0.7, with and without support floor) with exact CIs
  v7_af_strata_myh7.csv         AF-bin composition of ClinVar B/LB, P/LP, VUS (and stand-ins)
  v7_param_sweep_myh7.csv       sensitivity of S0/S1 AUROC to radius, sigma, gap, pLDDT floor (T2)
  v7_benign_by_tier_myh7.csv    benign protein changes with tier membership, residues, regions
Run: python -m src.a1_myh7
"""
from __future__ import annotations
import json, time
import numpy as np
import pandas as pd
from scipy.stats import beta
from src.eng import *   # noqa
from src import perm7

TIERS = ["T1", "T2", "T3", "T3legacy_wide", "T3legacy_tight", "T1both", "T3_BA1"]
LEGACY = [("v6_agg_v4_wider", "data/processed/features.parquet", "label"),
          ("v6_agg_v4_tighter", "data/processed/features.parquet", "label_tight"),
          ("v6_agg_v2", "data/spec_aggregate_v2/features.parquet", "label"),
          ("v6_v5rcv_v4", "data/spec_v5rcv_v4/features.parquet", "label"),
          ("v6_v5rcv_v2", "data/spec_v5rcv_v2/features.parquet", "label"),
          ("v5_truncated", "data/legacy_v5/features.parquet", "label")]
SCORE_LIST = [("S0_scoreable", "S0"), ("S0_neutral", "S0_neutral"), ("S0_lowest", "S0_lowest"), ("S1", "S1"), ("S2", "S2"),
              ("raw_diff", "raw_diff"), ("n_nb", "n_nb"), ("flag_ge1", "covered_ge1"), ("S0_1d_scoreable", "S0_1d"),
              ("S1_1d", "S1_1d"), ("AlphaMissense", "o_alphamissense"), ("REVEL", "o_revel")]
PAIRS = [("S0", "S0_1d"), ("S1", "S1_1d"), ("S1", "n_nb"), ("S1", "o_alphamissense"), ("S1", "o_revel"), ("S2", "n_nb"),
         ("S2", "o_alphamissense"), ("S0_neutral", "n_nb"), ("S0", "o_alphamissense"), ("S1", "S0_neutral")]
REGIONS = [("motor 1-780", 1, 780), ("lever arm 781-843", 781, 843), ("S2 844-1200", 844, 1200), ("LMM 1201-1935", 1201, 1935)]


def legacy_ds(path, col, geom):
    f = pd.read_parquet(path)
    f = f[f[col].isin(["P", "B"]) & f.aapos.between(1, geom.n)].copy()
    f["lab"] = f[col]
    return build_ds(f, geom)


def exact_ci(k, n, a=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def grid_for(name, ds, nboot):
    y = ds.y.values; g = ds.aapos.values
    sc = {nm: ds[col].values for nm, col in SCORE_LIST if col in ds}
    pairs = [(a, b) for a, b in PAIRS if a in ds and b in ds]
    cols = [c for _, c in SCORE_LIST if c in ds]
    allc = {c: ds[c].values for c in set(cols) | set(sum(pairs, ()))}
    out = boot_auc(y, g, allc, pairs=pairs, n=nboot)
    rows = []
    for nm, col in SCORE_LIST:
        if col not in ds:
            continue
        o = out[col]; dec = pair_decomp(y, ds[col].values)
        rows.append({"dataset": name, "score": nm, "n_variants": int(np.isfinite(ds[col].values).sum()),
                     "n_P": dec["n_P"], "n_B": dec["n_B"], "n_B_residues": int(ds[(ds.y == 0) & np.isfinite(ds[col].values)].aapos.nunique()),
                     "auroc": r3(o["auroc"]), "ci_low": r3(o["ci_low"]), "ci_high": r3(o["ci_high"]),
                     "p_pair_greater": r3(dec["p_greater"]), "tie_fraction": r3(dec["p_tie"]), "p_pair_smaller": r3(dec["p_smaller"]),
                     "n_distinct_scores": dec["n_distinct"]})
    prow = []
    for a, b in pairs:
        o = out[(a, b)]
        prow.append({"dataset": name, "A": a, "B": b, "delta": r3(o["delta"]), "ci_low": r3(o["ci_low"]), "ci_high": r3(o["ci_high"]),
                     "boot_sd": r3(o["boot_sd"], 4), "mdd_80": r3(o["mdd_80"]), "n": o["n"],
                     "p_two_sided_boot": diff_p(o["reps"]), "p_tost_0.02": tost_p(o["reps"], 0.02), "n_boot": len(o["reps"])})
    return rows, prow


def boot_cross(yA, sA, yB, sB, groups, n=5000, seed=SEED):
    """AUROC_A(yA,sA) - AUROC_B(yB,sB) on the same variants (different labels/scores); residue-cluster bootstrap."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed); reps = []
    tries = 0
    while len(reps) < n and tries < 4 * n:
        tries += 1
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if yA[idx].min() == yA[idx].max() or yB[idx].min() == yB[idx].max():
            continue
        reps.append(auroc(yA[idx], sA[idx]) - auroc(yB[idx], sB[idx]))
    reps = np.array(reps)
    return auroc(yA, sA) - auroc(yB, sB), reps


def shared_contrast(name, A, B, score, key, n=5000):
    """Between-label-set contrast on variants labelled in BOTH sets (AUROC_A - AUROC_B)."""
    m = A.merge(B, on=key, suffixes=("_A", "_B"))
    d, reps = boot_cross(m.y_A.values, m[f"{score}_A"].values.astype(float), m.y_B.values, m[f"{score}_B"].values.astype(float),
                         m.aapos_A.values if "aapos_A" in m else m.aapos.values, n)
    nB_A = int((m.y_A == 0).sum()); nB_B = int((m.y_B == 0).sum())
    return {"contrast": name, "score": score, "n_shared": len(m), "n_B_in_A": nB_A, "n_B_in_B": nB_B,
            "auroc_A": r3(auroc(m.y_A.values, m[f"{score}_A"].values.astype(float))),
            "auroc_B": r3(auroc(m.y_B.values, m[f"{score}_B"].values.astype(float))),
            "delta": r3(d), "ci_low": r3(np.percentile(reps, 2.5)), "ci_high": r3(np.percentile(reps, 97.5)),
            "boot_sd": r3(reps.std(), 4), "mdd_80": r3(2.8 * reps.std()), "p_two_sided_boot": diff_p(reps),
            "p_tost_0.02": tost_p(reps, 0.02), "n_boot": len(reps)}


def union_contrast(name, A, B, score, n=5000, seed=SEED):
    """AUROC_A - AUROC_B where A and B are different label sets over (partly) different variants; residue clusters
    are resampled from the union and each set is evaluated on its own variants (composition + label effects together)."""
    res = np.union1d(A.aapos.unique(), B.aapos.unique())
    mA = {r: np.where(A.aapos.values == r)[0] for r in res}; mB = {r: np.where(B.aapos.values == r)[0] for r in res}
    yA, yB = A.y.values, B.y.values; sA = A[score].values.astype(float); sB = B[score].values.astype(float)
    rng = np.random.default_rng(seed); reps = []; tries = 0
    while len(reps) < n and tries < 4 * n:
        tries += 1
        samp = rng.integers(0, len(res), len(res))
        iA = np.concatenate([mA[res[k]] for k in samp]); iB = np.concatenate([mB[res[k]] for k in samp])
        if yA[iA].min() == yA[iA].max() or yB[iB].min() == yB[iB].max():
            continue
        reps.append(auroc(yA[iA], sA[iA]) - auroc(yB[iB], sB[iB]))
    reps = np.array(reps)
    d = auroc(yA, sA) - auroc(yB, sB)
    return {"contrast": name, "score": score, "n_A": len(A), "n_B_A": int((A.y == 0).sum()), "n_B_B": int((B.y == 0).sum()), "n_B_set": len(B),
            "auroc_A": r3(auroc(yA, sA)), "auroc_B": r3(auroc(yB, sB)), "delta": r3(d), "ci_low": r3(np.percentile(reps, 2.5)),
            "ci_high": r3(np.percentile(reps, 97.5)), "boot_sd": r3(reps.std(), 4), "n_boot": len(reps)}


def main():
    t0 = time.time()
    geom = Geom(pdb_path("MYH7"))
    pt = pd.read_parquet(PROC / "MYH7" / "protein_table.parquet")
    ds = {t: tier_dataset("MYH7", t, geom, pt) for t in TIERS}
    for nm, path, col in LEGACY:
        ds[nm] = legacy_ds(path, col, geom)
    grid, prow = [], []
    for nm, d in ds.items():
        nb = 5000 if nm in ("T1", "T2", "T3") else 3000
        r, p = grid_for(nm, d, nb); grid += r; prow += p
        print(f"[grid] {nm}: n={len(d)} B={int((d.y==0).sum())} ({time.time()-t0:.0f}s)", flush=True)
    G = pd.DataFrame(grid); G.to_csv(TAB / "v7_estimand_grid.csv", index=False)
    PR = pd.DataFrame(prow)

    # ---- k sensitivity (T2)
    ks = []
    for k in (0.25, 0.5, 1.0, 2.0, 4.0):
        d = tier_dataset("MYH7", "T2", geom, pt, k=k)
        o = boot_auc(d.y.values, d.aapos.values, {"S1": d.S1.values, "S2": d.S2.values}, n=1000)
        for s in ("S1", "S2"):
            ks.append({"tier": "T2", "k": k, "score": s, "auroc": r3(o[s]["auroc"]), "ci_low": r3(o[s]["ci_low"]), "ci_high": r3(o[s]["ci_high"]),
                       "tie_fraction": r3(pair_decomp(d.y.values, d[s].values)["p_tie"])})
    pd.DataFrame(ks).to_csv(TAB / "v7_estimand_k_sensitivity.csv", index=False)

    # ---- between-set contrasts (F3 d..g)
    key = ["aaref", "aapos", "aaalt"]
    cr = []
    for score in ("S1", "S0_neutral"):
        cr.append(shared_contrast("T2 minus T1 (shared variants)", ds["T2"].merge(ds["T1"][key], on=key), ds["T1"], score, key))
        cr.append(shared_contrast("T3 minus T2", ds["T3"].merge(ds["T2"][key], on=key), ds["T2"], score, key))
        cr.append(shared_contrast("T3legacy_wide minus T3", ds["T3legacy_wide"].merge(ds["T3"][key], on=key), ds["T3"], score, key))
    # cohort contrasts at nucleotide level (hgvs_g)
    for score in ("S1", "S0_neutral"):
        a = ds["v6_agg_v4_wider"]; b = ds["v6_v5rcv_v4"]
        cr.append(shared_contrast("v6 aggregate/v4 minus any-record/v4 (shared nt variants)", a, b, score, ["hgvs_g"]))
        c = ds["v5_truncated"]
        cr.append(shared_contrast("v6 aggregate/v4 minus truncated v5 (shared nt variants)", a, c, score, ["hgvs_g"]))
        cr.append(shared_contrast("v6 any-record/v4 minus truncated v5 (shared nt variants)", b, c, score, ["hgvs_g"]))
    CR = pd.DataFrame(cr)
    uc = []
    for score in ("S1", "S0_neutral"):
        for nmA, nmB in (("T2", "T1"), ("T3", "T2"), ("T3legacy_wide", "T3"), ("T3legacy_wide", "T2"), ("T3legacy_tight", "T2"), ("T1both", "T1")):
            uc.append(union_contrast(f"{nmA} minus {nmB} (each on its own variants; composition + labels)", ds[nmA], ds[nmB], score))
    pd.DataFrame(uc).to_csv(TAB / "v7_contrasts_union_sets.csv", index=False)

    # ---- F3 table: pre-listed tests, Holm
    f3 = []
    def pick(dsname, A, B):
        r = PR[(PR.dataset == dsname) & (PR.A == A) & (PR.B == B)].iloc[0]
        return {"test": f"{dsname}: {A} minus {B}", **r.to_dict()}
    f3.append(pick("T3legacy_tight", "S0", "S0_1d"))
    f3.append(pick("T2", "S1", "S1_1d")); f3.append(pick("T2", "S1", "n_nb"))
    f3.append(pick("T2", "S1", "o_alphamissense")); f3.append(pick("T2", "S1", "o_revel"))
    for nm_ in ("T2 minus T1 (shared variants)", "T3 minus T2", "T3legacy_wide minus T3"):
        r = CR[(CR.contrast == nm_) & (CR.score == "S1")].iloc[0]; f3.append({"test": f"{nm_}: S1", **r.to_dict()})
    for nm_ in ("v6 aggregate/v4 minus any-record/v4 (shared nt variants)", "v6 aggregate/v4 minus truncated v5 (shared nt variants)"):
        r = CR[(CR.contrast == nm_) & (CR.score == "S1")].iloc[0]; f3.append({"test": f"{nm_}: S1", **r.to_dict()})
    F3 = pd.DataFrame(f3)
    F3["p_holm_F3"] = holm(F3.p_two_sided_boot.values)
    F3["family"] = "F3"
    F3.to_csv(TAB / "v7_contrasts_F3.csv", index=False)
    PR.to_csv(TAB / "v7_contrasts_paired.csv", index=False)
    CR.to_csv(TAB / "v7_contrasts_between_sets.csv", index=False)

    # ---- permutation tests
    prm = []
    for t in TIERS:
        d = ds[t]
        res = perm7.run(d, geom, nperm=perm7.N_PERM, kinds=("strata", "all"), residue_level=(t in ("T1", "T2", "T3", "T3legacy_wide")))
        for r in perm7.summarize(res):
            prm.append({"tier": t, **r, "n_P": int(d.y.sum()), "n_B": int((1 - d.y).sum())})
        if t == "T2":
            for kd, arr in res["nulls"].items():
                np.save(TAB / f"v7_perm_null_{kd}_T2.npy", arr)
        print(f"[perm] {t} ({time.time()-t0:.0f}s)", flush=True)
    PM = pd.DataFrame(prm)
    f1 = PM[(PM.null == "null2_coverage_burial_strata") & PM.tier.isin(["T1", "T2", "T3", "T3legacy_wide"])].index
    PM["p_holm_F1"] = np.nan
    PM.loc[f1, "p_holm_F1"] = holm(PM.loc[f1, "p_one_sided"].values)
    PM.to_csv(TAB / "v7_perm_tests.csv", index=False)

    # ---- jackknife over benign clusters (primary contrasts, T2 and T1)
    jk = []
    for t in ("T1", "T2", "T3"):
        d = ds[t]; y = d.y.values; g = d.aapos.values
        S1 = d.S1.values; cnt = d.n_nb.values.astype(float); AM = d.o_alphamissense.values
        for nm, fn in [("AUROC S1", lambda i: auroc(y[i], S1[i])),
                       ("S1 minus n_nb", lambda i: auroc(y[i], S1[i]) - auroc(y[i], cnt[i])),
                       ("S1 minus AlphaMissense", lambda i: auroc(y[i], S1[i]) - auroc(y[i], AM[i]))]:
            th, se, df, lo, hi = jackknife_benign(y, g, fn)
            jk.append({"tier": t, "quantity": nm, "estimate": r3(th), "jackknife_se": r3(se, 4), "df": df,
                       "ci_low_t": r3(lo), "ci_high_t": r3(hi), "n_benign_clusters": df + 1})
    pd.DataFrame(jk).to_csv(TAB / "v7_jackknife_primary.csv", index=False)

    # ---- PM1-type rule call rates (S0 thresholds, support floor)
    cr_rows = []
    for t in ("T1", "T2"):
        d = ds[t]
        for thr in (0.5, 0.7):
            for floor in (0, 3):
                called = (d.S0 >= thr) & (d.n_nb >= floor)
                for cls, lab_ in (("P", 1), ("B", 0)):
                    sub = d[d.y == lab_]; k = int(called[d.y == lab_].sum()); n = len(sub)
                    lo, hi = exact_ci(k, n)
                    cr_rows.append({"tier": t, "threshold": thr, "support_floor": floor, "class": cls, "n": n, "n_called": k,
                                    "rate": r3(k / n), "ci_low": r3(lo), "ci_high": r3(hi)})
    CRR = pd.DataFrame(cr_rows)
    lr = []
    for (t, thr, fl), g_ in CRR.groupby(["tier", "threshold", "support_floor"]):
        rP = g_[g_["class"] == "P"].iloc[0]; rB = g_[g_["class"] == "B"].iloc[0]
        lr.append({"tier": t, "threshold": thr, "support_floor": fl, "LR_plus": r3(rP.rate / rB.rate) if rB.rate else None})
    CRR = CRR.merge(pd.DataFrame(lr), on=["tier", "threshold", "support_floor"])
    CRR.to_csv(TAB / "v7_pm1_call_rates.csv", index=False)

    # ---- AF strata (protein level, legacy global AF = max over alleles of max(exome, genome) AF)
    af = pt.copy()
    af["af_bin"] = pd.cut(af.gn_af_legacy_max.fillna(0), [-1, 0, 1e-5, 1e-4, 1.1], labels=["absent (AC 0)", "<1e-5", "1e-5 to 1e-4", ">1e-4"])
    rows = []
    for cls, mask in (("ClinVar B/LB (aggregate)", af.clinvar_protein_label == "B"), ("ClinVar P/LP", af.clinvar_protein_label == "P"),
                      ("ClinVar VUS/conflicting", af.clinvar_protein_label == "VUS"), ("no ClinVar record (gnomAD only)", ~af.in_clinvar)):
        t_ = af[mask].af_bin.value_counts().reindex(["absent (AC 0)", "<1e-5", "1e-5 to 1e-4", ">1e-4"]).fillna(0).astype(int)
        for b, n in t_.items():
            rows.append({"class": cls, "af_bin": b, "n": int(n), "share_of_class": r3(n / max(int(mask.sum()), 1))})
    pd.DataFrame(rows).to_csv(TAB / "v7_af_strata_myh7.csv", index=False)
    facc = {"standins_T3_faf95_ge_2e-4": int(pt.standin_T3.sum()),
            "clinvar_VUS_protein_changes_meeting_FAF_rule": int(pt.vus_meeting_faf_rule.sum()),
            "clinvar_P_protein_changes_meeting_FAF_rule": int(pt.P_meeting_faf_rule.sum()),
            "standins_legacy_wide": int(pt.standin_legacy_wide.sum()), "standins_legacy_tight": int(pt.standin_legacy_tight.sum()),
            "clinvar_VUS_with_legacy_af_ge_1e-5": int(((pt.clinvar_protein_label == "VUS") & (pt.gn_af_legacy_max.fillna(0) >= 1e-5)).sum()),
            "clinvar_P_with_legacy_af_ge_1e-5": int(((pt.clinvar_protein_label == "P") & (pt.gn_af_legacy_max.fillna(0) >= 1e-5)).sum()),
            "clinvar_P_with_legacy_af_ge_5e-5": int(((pt.clinvar_protein_label == "P") & (pt.gn_af_legacy_max.fillna(0) >= 5e-5)).sum())}
    # ---- benign list by tier
    bl = pt[pt.lab_T2.eq("B") | pt.standin_T3 | pt.standin_legacy_wide].copy()
    bl["region"] = bl.aapos.map(lambda p: next((n for n, a, b in REGIONS if a <= p <= b), "other"))
    bl["in_T1"] = bl.lab_T1.eq("B"); bl["in_T2"] = bl.lab_T2.eq("B"); bl["in_T3"] = bl.lab_T3.eq("B")
    bl["in_legacy_wide"] = bl.lab_T3legacy_wide.eq("B"); bl["in_legacy_tight"] = bl.lab_T3legacy_tight.eq("B")
    bl[["aaref", "aapos", "aaalt", "region", "in_T1", "in_T2", "in_T3", "in_legacy_wide", "in_legacy_tight", "stars_B_max", "erepo_B", "n_nt",
        "gn_af_legacy_max", "faf95_popmax", "faf95_popmax_group", "ac_in_popmax_group", "alphamissense", "revel"]
       ].sort_values("aapos").to_csv(TAB / "v7_benign_by_tier_myh7.csv", index=False)
    # tier summary with scoreable benign counts
    ts = []
    for t in TIERS:
        d = ds[t]
        ts.append({"tier": t, "n_P": int(d.y.sum()), "n_B": int((1 - d.y).sum()), "B_residues": int(d[d.y == 0].aapos.nunique()),
                   "P_residues": int(d[d.y == 1].aapos.nunique()), "n_scoreable_S0": int(d.S0.notna().sum()),
                   "n_B_scoreable_S0": int(((d.y == 0) & d.S0.notna()).sum()), "n_P_scoreable_S0": int(((d.y == 1) & d.S0.notna()).sum()),
                   "B_no_labelled_neighbour_pct": r3(100 * ((d.y == 0) & (d.n_nb == 0)).sum() / max((d.y == 0).sum(), 1), 1),
                   "P_no_labelled_neighbour_pct": r3(100 * ((d.y == 1) & (d.n_nb == 0)).sum() / max((d.y == 1).sum(), 1), 1)})
    TS = pd.DataFrame(ts); TS.to_csv(TAB / "v7_tier_summary_myh7.csv", index=False)
    json.dump(facc, open(TAB / "v7_myh7_af_rule_counts.json", "w"), indent=1)

    # ---- parameter sweep (T2; S0a, S0b, S1) one-at-a-time
    sw = []
    base = dict(radius=RADIUS, sigma=SIGMA, win=WIN, min_plddt=MIN_PLDDT)
    for par, vals in (("radius", [6, 8, 10, 12, 15]), ("sigma", [3, 5, 8]), ("win", [2, 4, 8]), ("min_plddt", [30, 50, 70])):
        for v in vals:
            kw = dict(base); kw[par] = v
            gg = Geom(pdb_path("MYH7"), **kw)
            d = tier_dataset("MYH7", "T2", gg, pt)
            o = boot_auc(d.y.values, d.aapos.values, {"S0": d.S0.values, "S0_neutral": d.S0_neutral.values, "S1": d.S1.values, "n_nb": d.n_nb.values.astype(float)}, n=1000)
            for s in ("S0", "S0_neutral", "S1", "n_nb"):
                sw.append({"param": par, "value": v, "is_default": v == base[par], "score": s, "auroc": r3(o[s]["auroc"]),
                           "ci_low": r3(o[s]["ci_low"]), "ci_high": r3(o[s]["ci_high"]),
                           "tie_fraction": r3(pair_decomp(d.y.values, d[{"S0_neutral": "S0_neutral"}.get(s, s)].values)["p_tie"])})
    pd.DataFrame(sw).to_csv(TAB / "v7_param_sweep_myh7.csv", index=False)
    print(f"a1_myh7 done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
