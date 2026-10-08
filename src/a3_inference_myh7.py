"""v7 analysis 3 (roadmap 1.4, 2.5) for MYH7: repeated grouped CV with nested models, simulation power under the
observed tie structure, small-cluster inference for the coverage association.

  v7_cv_nested_summary.csv      model AUROC mean and SD over 20 repeats (residue-grouped; 100-residue blocks) per tier
  v7_cv_nested_contrasts.csv    nested contrasts with CI, MDD, TOST at 0.02, Holm (F2)
  v7_cv_legacy_single_split.csv the v6 design (one ungrouped StratifiedKFold seed 42) for comparison
  v7_power_sim_myh7.csv         power curve under the observed tie structure; v7_power_summary_myh7.csv
  v7_coverage_inference_myh7.csv exact / Haldane / wild cluster bootstrap / CR2 for the coverage association
Run: python -m src.a3_inference_myh7
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.eng import *   # noqa
from src.inf7 import *   # noqa
from src.inf7 import _design, _lr


def legacy_single_split(ds, geom):
    y = ds.y.values; g = ds.aapos.values; panel_cols = [c for c in PANEL if c in ds.columns]
    oof = {m: np.zeros(len(ds)) for m in MODELS}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(ds, y):
        for m in MODELS:
            X = _design(ds, geom, tr, m, panel_cols)
            oof[m][te] = _lr().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    rows = []
    for A, B, q in CONTRASTS:
        o = boot_auc(y, g, {"a": oof[A], "b": oof[B]}, pairs=[("a", "b")], n=3000)[("a", "b")]
        rows.append({"design": "legacy v6-style single ungrouped StratifiedKFold(5, seed 42)", "A": A, "B": B, "question": q,
                     "auroc_A": r3(auroc(y, oof[A])), "auroc_B": r3(auroc(y, oof[B])), "delta": r3(o["delta"]),
                     "ci_low": r3(o["ci_low"]), "ci_high": r3(o["ci_high"])})
    return rows


def main():
    t0 = time.time()
    geom = Geom(pdb_path("MYH7"))
    pt = pd.read_parquet(PROC / "MYH7" / "protein_table.parquet")
    summ, contr, legacy, pw, pws, cov = [], [], [], [], [], []
    for tier in ("T1", "T2", "T3"):
        ds = tier_dataset("MYH7", tier, geom, pt)
        for group in ("residue", "block100"):
            oof = nested_cv(ds, geom, repeats=20, group="residue" if group == "residue" else "block")
            s, c = cv_summary(ds, oof, tag=f"{tier}|{group}")
            for r in s: r.update(tier=tier, grouping=group, n_P=int(ds.y.sum()), n_B=int((1 - ds.y).sum()))
            for r in c: r.update(tier=tier, grouping=group, n_B=int((1 - ds.y).sum()))
            summ += s; contr += c
            print(f"[cv] {tier} {group} ({time.time()-t0:.0f}s)", flush=True)
        legacy += [{"tier": tier, **r} for r in legacy_single_split(ds, geom)]
        # coverage inference
        for k in (1, 3):
            cov.append({"tier": tier, **coverage_inference(ds, k)})
        # power
        P_s0 = ds[(ds.y == 1) & ds.S0.notna()].S0.values; B_s0 = ds[(ds.y == 0) & ds.S0.notna()].S0.values
        P_s1 = ds[ds.y == 1].S1.values; B_s1 = ds[ds.y == 0].S1.values
        for sname, P, B in (("S0_scoreable", P_s0, B_s0), ("S1", P_s1, B_s1)):
            pool = lower_pool(P, B)
            rows = power_curve(P, len(B), pool, nP=len(P))
            for r in rows: pw.append({"tier": tier, "score": sname, "n_P": len(P), "n_B": len(B), **r})
            mdd = min_detectable(rows)
            need = {}
            for tgt in (0.7, 0.8):
                need[tgt] = None
                for nB in (7, 10, 15, 20, 30, 50, 75, 100, 150, 250):
                    rr = power_curve(P, nB, pool, targets=(tgt,), nsim=300, nP=len(P))
                    if rr[0]["power"] is not None and rr[0]["power"] >= 0.8:
                        need[tgt] = nB; break
            pws.append({"tier": tier, "score": sname, "n_P": len(P), "n_B_observed": len(B), "n_distinct_scores": int(len(np.unique(np.concatenate([P, B])))),
                        "min_detectable_true_auroc_80pct": r3(mdd), "benign_n_needed_for_80pct_power_at_auroc_0.7": need[0.7],
                        "benign_n_needed_for_80pct_power_at_auroc_0.8": need[0.8],
                        "caveat": "unconditional score-level permutation test; null 2 conditions on coverage so its power is lower"})
        print(f"[power/cov] {tier} ({time.time()-t0:.0f}s)", flush=True)
    S = pd.DataFrame(summ); S.to_csv(TAB / "v7_cv_nested_summary.csv", index=False)
    C = pd.DataFrame(contr)
    # F2: Holm across 4 contrasts x 3 tiers (residue-grouped primary)
    f2 = C[C.grouping == "residue"].index
    C["p_holm_F2"] = np.nan
    C.loc[f2, "p_holm_F2"] = holm(C.loc[f2, "p_two_sided_boot"].values)
    C.to_csv(TAB / "v7_cv_nested_contrasts.csv", index=False)
    pd.DataFrame(legacy).to_csv(TAB / "v7_cv_legacy_single_split.csv", index=False)
    pd.DataFrame(pw).to_csv(TAB / "v7_power_sim_myh7.csv", index=False)
    pd.DataFrame(pws).to_csv(TAB / "v7_power_summary_myh7.csv", index=False)
    pd.DataFrame(cov).to_csv(TAB / "v7_coverage_inference_myh7.csv", index=False)
    print(f"a3 done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
