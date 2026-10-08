"""v6 figures. Every number drawn is read from results/tables/*.csv or computed here.
Figure numbers live only in captions (never burned into images); ASCII text only.

Run: python -m src.figures_v6
"""
from __future__ import annotations
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.common import *   # noqa
from src.coverage_v6 import BIN_NAMES, BINS

mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 11.5, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12})
P, B, V, G, S, O = "#b3243c", "#20607a", "#e3a72f", "#9aa0a6", "#c98a1a", "#6a4c93"


def T(name): return pd.read_csv(TAB / name)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14.5, 2.6)); ax.axis("off")
    steps = ["ClinVar bulk file\nGeneSymbol list holds MYH7", "Predictors, gnomAD v4\nannotated by id", "Benign stand in\nset built",
             "AlphaFold model and\nexact 3D-PPS (abstains)", "Fold by fold\ntraining", "Coverage checks, two\nnulls, block hold out", "Tables and\nfigures"]
    w, gap = 1.0, 0.30
    for i, s in enumerate(steps):
        x = i * (w + gap); col = P if i in (3, 5) else B
        ax.add_patch(plt.Rectangle((x, 0), w, 1.0, facecolor=col, alpha=0.13, edgecolor=col, linewidth=1.2))
        ax.text(x + w / 2, 0.5, s, ha="center", va="center", fontsize=9.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.04, 0.5), xytext=(x + w + 0.04, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=G, lw=1.3))
    ax.set_xlim(-0.15, len(steps) * (w + gap) - gap + 0.15); ax.set_ylim(-0.30, 1.30)
    ax.text(0, -0.22, "Every step writes a file that can be opened and checked. Red steps are the contributions of this paper.", fontsize=9.4, color=G)
    fig.savefig(FIG / "fig1_pipeline.png", bbox_inches="tight"); plt.close(fig)


def fig_positional(df):
    orph = T("v6_orphans_floor.csv")
    d = labelled(df, "wider")
    fig, ax = plt.subplots(figsize=(14.5, 4.2))
    ax.axvspan(1, 780, color="#3a6ea5", alpha=0.08, label="Motor domain (1 to 780)")
    ax.axvspan(709, 777, color=P, alpha=0.12, label="Converter (709 to 777)")
    for lab, c, s, a in (("P", P, 10, .40), ("B", B, 22, .85)):
        g = d[d.lab.eq(lab) & d.pps3d_norm.notna()]
        ax.scatter(g.aapos, g.pps3d_norm, s=s, c=c, alpha=a, label=f"{'Pathogenic' if lab=='P' else 'Benign'} (score defined, n={len(g)})")
    ab = d[d.pps3d_norm.isna()]
    ax.scatter(ab.aapos, np.full(len(ab), -1.18), s=6, c=G, alpha=.5, marker="|", label=f"Score undefined (n={len(ab)})")
    ax.scatter(orph.aapos, orph.pps3d_norm, s=80, facecolors="none", edgecolors=O, linewidths=1.8,
               label=f"Supported hotspot candidate (n={len(orph)})")
    ax.set_xlabel("MYH7 residue position"); ax.set_ylabel("3D-PPS"); ax.set_ylim(-1.3, 1.1)
    ax.legend(fontsize=9, ncol=3, loc="lower right", frameon=False)
    fig.savefig(FIG / "fig6_positional.png", bbox_inches="tight"); plt.close(fig)


def fig_pps_by_label(df):
    d = df.copy(); d["lab"] = d.label
    d = d[d.lab.isin(["P", "B", "VUS"])]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    order = ["B", "VUS", "P"]; cols = {"B": B, "VUS": V, "P": P}
    nm = {"B": "Benign", "VUS": "VUS", "P": "Pathogenic"}
    for a, col, ttl, ylab in zip(ax, ["pps3d_norm", "coordination"], ["3D Pathogenic Proximity Score (defined only)", "Residue burial (CA coordination)"],
                                 ["3D-PPS", "C-alpha neighbours within 10 Angstroms"]):
        data = [d.loc[d.lab.eq(k), col].dropna() for k in order]
        parts = a.violinplot(data, showmeans=True, showextrema=False)
        for pc, k in zip(parts["bodies"], order):
            pc.set_facecolor(cols[k]); pc.set_alpha(0.65)
        parts["cmeans"].set_color("black")
        a.set_xticks([1, 2, 3]); a.set_xticklabels([f"{nm[k]}\n(n={len(x)})" for k, x in zip(order, data)])
        a.set_title(ttl); a.set_ylabel(ylab)
    ax[0].axhline(0, ls="--", c=G, lw=1)
    fig.tight_layout(); fig.savefig(FIG / "fig2_pps_by_label.png", bbox_inches="tight"); plt.close(fig)


def fig_ascertainment(lab):
    y = lab.y.values; st = pd.cut(lab.n_nb, bins=BINS, labels=BIN_NAMES)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    nP, nB = int((y == 1).sum()), int((y == 0).sum())
    pc = [100 * ((st == n) & (y == 1)).sum() / nP for n in BIN_NAMES]
    bc = [100 * ((st == n) & (y == 0)).sum() / nB for n in BIN_NAMES]
    x = np.arange(len(BIN_NAMES)); w = 0.38
    ax[0].bar(x - w / 2, pc, w, color=P, label=f"Pathogenic (n={nP})"); ax[0].bar(x + w / 2, bc, w, color=B, label=f"Benign (n={nB})")
    ax[0].set_xticks(x); ax[0].set_xticklabels(BIN_NAMES, fontsize=9.4)
    ax[0].set_xlabel("Labelled non-local 3D neighbours"); ax[0].set_ylabel("% of class"); ax[0].set_title("A. The coverage gap")
    ax[0].legend(fontsize=8.8, frameon=False)
    cp = 100 * (lab.n_nb[y == 1] >= 3).mean(); cb = 100 * (lab.n_nb[y == 0] >= 3).mean()
    ax[0].annotate(f"{cp:.1f}% vs {cb:.1f}% have 3 or more.", xy=(1.9, max(pc + bc) * 0.62), fontsize=8.8, color=G)
    strata = T("v6_coverage_strata.csv"); strata = strata[strata.set == "wider"]
    xs = np.arange(len(strata))
    ax[1].plot(xs, strata.alphamissense, "-s", color=B, lw=2, label="AlphaMissense")
    ax[1].plot(xs, strata.pps3d_norm.astype(float), "-o", color=P, lw=2, label="3D-PPS (undefined in first group)")
    ax[1].axhline(0.5, ls="--", c=G, lw=1)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{a}\nP={p}, B={b}" for a, p, b in zip(strata.neighbours, strata.n_P, strata.n_B)], fontsize=8.2)
    ax[1].set_ylim(0.3, 1.03); ax[1].set_ylabel("AUROC within group"); ax[1].set_xlabel("Labelled non-local 3D neighbours")
    ax[1].set_title("B. Matched on coverage"); ax[1].legend(fontsize=8.8, frameon=False, loc="lower right")
    nf, fl = len(T("v6_orphans_nofloor.csv")), len(T("v6_orphans_floor.csv"))
    ax[2].bar(["No floor\n(score >= 0.5)", "Support floor\n(3 neighbours)"], [nf, fl], color=[S, P], width=0.55)
    for i, v in enumerate([nf, fl]): ax[2].text(i, v + 0.15, str(v), ha="center", fontweight="bold")
    ax[2].set_ylabel("Hotspot candidates"); ax[2].set_title("C. The support floor")
    fig.tight_layout(w_pad=3.0); fig.savefig(FIG / "fig_v2_ascertainment.png", bbox_inches="tight"); plt.close(fig)


def fig_permutation(lab):
    perm = T("v6_permutation.csv"); pr = perm[(perm.cohort == "primary") & (perm.set == "wider")]
    n1 = np.load(TAB / "v6_permutation_null1_wider.npy"); n2 = np.load(TAB / "v6_permutation_null2_wider.npy")
    obs = float(pr.observed_auroc.iloc[0])
    p1 = float(pr[pr.null.str.startswith("null1")].p_value.iloc[0]); p2 = float(pr[pr.null.str.startswith("null2")].p_value.iloc[0])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax[0].hist(n1, bins=50, color=G, alpha=0.65, label=f"Shuffle all labels (p = {p1:.3f})")
    ax[0].hist(n2, bins=50, color=S, alpha=0.55, label=f"Shuffle within coverage and burial strata (p = {p2:.3f})")
    ax[0].axvline(obs, color=P, lw=2); ax[0].text(obs + 0.01, ax[0].get_ylim()[1] * .55, f"observed {obs:.3f}", color=P, fontsize=9.4)
    ax[0].set_xlabel("AUROC on scoreable variants"); ax[0].set_ylabel("Number of shuffles"); ax[0].set_title("A. Two label shuffling nulls")
    ax[0].legend(fontsize=7.6, frameon=False, loc="upper right")
    t4 = T("v6_table4_auroc.csv"); t4 = t4[t4.set == "wider"].set_index("predictor"); cf = T("v6_coverage_flag.csv"); cf = cf[cf.set == "wider"].iloc[0]
    names = ["Chance", "Coverage flag, 1 or more", "Coverage flag, 3 or more", "3D-PPS (scoreable)", "1D control (defined)", "AlphaMissense"]
    keys = [None, "coverage_flag_ge1", "coverage_flag_ge3", "pps3d_norm", "pps1d_norm", "alphamissense"]
    vals = [0.5] + [float(t4.loc[k, "auroc"]) for k in keys[1:]]
    lo = [0] + [float(t4.loc[k, "auroc"]) - float(t4.loc[k, "ci_low"]) for k in keys[1:]]
    hi = [0] + [float(t4.loc[k, "ci_high"]) - float(t4.loc[k, "auroc"]) for k in keys[1:]]
    ax[1].barh(np.arange(6), vals, color=[G, S, S, P, O, B], height=0.6, xerr=[lo, hi], error_kw=dict(ecolor="black", lw=1, capsize=3))
    ax[1].set_yticks(np.arange(6)); ax[1].set_yticklabels(names, fontsize=8.8); ax[1].set_xlim(0.3, 1.0)
    for i, v in enumerate(vals): ax[1].text(0.31, i, f"{v:.3f}", va="center", fontsize=8.8, color="white")
    ax[1].set_xlabel("AUROC (95% residue-cluster CI)"); ax[1].set_title("B. Coverage alone against the score")
    bl = T("v6_burial_logistic.csv"); bl = bl[(bl.set == "wider") & (bl.coverage_coding == "count")].iloc[0]
    ax[2].bar([0], [bl.spearman_count_burial], color=B, width=0.45); ax[2].set_ylim(0, 1.0)
    ax[2].set_ylabel("Spearman correlation", color=B)
    ax[2].text(0, bl.spearman_count_burial + 0.03, f"{bl.spearman_count_burial:.2f}\nVIF {bl.VIF_count_burial:.1f}", ha="center", fontsize=9.4)
    ax2 = ax[2].twinx(); ax2.bar([1], [bl.beta_coverage], color=P, width=0.45); ax2.set_ylim(0, max(1.0, bl.beta_coverage * 1.4))
    ax2.set_ylabel("Logistic coefficient", color=P); ax2.spines["right"].set_visible(True)
    ax2.text(1, bl.beta_coverage + 0.03, f"{bl.beta_coverage:.2f}\ncluster p = {bl.p_coverage_cluster:.2g}", ha="center", fontsize=9.4)
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["Coverage vs burial\n(correlation)", "Coverage vs class\n(after burial)"], fontsize=9.4)
    ax[2].set_title("C. Coverage and burial")
    fig.tight_layout(w_pad=3.5); fig.savefig(FIG / "fig_v3_permutation.png", bbox_inches="tight"); plt.close(fig)


def fig_coverage_discordance(lab):
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
    cov = [100 * (lab["n_nb"] >= k).mean() for k in range(0, 8)]
    ax[0].plot(range(0, 8), cov, "-o", color="#4a4a4a", lw=2); ax[0].fill_between(range(0, 8), cov, alpha=0.12, color="#4a4a4a")
    ax[0].set_xlabel("Minimum labelled neighbours required"); ax[0].set_ylabel("% of labelled variants scoreable")
    ax[0].set_title("A. Coverage against confidence"); ax[0].axvline(3, ls="--", c=P, lw=1)
    ax[0].annotate(f"at 3 or more: {cov[3]:.0f}% scoreable", xy=(3.15, cov[3] + 6), fontsize=9.4, color=P)
    D = T("v6_discordance_sensitivity.csv")
    order = ["lenient", "nominal", "ClinGen PP3 supporting", "stringent"]; show = ["Loose", "Usual", "ClinGen\nPP3", "Strict"]
    D = D.set_index("thresholds").loc[order].reset_index()
    ax[1].bar(range(4), D.pct_discordant, color=[G, "#4a4a4a", P, B], width=0.6)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(show, fontsize=9.4); ax[1].set_ylabel("% of evaluable VUS discordant"); ax[1].set_ylim(0, 75)
    for i, v in enumerate(D.pct_discordant): ax[1].text(i, v + 1.2, f"{v}%", ha="center", fontsize=9.4)
    ax[1].set_title("B. Disagreement depends on cut offs")
    fig.tight_layout(); fig.savefig(FIG / "fig_v2_coverage_discordance.png", bbox_inches="tight"); plt.close(fig)


def fig_calibration():
    s = T("v6_calibration_summary.csv"); s = s[s.set == "wider"]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot([0, 1], [0, 1], "--", c=G, lw=1.2, label="Perfect agreement")
    for nm, col in (("balanced", P), ("unweighted", B)):
        c = T(f"v6_calibration_wider_{nm}.csv"); r = s[s.weights == nm].iloc[0]
        ax.plot(c.mean_pred, c.frac_pos, "-o", color=col, lw=2, ms=4, label=f"{nm} weights, slope {r.cal_slope:.2f}, Brier {r.brier:.3f}")
    ax.set_xlabel("Predicted probability of pathogenic"); ax.set_ylabel("Observed share pathogenic")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8.2, loc="lower right", frameon=False)
    r = s.iloc[0]; ax.text(0.04, 0.93, f"Brier of a constant guess {r.brier_constant_prevalence:.3f}", fontsize=9.4)
    fig.tight_layout(); fig.savefig(FIG / "fig9_calibration.png", bbox_inches="tight"); plt.close(fig)


def fig_spatial():
    R = T("spatial_block_cv.csv"); R = R[(R.cohort == "primary") & (R.set == "wider") & (R.scheme == "blocks100") & (R.subset == "all held-out variants")]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    R = R.iloc[::-1]
    lo = R.auroc - R.ci_low; hi = R.ci_high - R.auroc
    ax.barh(np.arange(len(R)), R.auroc, xerr=[lo, hi], color=[B if "Alpha" in m or "REVEL" in m else (P if "3D" in m else G) for m in R.method],
            error_kw=dict(ecolor="black", lw=1, capsize=3), height=0.6)
    ax.set_yticks(np.arange(len(R))); ax.set_yticklabels([f"{m} (n={n}, benign={b})" for m, n, b in zip(R.method, R.n, R.n_B)], fontsize=8.4)
    ax.axvline(0.5, ls="--", c=G, lw=1); ax.set_xlim(0.3, 1.02); ax.set_xlabel("AUROC on variants in held-out 100-residue blocks")
    fig.tight_layout(); fig.savefig(FIG / "fig_v6_spatial_block_cv.png", bbox_inches="tight"); plt.close(fig)


def fig_gap():
    g = T("v6_diag_residue_blocks.csv")
    g.columns = ["block"] + list(g.columns[1:])
    x = np.arange(len(g)); w = 0.4
    fig, ax = plt.subplots(figsize=(13.5, 3.8))
    ax.bar(x + w / 2, g.v6_clinvar_missense, w, color=B, label="Complete ClinVar release (this work)")
    ax.bar(x - w / 2, g.old_clinvar_missense, w, color=P, label="v5 MyVariant symbol query")
    ax.set_xticks(x); ax.set_xticklabels([b.strip("()").replace(", ", "-").replace("]", "") for b in g.block], rotation=60, fontsize=8)
    ax.set_ylabel("ClinVar missense variants per 100 residues"); ax.set_xlabel("MYH7 residue block"); ax.legend(frameon=False, fontsize=9.4)
    fig.tight_layout(); fig.savefig(FIG / "fig_v6_data_gap.png", bbox_inches="tight"); plt.close(fig)


def main():
    df = load(); lab = pd.read_parquet("data/processed/labelled_wider.parquet")
    fig_pipeline(); fig_positional(df); fig_pps_by_label(df); fig_ascertainment(lab); fig_permutation(lab)
    fig_coverage_discordance(lab); fig_calibration(); fig_spatial(); fig_gap()
    print(f"[figures_v6] saved to {FIG}")


if __name__ == "__main__":
    main()
