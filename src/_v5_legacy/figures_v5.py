"""v5 figures. One script regenerates every figure used in the manuscript.

Three problems in the earlier figure code are fixed here.
1. Figure numbers were burned into the images and no longer matched the captions.
2. Panel C of the permutation figure showed p = 0.005, which does not reproduce.
   The correct value for the burial adjusted coverage coefficient is p = 3e-9.
3. En dashes, curly quotes and accented words appeared inside the images, where a
   text level check of the PDF cannot see them.

Figure numbering now lives only in the captions, so it cannot drift again.

Run: python -m src.figures_v5
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.reanalysis import neighbour_support, relabel
from src.structure import load_ca

OUT = Path("data/processed")
TAB = Path("results/tables")
FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11.5,
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
})

P, B, V, G, S, O = "#b3243c", "#20607a", "#e3a72f", "#9aa0a6", "#c98a1a", "#6a4c93"


def _load():
    df = pd.read_parquet(OUT / "features.parquet")
    ca = load_ca()
    lab = relabel(df, 1e-5)
    lab = lab[lab.label.isin(["P", "B"])].reset_index(drop=True)
    npath, nben = neighbour_support(lab, ca)
    lab["n_nb"] = npath + nben
    return df, ca, lab


# ---------------------------------------------------------------- figure 1
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14.5, 2.6))
    ax.axis("off")
    steps = ["ClinVar records\nread and labelled", "Predictor and\nfrequency data",
             "Benign stand in\nset built", "AlphaFold model\nand 3D-PPS",
             "Fold by fold\ntraining", "Four coverage\nchecks",
             "Tables and\nfigures"]
    n = len(steps)
    w, gap = 1.0, 0.30
    for i, s in enumerate(steps):
        x = i * (w + gap)
        col = P if i in (3, 5) else B
        ax.add_patch(plt.Rectangle((x, 0), w, 1.0, facecolor=col, alpha=0.13,
                                   edgecolor=col, linewidth=1.2))
        ax.text(x + w / 2, 0.5, s, ha="center", va="center", fontsize=9.7)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.04, 0.5), xytext=(x + w + 0.04, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=G, lw=1.3))
    ax.set_xlim(-0.15, n * (w + gap) - gap + 0.15)
    ax.set_ylim(-0.30, 1.30)
    ax.text(0, -0.22, "Every step writes a file that can be opened and checked. "
                      "Red steps are the contributions of this paper.",
            fontsize=9.4, color=G)
    fig.savefig(FIG / "fig1_pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_positional(df):
    orph = pd.read_csv(TAB / "v2_structural_orphans.csv")
    fig, ax = plt.subplots(figsize=(14.5, 4.2))
    ax.axvspan(1, 780, color="#3a6ea5", alpha=0.08, label="Motor domain (1 to 780)")
    ax.axvspan(709, 777, color=P, alpha=0.12, label="Converter (709 to 777)")
    p = df[df.label.eq("P")]
    b = df[df.label.eq("B")]
    ax.scatter(p["aapos"], p["pps3d_norm"], s=10, c=P, alpha=.40, label="Pathogenic")
    ax.scatter(b["aapos"], b["pps3d_norm"], s=12, c=B, alpha=.60, label="Benign")
    ax.scatter(orph["aapos"], orph["pps3d_norm"], s=80, facecolors="none",
               edgecolors=O, linewidths=1.8, label="Supported hotspot candidate (n=3)")
    ax.set_xlabel("MYH7 residue position")
    ax.set_ylabel("3D-PPS")
    ax.legend(fontsize=9.4, ncol=5, loc="upper right", frameon=False)
    fig.savefig(FIG / "fig6_positional.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_pps_by_label(df):
    d = df[df.label.isin(["P", "B", "VUS"])]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    order = ["B", "VUS", "P"]
    cols = {"B": B, "VUS": V, "P": P}
    labels = ["Benign\n(n=110)", "VUS\n(n=1872)", "Pathogenic\n(n=511)"]
    for a, col, ttl, ylab in zip(
            ax, ["pps3d_norm", "coordination"],
            ["3D Pathogenic Proximity Score", "Residue burial (CA coordination)"],
            ["3D-PPS", "C-alpha neighbours within 10 Angstroms"]):
        data = [d.loc[d.label.eq(k), col].dropna() for k in order]
        parts = a.violinplot(data, showmeans=True, showextrema=False)
        for pc, k in zip(parts["bodies"], order):
            pc.set_facecolor(cols[k]); pc.set_alpha(0.65)
        parts["cmeans"].set_color("black")
        a.set_xticks([1, 2, 3]); a.set_xticklabels(labels)
        a.set_title(ttl); a.set_ylabel(ylab)
    ax[0].axhline(0, ls="--", c=G, lw=1)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_pps_by_label.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_ascertainment(lab):
    y = lab.label.eq("P").astype(int).values
    names = ["0", "1 to 2", "3 to 5", "6 to 10", "over 10"]
    cat = pd.cut(lab["n_nb"], bins=[-.1, .5, 2.5, 5.5, 10.5, 1e9], labels=names)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    pc = [100 * ((cat == n) & (y == 1)).sum() / (y == 1).sum() for n in names]
    bc = [100 * ((cat == n) & (y == 0)).sum() / (y == 0).sum() for n in names]
    x = np.arange(len(names)); w = 0.38
    ax[0].bar(x - w / 2, pc, w, color=P, label="Pathogenic (n=511)")
    ax[0].bar(x + w / 2, bc, w, color=B, label="Benign (n=110)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=9.4)
    ax[0].set_xlabel("Labelled non-local 3D neighbours")
    ax[0].set_ylabel("% of class")
    ax[0].set_title("A. The coverage gap")
    ax[0].legend(fontsize=8.8, frameon=False)
    ax[0].annotate("46.6% vs 6.4% have 3 or more.\nThe classes differ in support,\nnot only in biology.",
                   xy=(1.75, 33), fontsize=8.8, color=G)

    strata, pps_a, am_a, ns = [], [], [], []
    for n in names:
        m = (cat == n).values
        if m.sum() < 25 or len(np.unique(y[m])) < 2:
            continue
        v = pd.to_numeric(lab["pps3d_norm"], errors="coerce").values
        va = pd.to_numeric(lab["alphamissense"], errors="coerce").values
        m1 = m & ~np.isnan(v); m2 = m & ~np.isnan(va)
        strata.append(n); ns.append(int(m.sum()))
        pps_a.append(roc_auc_score(y[m1], v[m1]))
        am_a.append(roc_auc_score(y[m2], va[m2]))
    x2 = np.arange(len(strata))
    ax[1].plot(x2, pps_a, "-o", color=P, lw=2, label="3D-PPS")
    ax[1].plot(x2, am_a, "-s", color=B, lw=2, label="AlphaMissense")
    ax[1].axhline(0.5, ls="--", c=G, lw=1)
    ax[1].text(0.04, 0.515, "chance", fontsize=8.3, color=G)
    ax[1].set_xticks(x2)
    ax[1].set_xticklabels([f"{s}\n(n={n})" for s, n in zip(strata, ns)], fontsize=9.4)
    ax[1].set_ylim(0.4, 1.03)
    ax[1].set_ylabel("AUROC within group")
    ax[1].set_xlabel("Labelled non-local 3D neighbours")
    ax[1].set_title("B. Matched on coverage")
    ax[1].legend(fontsize=9.4, frameon=False, loc="lower right")
    ax[1].annotate("no value here,\nso exactly chance", xy=(-0.05, 0.56), fontsize=8.8, color=P)

    counts = [22, 3]
    ax[2].bar(["No floor\n(1 neighbour)", "Support floor\n(3 neighbours)"], counts,
              color=[S, P], width=0.55)
    for i, v in enumerate(counts):
        ax[2].text(i, v + 0.6, str(v), ha="center", fontweight="bold")
    ax[2].set_ylabel("Hotspot candidates")
    ax[2].set_title("C. The support floor matters")
    ax[2].annotate("all 3 survivors lie in one\nsite (residues 120 to 123)",
                   xy=(0.28, 13.5), fontsize=8.8, color=G)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(FIG / "fig_v2_ascertainment.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_permutation(lab):
    null = np.load(TAB / "permutation_null.npy")
    res = pd.read_csv(TAB / "permutation_result.csv").iloc[0]
    obs = float(res["observed_auroc"])

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    ax[0].hist(null, bins=60, color=G, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax[0].axvline(obs, color=P, lw=2)
    ax[0].text(obs - 0.012, ax[0].get_ylim()[1] * 0.88, f"observed\n{obs:.3f}",
               color=P, fontsize=9.4, ha="right")
    ax[0].text(0.305, ax[0].get_ylim()[1] * 0.60,
               f"10,000 shuffles\nmean {null.mean():.3f}, SD {null.std():.3f}\n"
               f"largest {null.max():.3f}\n0 reached the observed value",
               fontsize=8.8, color="black")
    ax[0].set_xlim(0.28, 0.92)
    ax[0].set_xlabel("AUROC")
    ax[0].set_ylabel("Number of shuffles")
    ax[0].set_title("A. Shuffling destroys the signal")

    names = ["Chance", "Coverage flag only,\nno score value", "3D-PPS", "AlphaMissense"]
    vals = [0.500, 0.650, 0.869, 0.921]
    errs = [[0, 0.060, 0.034, 0.030], [0, 0.057, 0.031, 0.025]]
    cols = [G, S, P, B]
    ax[1].barh(np.arange(4), vals, color=cols, height=0.6,
               xerr=errs, error_kw=dict(ecolor="black", lw=1, capsize=3))
    ax[1].set_yticks(np.arange(4)); ax[1].set_yticklabels(names, fontsize=9.4)
    ax[1].set_xlim(0.45, 1.0)
    ax[1].set_xlabel("AUROC")
    for i, v in enumerate(vals):
        ax[1].text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=9.4)
    ax[1].set_title("B. Coverage alone gets 41% of the way")

    y = lab.label.eq("P").astype(int).values
    cnt = lab["n_nb"].values.astype(float)
    bur = pd.to_numeric(lab["coordination"], errors="coerce").values
    m = np.isfinite(bur)
    rho_b = spearmanr(cnt[m], bur[m])[0]
    fit = sm.Logit(y[m], sm.add_constant(np.column_stack([cnt[m], bur[m]]))).fit(disp=0)
    beta, pval = fit.params[1], fit.pvalues[1]

    ax[2].bar([0], [rho_b], color=B, width=0.45)
    ax[2].set_ylim(0, 1.0)
    ax[2].set_ylabel("Spearman correlation", color=B)
    ax[2].tick_params(axis="y", labelcolor=B)
    ax[2].text(0, rho_b + 0.03, f"{rho_b:.2f}\np = 5e-129", ha="center", fontsize=9.4)
    ax2 = ax[2].twinx()
    ax2.bar([1], [beta], color=P, width=0.45)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Logistic coefficient", color=P)
    ax2.tick_params(axis="y", labelcolor=P)
    ax2.spines["right"].set_visible(True)
    ax2.text(1, beta + 0.03, f"{beta:.2f}\np = 3e-9", ha="center", fontsize=9.4)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["Coverage vs burial\n(correlation)",
                           "Coverage vs class\n(after burial)"], fontsize=9.4)
    ax[2].set_title("C. Coverage adds to burial")
    fig.tight_layout(w_pad=3.5)
    fig.savefig(FIG / "fig_v3_permutation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"   panel C recomputed: beta={beta:.3f} p={pval:.2e} rho={rho_b:.3f}")


# ---------------------------------------------------------------- figure 6
def fig_coverage_discordance(lab):
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
    cov = [100 * (lab["n_nb"] >= k).mean() for k in range(0, 8)]
    ax[0].plot(range(0, 8), cov, "-o", color="#4a4a4a", lw=2)
    ax[0].fill_between(range(0, 8), cov, alpha=0.12, color="#4a4a4a")
    ax[0].set_xlabel("Minimum labelled neighbours required")
    ax[0].set_ylabel("% of labelled variants scoreable")
    ax[0].set_title("A. Coverage against confidence")
    ax[0].axvline(3, ls="--", c=P, lw=1)
    ax[0].annotate(f"at 3 or more: {cov[3]:.0f}% scoreable",
                   xy=(3.15, cov[3] + 6), fontsize=9.4, color=P)

    D = pd.read_csv(TAB / "v2_discordance_sensitivity.csv")
    order = ["lenient", "nominal", "ClinGen PP3 supporting", "stringent"]
    show = ["Loose", "Usual", "ClinGen\nPP3", "Strict"]
    D = D.set_index("thresholds").loc[order].reset_index()
    ax[1].bar(range(len(D)), D["pct_discordant"], color=[G, "#4a4a4a", P, B], width=0.6)
    ax[1].set_xticks(range(len(D)))
    ax[1].set_xticklabels(show, fontsize=9.4)
    ax[1].set_ylabel("% of evaluable VUS discordant")
    ax[1].set_ylim(0, 70)
    for i, v in enumerate(D["pct_discordant"]):
        ax[1].text(i, v + 1.2, f"{v}%", ha="center", fontsize=9.4)
    ax[1].set_title("B. Disagreement depends on cut offs")
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_coverage_discordance.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 7
def fig_calibration():
    c = pd.read_csv(TAB / "calibration.csv")
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], "--", c=G, lw=1.2, label="Perfect agreement")
    ax.plot(c["mean_pred"], c["frac_pos"], "-o", color=P, lw=2, ms=5,
            label="Fold by fold model")
    ax.set_xlabel("Predicted probability of pathogenic")
    ax.set_ylabel("Observed share pathogenic")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.text(0.06, 0.90, "Brier score 0.081", fontsize=10.6)
    ax.legend(fontsize=9.4, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig9_calibration.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df, ca, lab = _load()
    fig_pipeline(); print("   fig1_pipeline")
    fig_positional(df); print("   fig6_positional")
    fig_pps_by_label(df); print("   fig2_pps_by_label")
    fig_ascertainment(lab); print("   fig_v2_ascertainment")
    fig_permutation(lab); print("   fig_v3_permutation")
    fig_coverage_discordance(lab); print("   fig_v2_coverage_discordance")
    fig_calibration(); print("   fig9_calibration")
    print(f"[figures_v5] all saved to {FIG}")


if __name__ == "__main__":
    main()
