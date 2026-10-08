"""v2 figures for the corrected manuscript: ascertainment diagnosis and its consequences.
Run: python -m src.figures_v2
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.structure import load_ca
from src.reanalysis import neighbour_support, relabel

FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9.5,
                     "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
P, B, A, G, S = "#b3243c", "#20607a", "#4a4a4a", "#9aa0a6", "#c98a1a"


def main():
    df0 = pd.read_parquet("data/processed/features.parquet"); ca = load_ca()
    d = relabel(df0, 1e-5)
    lab = d[d.label.isin(["P", "B"])].reset_index(drop=True)
    npath, nben = neighbour_support(lab, ca)
    lab["n_nb"] = npath + nben
    y = lab.label.eq("P").astype(int).values

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))

    # (A) the ascertainment gap
    bins = [0, 1, 3, 6, 11, 10**9]
    names = ["0", "1–2", "3–5", "6–10", ">10"]
    cat = pd.cut(lab["n_nb"], bins=[-.1, .5, 2.5, 5.5, 10.5, 1e9], labels=names)
    pc = [100 * ((cat == n) & (y == 1)).sum() / max((y == 1).sum(), 1) for n in names]
    bc = [100 * ((cat == n) & (y == 0)).sum() / max((y == 0).sum(), 1) for n in names]
    x = np.arange(len(names)); w = 0.38
    ax[0].bar(x - w/2, pc, w, color=P, label=f"Pathogenic (n={int((y==1).sum())})")
    ax[0].bar(x + w/2, bc, w, color=B, label=f"Benign (n={int((y==0).sum())})")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names)
    ax[0].set_xlabel("Labelled non-local 3D neighbours"); ax[0].set_ylabel("% of class")
    ax[0].set_title("A. Ascertainment gap in structural support", fontsize=10, weight="bold")
    ax[0].legend(fontsize=7.5, frameon=False)
    ax[0].annotate("47% vs 6% have ≥3\n→ class differs in\nsupport, not only biology",
                   xy=(2.6, 40), fontsize=7.5, color=G)

    # (B) within-stratum AUROC: the decisive control
    strata, pps_a, am_a, ns = [], [], [], []
    for n in names:
        m = (cat == n).values
        if m.sum() < 25 or len(np.unique(y[m])) < 2:
            continue
        v = pd.to_numeric(lab["pps3d_norm"], errors="coerce").values
        va = pd.to_numeric(lab["alphamissense"], errors="coerce").values
        m1 = m & ~np.isnan(v); m2 = m & ~np.isnan(va)
        strata.append(n); ns.append(m.sum())
        pps_a.append(roc_auc_score(y[m1], v[m1]))
        am_a.append(roc_auc_score(y[m2], va[m2]))
    x2 = np.arange(len(strata))
    ax[1].plot(x2, pps_a, "-o", color=P, lw=2, label="3D-PPS")
    ax[1].plot(x2, am_a, "-s", color=B, lw=2, label="AlphaMissense")
    ax[1].axhline(0.5, ls="--", c=G, lw=1)
    ax[1].text(0.04, 0.515, "chance", fontsize=7, color=G)
    ax[1].set_xticks(x2); ax[1].set_xticklabels([f"{s}\n(n={n})" for s, n in zip(strata, ns)],
                                                fontsize=8)
    ax[1].set_ylim(0.4, 1.03); ax[1].set_ylabel("AUROC within stratum")
    ax[1].set_xlabel("Labelled non-local 3D neighbours")
    ax[1].set_title("B. Matched on support, the advantage vanishes", fontsize=10, weight="bold")
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")
    ax[1].annotate("undefined →\nexactly chance", xy=(0.02, 0.55), fontsize=7.5, color=P)

    # (C) the discovery claim collapses with a support requirement
    counts = [22, 3]
    ax[2].bar(["Naïve\n(≥1 neighbour)", "Support-controlled\n(≥3 neighbours)"], counts,
              color=[S, P], width=0.55)
    for i, v in enumerate(counts):
        ax[2].text(i, v + 0.6, str(v), ha="center", fontweight="bold")
    ax[2].set_ylabel("‘Structural orphan’ candidates")
    ax[2].set_title("C. Discovery claims are support-dependent", fontsize=10, weight="bold")
    ax[2].annotate("all 3 survivors lie in\na single site (res 120–123)",
                   xy=(0.55, 12), fontsize=7.5, color=G)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_ascertainment.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_v2_ascertainment.png")

    # second figure: coverage / abstention + discordance sensitivity
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    cov = [100 * (lab["n_nb"] >= k).mean() for k in range(0, 8)]
    ax[0].plot(range(0, 8), cov, "-o", color=A, lw=2)
    ax[0].fill_between(range(0, 8), cov, alpha=0.12, color=A)
    ax[0].set_xlabel("Minimum labelled neighbours required")
    ax[0].set_ylabel("% of variants scoreable")
    ax[0].set_title("A. Coverage–confidence trade-off", fontsize=10, weight="bold")
    ax[0].axvline(3, ls="--", c=P, lw=1)
    ax[0].annotate(f"at ≥3: {cov[3]:.0f}% scoreable", xy=(3.15, cov[3] + 6), fontsize=8, color=P)

    D = pd.read_csv("results/tables/v2_discordance_sensitivity.csv")
    ax[1].bar(range(len(D)), D["pct_discordant"], color=[G, A, P, B], width=0.6)
    ax[1].set_xticks(range(len(D)))
    ax[1].set_xticklabels([t.replace(" ", "\n") for t in D["thresholds"]], fontsize=7.5)
    ax[1].set_ylabel("% of evaluable VUS discordant")
    ax[1].set_ylim(0, 70)
    for i, v in enumerate(D["pct_discordant"]):
        ax[1].text(i, v + 1.2, f"{v}%", ha="center", fontsize=8)
    ax[1].set_title("B. Discordance depends on thresholds", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_coverage_discordance.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_v2_coverage_discordance.png")


if __name__ == "__main__":
    main()
