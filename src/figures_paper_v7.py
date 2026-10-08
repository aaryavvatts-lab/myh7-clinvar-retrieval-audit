"""Figures for the v7 manuscript and its supplement (paper/figures/*.png). Every plotted value is read from
results/tables/*.csv. No figure numbers or titles inside the images (captions carry them); panel letters only.
Okabe-Ito palette (colour-blind safe), fonts >= 7.5 pt at the final 6.5 inch text width.
Run from v7/:  ../.venv/bin/python -m src.figures_paper_v7
"""
from __future__ import annotations
import re
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAB = Path("results/tables"); OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7.6, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 150})
BLU, ORA, VER, GRN, PUR, SKY, GRY, BLK = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#7F7F7F", "#222222"
T = lambda n: pd.read_csv(TAB / n)


def letter(ax, s, x=-0.13, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold")


def parse_ci(s):
    a, b = re.findall(r"-?\d+\.?\d*", str(s))[:2]
    return float(a), float(b)


# ------------------------------------------------------------------ Fig 1 retrieval audit
def fig1():
    d = T("v7_retrieval_audit_gene.csv")
    p = d[d.in_panel_A].sort_values("loss_pct")
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 4.3), gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(p)); left = np.zeros(len(p))
    # the third class (no GRCh37 coordinate, 10 variants in all) is too small to draw and is stated in the caption
    for col, lab, colr in (("n_symbol_field", "Different gene symbol stored", ORA), ("n_snapshot_missing", "Absent from the ClinVar snapshot", BLU)):
        v = 100 * p[col].values / p.n_clinvar_missense.values
        ax[0].barh(y, v, left=left, color=colr, label=lab, height=0.72, edgecolor="white", linewidth=0.6); left += v
    ax[0].axvline(5, color=BLK, ls="--", lw=0.8)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{g} ({n:,})" for g, n in zip(p.gene, p.n_clinvar_missense)], fontsize=7.4)
    ax[0].set_xlabel("ClinVar missense SNVs not returned (%)"); ax[0].legend(frameon=False, loc="lower right", fontsize=7.2)
    letter(ax[0], "A", x=-0.32)
    t = d[d.in_top100_A2]
    bins = np.arange(0, 105, 5)
    ax[1].hist(t.loss_pct, bins=bins, color=BLU, edgecolor="white")
    ax[1].axvline(5, color=BLK, ls="--", lw=0.8)
    n5 = int((t.loss_pct > 5).sum())
    ax[1].text(0.97, 0.93, f"{n5} of {len(t)} genes\nlose more than 5%", transform=ax[1].transAxes, ha="right", va="top", fontsize=8)
    ax[1].set_xlabel("Loss per gene (%)"); ax[1].set_ylabel("Number of genes")
    letter(ax[1], "B", x=-0.2)
    fig.tight_layout(); fig.savefig(OUT / "f1_retrieval_audit.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 2 MYH7 estimand grid
def fig2():
    g = T("v7_estimand_grid.csv")
    sets = [("T2", "Any ClinVar benign"), ("T1", "Curated benign"), ("T3legacy_tight", "Global-frequency stand-ins")]
    rows = [("S0_scoreable", "S0, scoreable only", GRY), ("S0_neutral", "S0, undefined as neutral", GRY), ("S1", "S1 (shrunk)", BLU),
            ("S2", "S2 (class-balanced)", SKY), ("n_nb", "Neighbour count", ORA), ("S1_1d", "1D S1", GRN),
            ("AlphaMissense", "AlphaMissense", VER), ("REVEL", "REVEL", PUR)]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 3.5), sharey=True)
    for ax, (ds, nm) in zip(axs, sets):
        for i, (sc, lab, col) in enumerate(rows):
            r = g[(g.dataset == ds) & (g.score == sc)].iloc[0]
            ax.errorbar(r.auroc, i, xerr=[[r.auroc - r.ci_low], [r.ci_high - r.auroc]], fmt="o", color=col, ms=4.5, capsize=2, lw=1.2)
            ax.text(1.03, i, f"{r.tie_fraction:.2f}", fontsize=7, va="center", ha="left", transform=ax.get_yaxis_transform())
        nb = int(g[(g.dataset == ds) & (g.score == "S1")].n_B.iloc[0])
        ax.axvline(0.5, color=GRY, ls="--", lw=0.8)
        ax.set_xlim(0.3, 1.0); ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("AUROC"); ax.text(0.62, 0.0, f"{nb} benign", fontsize=7.6, color=BLK, va="center")
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([r[1] for r in rows]); axs[0].invert_yaxis()
    for ax in axs:
        ax.text(1.03, -0.75, "Ties", fontsize=7, ha="left", va="bottom", transform=ax.get_yaxis_transform())
    for k, ax in enumerate(axs):
        ax.text(0.015, 0.97, "ABC"[k], transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    fig.tight_layout(); fig.subplots_adjust(wspace=0.22); fig.savefig(OUT / "f2_myh7_estimands.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 3 per-gene forest
def fig3():
    s = T("v7_gene_summary.csv"); s = s[s.tier == "T2"].copy()
    s = s.sort_values("n_B")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    spec = [("auroc_S1", "ci_S1", "S1 (3D, shrunk)", BLU, "o", -0.24), ("auroc_n_nb", "ci_n_nb", "Neighbour count", ORA, "s", 0.0),
            ("auroc_AlphaMissense", "ci_AlphaMissense", "AlphaMissense", VER, "D", 0.24)]
    for k, (a, c, lab, col, mk, off) in enumerate(spec):
        for i, (_, r) in enumerate(s.iterrows()):
            lo, hi = parse_ci(r[c])
            ax.errorbar(r[a], i + off, xerr=[[r[a] - lo], [hi - r[a]]], fmt=mk, color=col, ms=4, capsize=1.8, lw=1.1, label=lab if i == 0 else None)
    ax.axvline(0.5, color=GRY, ls="--", lw=0.8)
    ax.set_yticks(range(len(s))); ax.set_yticklabels([f"{r.gene}  ({int(r.n_P)} P, {int(r.n_B)} B)" for _, r in s.iterrows()])
    ax.set_xlim(0.55, 1.0); ax.set_xlabel("AUROC (ties 0.5), 95% residue-cluster bootstrap interval")
    ax.legend(frameon=False, loc="upper left", fontsize=7.8)
    fig.tight_layout(); fig.savefig(OUT / "f3_gene_forest.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 4 meta-analysis
def fig4():
    p = T("v7_gene_pairs.csv"); p = p[(p.tier == "T2") & (p.A == "S1")]
    m = T("v7_meta_pooled.csv"); m = m[m.set == "amended_pool"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.8))
    for ax, (b, nm, lett) in zip(axs, (("o_alphamissense", "AlphaMissense", "A"), ("n_nb", "neighbour count", "B"))):
        d = p[p.B == b].sort_values("delta", ascending=False).reset_index(drop=True)
        for i, r in d.iterrows():
            ax.errorbar(r.delta, i, xerr=[[r.delta - r.ci_low], [r.ci_high - r.delta]], fmt="o", color=BLU, ms=4, capsize=2, lw=1.1)
        mp = m[(m.A == "S1") & (m.B == b)].iloc[0]
        yy = len(d) + 0.8
        ax.errorbar(mp.pooled, yy, xerr=[[mp.pooled - mp.ci_low_hk], [mp.ci_high_hk - mp.pooled]], fmt="D", color=BLK, ms=6, capsize=3, lw=2)
        ax.plot([mp.pred_low, mp.pred_high], [yy + 0.8, yy + 0.8], color=GRY, lw=1.4, ls="--")
        ax.axvline(0, color=GRY, ls=":", lw=0.9)
        ax.set_yticks(list(range(len(d))) + [yy]); ax.set_yticklabels(list(d.gene) + ["Pooled"])
        ax.invert_yaxis()
        ax.set_xlabel(f"S1 minus {nm}: AUROC difference")
        ax.set_ylim(yy + 3.4, -0.7)
        ax.text(0.02, yy + 2.5, f"pooled {mp.pooled:+.3f} (HK 95% CI {mp.ci_low_hk:+.3f} to {mp.ci_high_hk:+.3f})\nI2 {mp.I2_pct:.0f}%; dashed line = 95% prediction interval",
                transform=ax.get_yaxis_transform(), fontsize=6.9, va="center")
        letter(ax, lett, x=-0.22)
    fig.tight_layout(); fig.savefig(OUT / "f4_meta_forest.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 5 nested contrasts, MYH7 tiers and genes
def fig5():
    nc = T("v7_cv_nested_contrasts.csv"); nc = nc[nc.grouping == "residue"]
    gc = T("v7_gene_cv_contrasts.csv"); gc = gc[(gc.tier == "T2") & (gc.question == "S1 beyond count (primary nested)")].copy()
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    qs = ["count over panel", "S1 over panel", "S1 beyond count (primary nested)", "count beyond S1"]
    lab = ["Count over panel", "S1 over panel", "S1 beyond count", "Count beyond S1"]
    tiers = [("T1", "Curated benign (12)", GRN, -0.22), ("T2", "Any ClinVar benign (20)", BLU, 0.0), ("T3", "With population stand-ins (22)", PUR, 0.22)]
    ax.axvspan(-0.02, 0.02, color="#e8e8e8", lw=0)
    for tt, nm, col, off in tiers:
        for i, q in enumerate(qs):
            r = nc[(nc.tier == tt) & (nc.question == q)].iloc[0]
            ax.errorbar(r.delta_mean, i + off, xerr=[[r.delta_mean - r.ci_low], [r.ci_high - r.delta_mean]], fmt="o", color=col, ms=4, capsize=2, lw=1.1, label=nm if i == 0 else None)
    ax.axvline(0, color=GRY, lw=0.8)
    ax.set_yticks(range(len(qs))); ax.set_yticklabels(lab); ax.invert_yaxis()
    ax.set_xlabel("MYH7: AUROC difference, nested models"); h, l = ax.get_legend_handles_labels()
    letter(ax, "A", x=-0.5)
    ax = axs[1]
    gc = gc.sort_values("delta_mean")
    ax.axvspan(-0.02, 0.02, color="#e8e8e8", lw=0)
    for i, (_, r) in enumerate(gc.iterrows()):
        ax.errorbar(r.delta_mean, i, xerr=[[r.delta_mean - r.ci_low], [r.ci_high - r.delta_mean]], fmt="o", color=BLU, ms=4, capsize=2, lw=1.1)
    ax.axvline(0, color=GRY, lw=0.8)
    ax.set_yticks(range(len(gc))); ax.set_yticklabels(list(gc.gene))
    ax.set_xlabel("S1 beyond count: AUROC difference"); letter(ax, "B", x=-0.28)
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "f5_nested.png"); plt.close(fig)


# ------------------------------------------------------------------ Supplement figures
def place_labels(fig, ax, xs, ys, names, fs=6.4):
    """Greedy label placement: for each point try 8 offsets and keep the one whose box overlaps the fewest
    points and already placed labels (display coordinates, after the axes limits are fixed)."""
    ax.set_xscale("log")
    xr = (min(xs), max(xs)); yr = (min(ys), max(ys))
    ax.set_xlim(xr[0] / 1.5, xr[1] * 1.9); ax.set_ylim(yr[0] - 0.12 * (yr[1] - yr[0]) - 1e-9, yr[1] + 0.12 * (yr[1] - yr[0]) + 1e-9)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.c_[xs, ys])
    boxes = []
    cands = [(5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"), (-5, -3, "right", "top"),
             (0, 7, "center", "bottom"), (0, -7, "center", "top"), (8, 0, "left", "center"), (-8, 0, "right", "center")]
    order = np.argsort(ys)
    for i in order:
        best = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(names[i], (xs[i], ys[i]), fontsize=fs, xytext=(dx * 1.6, dy * 1.6), textcoords="offset points", ha=ha, va=va)
            bb = t.get_window_extent(rend).expanded(1.05, 1.15)
            ov = sum(bb.contains(*p) for j, p in enumerate(pts) if j != i) * 3 + sum(bb.overlaps(b) for b in boxes) * 2
            axbb = ax.get_window_extent(rend)
            ov += 5 * (bb.x0 < axbb.x0 or bb.x1 > axbb.x1 or bb.y0 < axbb.y0 or bb.y1 > axbb.y1)
            t.remove()
            if best is None or ov < best[0]:
                best = (ov, dx, dy, ha, va, bb)
        _, dx, dy, ha, va, bb = best
        ax.annotate(names[i], (xs[i], ys[i]), fontsize=fs, xytext=(dx * 1.6, dy * 1.6), textcoords="offset points", ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=GRY, shrinkA=0, shrinkB=2),
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.9), zorder=5)
        boxes.append(bb)


def figS1():
    s = T("v7_gene_summary.csv"); s = s[s.tier == "T2"]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 3.2))
    for ax, (col, lab, let) in zip(axs, (("auroc_S1", "AUROC of S1", "A"), ("delta_S1_minus_n_nb", "S1 minus count", "B"), ("S0_tie_fraction", "S0 tie fraction", "C"))):
        ax.scatter(s.n_B, s[col], color=BLU, s=16)
        place_labels(fig, ax, s.n_B.values, s[col].values, list(s.gene))
        ax.set_xscale("log"); ax.set_xlabel("Benign protein changes"); ax.set_ylabel(lab)
        if col == "delta_S1_minus_n_nb":
            ax.axhline(0, color=GRY, ls=":", lw=0.9)
        letter(ax, let, x=-0.28)
    fig.tight_layout(); fig.savefig(OUT / "s1_scarcity.png"); plt.close(fig)


def figS2():
    d = T("v7_power_sim_myh7.csv"); d = d[d.tier == "T2"]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for sc, col, lab in (("S0_scoreable", GRY, "S0, scoreable only (7 benign)"), ("S1", BLU, "S1, all variants (20 benign)")):
        x = d[d.score == sc]
        ax.plot(x.true_auroc, x.power, "o-", color=col, ms=3.5, lw=1.3, label=lab)
    ax.axhline(0.8, color=GRY, ls=":", lw=0.9)
    ax.set_xlabel("True AUROC of the simulated score"); ax.set_ylabel("Power (one-sided 5% permutation test)")
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "s2_power.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, figS1, figS2):
        f()
    print(sorted(p.name for p in OUT.glob("*.png")))
