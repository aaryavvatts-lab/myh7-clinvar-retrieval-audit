"""v7 figures (PNG, results/figures/v7_*.png). All values are read from results/tables/*.csv; nothing is typed in.
Palette: colour-blind-safe (Okabe-Ito subset); marks thin; legends present; direct annotation of tie fractions.
Run: python -m src.figures_v7
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.genes import *   # noqa

C = {"S0": "#999999", "S1": "#0072B2", "count": "#E69F00", "AM": "#D55E00", "REVEL": "#CC79A7", "1D": "#009E73", "ink": "#222222"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "figure.dpi": 120,
                     "axes.edgecolor": "#555555", "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"]})
OUT = FIG


def fig_retrieval():
    d = pd.read_csv(TAB / "v7_retrieval_audit_gene.csv")
    p = d[d.in_panel_A].sort_values("loss_pct")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(p)); left = np.zeros(len(p))
    for col, lab, colr in (("n_symbol_field", "gene symbol stored differently", "#D55E00"), ("n_snapshot_missing", "absent from MyVariant ClinVar snapshot", "#0072B2"),
                           ("n_other", "other", "#999999"), ("n_no_hg19", "no hg19 coordinate", "#CCCCCC")):
        v = 100 * p[col].values / p.n_clinvar_missense.values
        ax[0].barh(y, v, left=left, color=colr, label=lab, height=0.7, edgecolor="white", linewidth=0.8); left += v
    ax[0].axvline(5, color=C["ink"], ls="--", lw=0.8); ax[0].text(5.3, -0.9, "5%", fontsize=8)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{g} (n={n})" for g, n in zip(p.gene, p.n_clinvar_missense)], fontsize=8)
    ax[0].set_xlabel("ClinVar missense SNVs not returned by clinvar.gene.symbol query (%)"); ax[0].legend(fontsize=7, loc="lower right", frameon=False)
    ax[0].set_title("A. Panel genes (19)", loc="left", fontsize=9)
    t = d[d.in_top100_A2 & (d.n_clinvar_missense >= 20)]
    ax[1].hist(t.loss_pct, bins=np.arange(0, 55, 2.5), color="#0072B2", edgecolor="white")
    ax[1].axvline(5, color=C["ink"], ls="--", lw=0.8)
    ax[1].set_xlabel("loss per gene (%)"); ax[1].set_ylabel("number of genes"); ax[1].set_title(f"B. Top-100 genes by ClinVar missense (n={len(t)})", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "v7_fig1_retrieval_audit.png"); plt.close(fig)


def fig_estimand():
    g = pd.read_csv(TAB / "v7_estimand_grid.csv")
    sets = [("T2", "T2: any ClinVar B/LB (primary)"), ("T1", "T1: curated (>=2 stars / VCEP)"), ("T3legacy_tight", "legacy v6 stand-ins (AF>=5e-5)"), ("v5_truncated", "truncated v5 table")]
    order = [("S0_scoreable", "S0 scoreable only"), ("S0_neutral", "S0 undefined = neutral"), ("S1", "S1 shrunk (primary)"), ("S2", "S2 class-balanced"),
             ("n_nb", "neighbour count"), ("S1_1d", "1D S1"), ("AlphaMissense", "AlphaMissense"), ("REVEL", "REVEL")]
    fig, axs = plt.subplots(1, len(sets), figsize=(12, 4.2), sharey=True)
    for ax, (ds, title) in zip(axs, sets):
        d = g[g.dataset == ds].set_index("score")
        for i, (s, lab) in enumerate(order):
            r = d.loc[s]
            col = C["S1"] if s == "S1" else (C["AM"] if s in ("AlphaMissense", "REVEL") else (C["count"] if s == "n_nb" else (C["1D"] if s == "S1_1d" else C["S0"])))
            ax.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=1.6); ax.plot(r.auroc, i, "o", color=col, ms=5)
            ax.text(1.02, i, f"{r.tie_fraction:.2f}", va="center", fontsize=7, color="#444444", transform=ax.get_yaxis_transform())
        ax.axvline(0.5, color="#888888", lw=0.8, ls=":"); ax.set_xlim(0.3, 1.0)
        ax.set_title(f"{title}\n(B = {int(d.n_B.max())})", fontsize=8.5, loc="left"); ax.set_xlabel("AUROC, 95% CI")
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([l for _, l in order]); axs[0].invert_yaxis()
    axs[-1].text(1.02, -0.9, "tie\nfraction", fontsize=7, transform=axs[-1].get_yaxis_transform())
    fig.tight_layout(); fig.savefig(OUT / "v7_fig2_estimand_grid.png"); plt.close(fig)


def fig_gene_forest():
    s = pd.read_csv(TAB / "v7_gene_grid.csv"); s = s[s.tier == "T2"]
    genes = pd.read_csv(TAB / "v7_gene_summary.csv"); genes = genes[genes.tier == "T2"].sort_values("n_B")
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(genes) + 1.5))
    off = {"S1": -0.22, "n_nb": 0.0, "AlphaMissense": 0.22}
    for i, gname in enumerate(genes.gene):
        for sc, lab in (("S1", "S1 (3D, shrunk)"), ("n_nb", "neighbour count"), ("AlphaMissense", "AlphaMissense")):
            r = s[(s.gene == gname) & (s.score == sc)].iloc[0]
            col = {"S1": C["S1"], "n_nb": C["count"], "AlphaMissense": C["AM"]}[sc]
            ax.plot([r.ci_low, r.ci_high], [i + off[sc]] * 2, color=col, lw=1.3); ax.plot(r.auroc, i + off[sc], "o", color=col, ms=4, label=lab if i == 0 else None)
    ax.axvline(0.5, color="#888888", lw=0.8, ls=":")
    ax.set_yticks(range(len(genes))); ax.set_yticklabels([f"{g}  (B={int(b)}, P={int(p)})" for g, b, p in zip(genes.gene, genes.n_B, genes.n_P)], fontsize=8)
    ax.set_xlabel("AUROC (ties = 0.5), P vs B, tier T2, 95% residue-cluster CI"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "v7_fig3_gene_forest.png"); plt.close(fig)


def fig_meta():
    p = pd.read_csv(TAB / "v7_gene_pairs.csv"); p = p[p.tier == "T2"]
    m = pd.read_csv(TAB / "v7_meta_pooled.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, (A, B, ttl) in zip(axs, [("S1", "o_alphamissense", "S1 minus AlphaMissense"), ("S1", "n_nb", "S1 minus neighbour count")]):
        d = p[(p.A == A) & (p.B == B)].sort_values("delta")
        y = np.arange(len(d)) + 1
        ax.errorbar(d.delta, y, xerr=[d.delta - d.ci_low, d.ci_high - d.delta], fmt="o", color=C["S1"], ms=4, lw=1.2)
        ax.set_yticks(y); ax.set_yticklabels(d.gene, fontsize=8)
        r = m[(m.set == "amended_pool") & (m.A == A) & (m.B == B)]
        if len(r):
            r = r.iloc[0]
            ax.plot([r.ci_low_hk, r.ci_high_hk], [0, 0], color=C["ink"], lw=2.2); ax.plot(r.pooled, 0, "D", color=C["ink"], ms=7)
            ax.plot([r.pred_low, r.pred_high], [-0.7, -0.7], color="#777777", lw=1.0, ls="--")
            ax.text(0.02, 0.02, f"pooled {r.pooled:+.3f} (HK 95% CI {r.ci_low_hk:+.3f} to {r.ci_high_hk:+.3f})\nI2 = {r.I2_pct:.0f}%, tau = {r.tau:.3f}; dashed = 95% prediction interval",
                    transform=ax.transAxes, fontsize=7)
        ax.axvline(0, color="#888888", lw=0.8, ls=":"); ax.set_xlabel("difference in AUROC (same variants), residue-cluster 95% CI"); ax.set_title(ttl, loc="left", fontsize=9)
        ax.set_ylim(-1.2, len(d) + 0.8)
    fig.tight_layout(); fig.savefig(OUT / "v7_fig4_meta_forest.png"); plt.close(fig)


def fig_scarcity():
    s = pd.read_csv(TAB / "v7_gene_summary.csv"); s = s[s.tier == "T2"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (yc, ttl) in zip(axs, [("auroc_S1", "AUROC of S1"), ("delta_S1_minus_n_nb", "S1 minus count (AUROC)"), ("S0_tie_fraction", "S0 tie fraction")]):
        ax.scatter(s.n_B, s[yc], color=C["S1"], s=22)
        for _, r in s.iterrows():
            ax.annotate(r.gene, (r.n_B, r[yc]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("benign protein changes (T2)"); ax.set_title(ttl, loc="left", fontsize=9)
        if yc.startswith("delta"):
            ax.axhline(0, color="#888888", lw=0.8, ls=":")
    fig.tight_layout(); fig.savefig(OUT / "v7_fig5_scarcity.png"); plt.close(fig)


def fig_power():
    d = pd.read_csv(TAB / "v7_power_sim_myh7.csv"); d = d[(d.tier == "T2") & d.power.notna()]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for sc, col, lab in (("S0_scoreable", C["S0"], "S0, scoreable only"), ("S1", C["S1"], "S1, all variants")):
        x = d[d.score == sc]; ax.plot(x.true_auroc, x.power, "o-", color=col, ms=4, lw=1.4, label=f"{lab} (B = {int(x.n_B.iloc[0])})")
    ax.axhline(0.8, color="#888888", lw=0.8, ls=":"); ax.set_xlabel("true AUROC of the simulated score"); ax.set_ylabel("power (one-sided 5% permutation test)")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Power under the observed tie structure (MYH7, T2)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "v7_fig6_power.png"); plt.close(fig)


def fig_cv():
    c = pd.read_csv(TAB / "v7_cv_nested_contrasts.csv"); c = c[c.grouping == "residue"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    qs = list(dict.fromkeys(c.question)); tiers = ["T1", "T2", "T3"]
    cols = {"T1": "#009E73", "T2": C["S1"], "T3": "#CC79A7"}
    for j, q in enumerate(qs):
        for k, t in enumerate(tiers):
            r = c[(c.question == q) & (c.tier == t)].iloc[0]; yy = j + (k - 1) * 0.22
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color=cols[t], lw=1.4); ax.plot(r.delta_mean, yy, "o", color=cols[t], ms=4, label=t if j == 0 else None)
    ax.axvspan(-0.02, 0.02, color="#DDDDDD", alpha=0.6, lw=0); ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(range(len(qs))); ax.set_yticklabels(qs, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("AUROC difference, 20x residue-grouped 5-fold CV (95% CI); grey = +-0.02"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "v7_fig7_nested_cv.png"); plt.close(fig)


def main():
    for f in (fig_retrieval, fig_estimand, fig_gene_forest, fig_meta, fig_scarcity, fig_power, fig_cv):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:  # noqa
            print("FAILED", f.__name__, repr(e)); raise


if __name__ == "__main__":
    main()
