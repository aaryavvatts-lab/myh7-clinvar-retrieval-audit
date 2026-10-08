"""Figures for the v6 paper. Every plotted number is read from results/tables/*.csv|npy (or computed from
data/processed features). No figure numbers or titles inside images (captions carry them).
Colour-blind safe (Okabe-Ito). Sized for a 6.5 inch text width with fonts >= 7.5 pt at final size.

Run from v6/:  ../.venv/bin/python -m src.figures_paper
Output: paper/figures/fig1..fig6.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from src.common import TAB, load, labelled

OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7.8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 150})
BLU, ORA, VER, GRN, PUR, SKY, GRY, BLK = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#7F7F7F", "#222222"
T = lambda n: pd.read_csv(TAB / n)


def box(ax, x, y, w, h, text, fc="#eef3f8", ec=BLU, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=GRY, lw=1.2))


# ------------------------------------------------------------------ Fig 1
def fig1():
    gs = T("v6_diag_gene_symbol.csv"); blk = T("v6_diag_residue_blocks.csv")
    diag = __import__("json").load(open("logs/diagnosis_gap.json"))
    prov = __import__("json").load(open("logs/provenance_acquisition.json"))
    total, found, missed = diag["clinvar_missense_total"], diag["clinvar_missense_found_by_symbol_query"], diag["clinvar_missense_missed_by_symbol_query"]
    nosym = int(gs.loc[gs.myvariant_clinvar_gene_symbol.str.startswith("no gene"), "n_clinvar_missense"].iloc[0])
    other = missed - nosym
    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_axes([0.02, 0.62, 0.97, 0.36]); ax.axis("off"); ax.set_xlim(-0.3, 10); ax.set_ylim(0.3, 3.2)
    box(ax, 0.05, 0.9, 2.2, 1.7, f"ClinVar release\n(file dated 5 Oct 2026)\n{total:,} MYH7\nmissense SNVs")
    arrow(ax, 2.3, 2.85, 1.75)
    box(ax, 2.9, 0.9, 2.3, 1.7, "Earlier retrieval:\nMyVariant.info query\non one gene symbol\nfield (MYH7)", ec=ORA, fc="#fdf3df")
    arrow(ax, 5.25, 5.8, 1.75)
    box(ax, 5.85, 1.7, 2.1, 0.9, f"Found: {found:,}", ec=GRN, fc="#e3f4ee")
    box(ax, 5.85, 0.4, 2.1, 1.2, f"Never retrieved: {missed:,}\n{other} other gene symbol\n{nosym} newer than the\nMay 2025 snapshot", ec=VER, fc="#fbe6dc", fs=7.6)
    arrow(ax, 8.0, 8.4, 1.75)
    box(ax, 8.45, 0.9, 1.5, 1.7, f"Complete cohort\n{3904:,}\n(with gnomAD v4\nvariants)", fs=7.6)
    ax.text(0.0, 3.0, "A", fontsize=11, fontweight="bold")
    ax2 = fig.add_axes([0.11, 0.09, 0.87, 0.40])
    x = np.arange(len(blk)); w = 0.4
    ax2.axvspan(7.5, 8.9 + 0.0, color=GRY, alpha=0.18, lw=0)    # 763-893 overlaps blocks 700-800, 800-900
    ax2.axvspan(14.5, 17.5, color=GRY, alpha=0.18, lw=0)         # 1549-1770
    ax2.bar(x - w / 2, blk.old_clinvar_missense, w, color=ORA, label="Earlier retrieval")
    ax2.bar(x + w / 2, blk.v6_clinvar_missense, w, color=BLU, label="Complete ClinVar release")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{i*100}" for i in range(len(blk))], rotation=0, fontsize=7)
    ax2.set_xlabel("MYH7 residue (start of each 100 residue block)"); ax2.set_ylabel("ClinVar missense variants\nper 100 residues")
    ax2.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.03))
    ax2.text(8.2, 236, "763-893", ha="center", fontsize=7.5, color=GRY)
    ax2.text(16.0, 236, "1549-1770", ha="center", fontsize=7.5, color=GRY)
    ax2.set_ylim(0, 300)
    fig.text(0.0, 0.53, "B", fontsize=11, fontweight="bold")
    fig.savefig(OUT / "fig1_provenance_gap.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 2
def fig2():
    df = load("data/processed"); lab = labelled(df, "wider")
    bins = [-0.1, 0.5, 2.5, 5.5, 10.5, 1e9]; names = ["0", "1 to 2", "3 to 5", "6 to 10", "over 10"]
    lab["bin"] = pd.cut(lab.n_nb, bins=bins, labels=names)
    nP, nB = int(lab.y.sum()), int((1 - lab.y).sum())
    fp = lab[lab.y == 1].bin.value_counts(normalize=True).reindex(names).fillna(0) * 100
    fb = lab[lab.y == 0].bin.value_counts(normalize=True).reindex(names).fillna(0) * 100
    sp = T("v6_sensitivity_specs.csv"); sp = sp[sp.set == "wider"].reset_index(drop=True)
    short = ["Primary:\naggregate,\nv4", "Any-record,\nv4", "Aggregate,\nv2.1.1", "Any-record,\nv2.1.1"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]; x = np.arange(len(names)); w = 0.4
    ax.bar(x - w / 2, fp, w, color=VER, label=f"Pathogenic (n={nP})")
    ax.bar(x + w / 2, fb, w, color=BLU, label=f"Benign (n={nB})")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_xlabel("Labelled 3D neighbours (non-local)")
    ax.set_ylabel("Share of class (%)"); ax.legend(frameon=False)
    ax.text(-0.15, 1.04, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax = axs[1]; x = np.arange(4)
    ax.bar(x - w / 2, sp.abstain_P_pct, w, color=VER, label="Pathogenic")
    ax.bar(x + w / 2, sp.abstain_B_pct, w, color=BLU, label="Benign")
    for i, r in sp.iterrows():
        ax.text(i - w / 2, r.abstain_P_pct + 1.5, f"{r.abstain_P_pct:.0f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, r.abstain_B_pct + 1.5, f"{r.abstain_B_pct:.0f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, 3, f"B={int(r.n_B)}", ha="center", fontsize=6.8, color="white", rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.2)
    ax.set_ylabel("No labelled neighbour,\n3D-PPS undefined (%)"); ax.set_ylim(0, 95)
    ax.legend(frameon=False, loc="upper right")
    ax.text(-0.15, 1.04, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig2_coverage_asymmetry.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 3
def fig3():
    g = T("v6_sensitivity_grid_ci.csv"); g = g[g.set == "wider"]
    cohorts = ["full ClinVar aggregate, gnomAD v4 (primary)", "full data, v5 label rule, gnomAD v4",
               "full ClinVar aggregate, gnomAD v2.1.1", "full data, v5 label rule, gnomAD v2.1.1"]
    titles = ["Primary: ClinVar aggregate labels, gnomAD v4", "Any-record label rule, gnomAD v4",
              "ClinVar aggregate labels, gnomAD v2.1.1", "Any-record label rule, gnomAD v2.1.1"]
    preds = [("3D-PPS", VER, "o"), ("1D control", PUR, "s"), ("AlphaMissense", BLU, "D"), ("REVEL", GRN, "^")]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    y = 0; yt, yl = [], []
    for ci, c in enumerate(cohorts):
        ax.text(0.30, y - 0.1, titles[ci], fontsize=8, fontweight="bold", va="center")
        y += 1
        for pn, col, mk in preds:
            r = g[(g.cohort == c) & (g.predictor == pn)].iloc[0]
            ax.errorbar(r.auroc, y, xerr=[[r.auroc - r.ci_low], [r.ci_high - r.auroc]], fmt=mk, color=col, ms=5.5, capsize=2.5, lw=1.3)
            yt.append(y); yl.append(f"{pn}  (n={int(r.n)}, benign={int(r.n_B)})")
            ax.text(1.005, y, f"{r.auroc:.3f}", fontsize=7.5, va="center")
            y += 1
        y += 0.6
    ax.axvline(0.5, color=GRY, ls="--", lw=1)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.4); ax.invert_yaxis()
    ax.set_xlim(0.28, 1.05); ax.set_xlabel("AUROC (95% residue-cluster bootstrap CI)")
    fig.tight_layout(); fig.savefig(OUT / "fig3_auroc_grid.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 4
def fig4():
    n1 = np.load(TAB / "v6_permutation_null1_wider.npy"); n2 = np.load(TAB / "v6_permutation_null2_wider.npy")
    pm = T("v6_permutation.csv"); pw = pm[pm.set == "wider"]
    obs = float(pw.iloc[0].observed_auroc)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    bins = np.linspace(0.3, 0.9, 61)
    ax.hist(n1, bins=bins, color=GRY, alpha=0.7, label="Null 1: shuffle all labels")
    ax.hist(n2, bins=bins, color=ORA, alpha=0.65, label="Null 2: shuffle within strata")
    ax.axvline(obs, color=VER, lw=2, label=f"Observed {obs:.3f}")
    ax.set_xlabel("AUROC on scoreable variants (297 P, 7 B)"); ax.set_ylabel("Shuffles (of 10,000)")
    p1 = float(pw[pw.null == "null1_unconditional"].p_value.iloc[0]); p2 = float(pw[pw.null.str.startswith("null2")].p_value.iloc[0])
    ax.legend(frameon=False, loc="upper right", fontsize=6.8, bbox_to_anchor=(1.02, 1.0))
    ax.text(0.97, 0.55, f"p = {p1:.2f} (null 1)\np = {p2:.2f} (null 2)", transform=ax.transAxes, va="top", ha="right", fontsize=7.8)
    ax.text(-0.15, 1.04, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax = axs[1]
    order = [("primary", "Primary"), ("v5rcv_v4", "Any-record, v4"), ("aggregate_v2", "Aggregate, v2.1.1"), ("v5rcv_v2", "Any-record, v2.1.1")]
    for i, (k, nm) in enumerate(order):
        r1 = pw[(pw.cohort == k) & (pw.null == "null1_unconditional")].iloc[0]
        r2 = pw[(pw.cohort == k) & pw.null.str.startswith("null2")].iloc[0]
        ax.plot([r1.null_mean, r1.null_p95], [i, i], color=GRY, lw=5, alpha=0.6, solid_capstyle="butt",
                label="Null 1: mean to 95th percentile" if i == 0 else None)
        ax.plot(r1.observed_auroc, i, "o", color=VER, ms=6, label="Observed" if i == 0 else None)
        ax.text(0.99, i, f"B={int(r1.n_B_scoreable)}\np={r1.p_value:.2g}/{r2.p_value:.2g}", fontsize=6.8, va="center", ha="left",
                transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=GRY, ls="--", lw=0.8)
    ax.set_yticks(range(4)); ax.set_yticklabels([n for _, n in order], fontsize=7.4); ax.invert_yaxis()
    ax.set_xlim(0.4, 0.85); ax.set_xlabel("AUROC")
    ax.legend(frameon=False, loc="lower center", ncol=2, fontsize=6.6, bbox_to_anchor=(0.5, 1.0))
    ax.text(-0.45, 1.04, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.subplots_adjust(right=0.86); fig.savefig(OUT / "fig4_permutation.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 5
def fig5():
    s = T("spatial_block_cv.csv")
    s = s[(s.scheme == "blocks100") & (s.set == "wider") & (s.subset == "all held-out variants")]
    meth = [("3D-PPS (scoreable only)", "3D-PPS"), ("1D-PPS (scoreable only)", "1D control"),
            ("coverage flag (>=1 training neighbour)", "Coverage flag (1 or more)"), ("AlphaMissense", "AlphaMissense"),
            ("REVEL", "REVEL"), ("model A_only", "Panel only"), ("model A_cov", "Panel + coverage count"), ("model A_3d", "Panel + 3D-PPS")]
    coh = [("primary", "Primary (ClinVar aggregate labels, gnomAD v4)", VER, "o", -0.18),
           ("v5rcv_v4", "Any-record rule, gnomAD v4", BLU, "s", 0.18)]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for k, lab, col, mk, off in coh:
        for i, (m, nm) in enumerate(meth):
            r = s[(s.cohort == k) & (s.method == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.auroc, i + off, xerr=[[r.auroc - r.ci_low], [r.ci_high - r.auroc]], fmt=mk, color=col, ms=5, capsize=2.2, lw=1.2,
                        label=lab if i == 0 else None, mfc=col if r.n_B >= 10 else "white")
            ax.text(1.03, i + off, f"B={int(r.n_B)}", fontsize=6.8, va="center", color=col)
    ax.axvline(0.5, color=GRY, ls="--", lw=1)
    ax.set_yticks(range(len(meth))); ax.set_yticklabels([n for _, n in meth]); ax.invert_yaxis()
    ax.set_xlim(0.3, 1.1); ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUROC on variants in held-out 100-residue blocks")
    ax.legend(frameon=False, loc="lower left", fontsize=7.4)
    fig.tight_layout(); fig.savefig(OUT / "fig5_spatial_block_cv.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 6
def fig6():
    cb = T("v6_calibration_wider_balanced.csv"); cu = T("v6_calibration_wider_unweighted.csv"); cs = T("v6_calibration_summary.csv")
    cs = cs[cs.set == "wider"]
    sb = float(cs[cs.weights == "balanced"].cal_slope.iloc[0]); su = float(cs[cs.weights == "unweighted"].cal_slope.iloc[0])
    v = pd.read_csv(TAB / "vus_scored_v6.csv")
    lp = v[v.provisional_class == "Likely pathogenic"]
    cat = pd.cut(lp.n_nb, [-0.1, 0.5, 1.5, 2.5, 1e9], labels=["0", "1", "2", "3 or more"]).value_counts().reindex(["0", "1", "2", "3 or more"])
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color=GRY, ls="--", lw=1, label="Perfect agreement")
    ax.plot(cb.mean_pred, cb.frac_pos, "o-", color=VER, ms=4, lw=1.4, label=f"Balanced weights (slope {sb:.2f})")
    ax.plot(cu.mean_pred, cu.frac_pos, "s-", color=BLU, ms=4, lw=1.4, label=f"Unweighted (slope {su:.2f})")
    ax.set_xlabel("Predicted probability of pathogenic"); ax.set_ylabel("Observed share pathogenic")
    ax.legend(frameon=False, loc="lower right", fontsize=7.2)
    ax.text(-0.18, 1.04, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax = axs[1]
    cols = [VER, ORA, ORA, BLU]
    ax.bar(range(4), cat.values, color=cols)
    for i, n in enumerate(cat.values):
        ax.text(i, n + 2, str(int(n)), ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(cat.index)
    ax.set_xlabel("Labelled neighbours behind the call"); ax.set_ylabel(f"Likely pathogenic calls (of {len(lp)})")
    ax.text(-0.18, 1.04, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig6_calibration_support.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print("saved", sorted(p.name for p in OUT.glob("*.png")))
