"""Phase 6 — publication figures (300 dpi PNG) -> results/figures/.

Run:  python -m src.figures
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from src.structure import load_ca

OUT = Path("data/processed")
TAB = Path("results/tables")
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "figure.autolayout": True,
})
# colour-blind-friendly
C = {"P": "#d1495b", "B": "#2e7d95", "VUS": "#e3a72f", "orphan": "#6a4c93",
     "3d": "#d1495b", "1d": "#2e7d95", "a": "#3a6ea5", "muted": "#9aa0a6"}


def fig_pps_by_label():
    df = pd.read_parquet(OUT / "features.parquet")
    d = df[df.label.isin(["P", "B", "VUS"])]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    order = ["B", "VUS", "P"]
    for a, col, ttl in zip(ax, ["pps3d_norm", "coordination"],
                           ["3D Pathogenic Proximity Score", "Residue burial (CA coordination)"]):
        data = [d.loc[d.label.eq(k), col].dropna() for k in order]
        parts = a.violinplot(data, showmeans=True, showextrema=False)
        for pc, k in zip(parts["bodies"], order):
            pc.set_facecolor(C[k]); pc.set_alpha(0.65)
        parts["cmeans"].set_color("black")
        a.set_xticks([1, 2, 3]); a.set_xticklabels(["Benign\n(n=110)", "VUS\n(n=1872)", "Pathogenic\n(n=511)"])
        a.set_title(ttl); a.set_ylabel(col)
    ax[0].axhline(0, ls="--", c=C["muted"], lw=1)
    fig.suptitle("Figure 2. Structural constraint separates pathogenic from benign MYH7 variants",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.savefig(FIG / "fig2_pps_by_label.png", bbox_inches="tight"); plt.close(fig)


def fig_model_performance():
    oof = pd.read_csv(TAB / "oof_predictions.csv")
    cv = pd.read_csv(TAB / "cv_metrics.csv")
    uni = pd.read_csv(TAB / "univariate_aucs.csv")
    y = oof["y"].values
    names = {"oof_A_only": "Sequence tools only", "oof_A_1d": "+1D control",
             "oof_A_3d": "+3D-PPS", "oof_threeD_only": "Structure only",
             "oof_pop_3d": "ClinVar-free + 3D"}
    cols = {"oof_A_only": C["muted"], "oof_A_1d": C["1d"], "oof_A_3d": C["3d"],
            "oof_threeD_only": "#6a4c93", "oof_pop_3d": "#2a9d8f"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # ROC
    for k, lbl in names.items():
        fpr, tpr, _ = roc_curve(y, oof[k]); a = auc(fpr, tpr)
        ax[0].plot(fpr, tpr, color=cols[k], lw=2, label=f"{lbl} (AUC {a:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", c=C["muted"], lw=1)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("A. ROC (5-fold, leakage-controlled)"); ax[0].legend(fontsize=8, loc="lower right")
    # metrics bars
    x = np.arange(len(cv)); w = 0.38
    ax[1].bar(x - w/2, cv["auroc"], w, label="AUROC", color=C["a"])
    ax[1].bar(x + w/2, cv["auprc"], w, label="AUPRC", color=C["P"])
    ax[1].set_xticks(x); ax[1].set_xticklabels(
        [names["oof_"+f].replace(" ", "\n") for f in cv["feature_set"]], fontsize=8)
    ax[1].set_ylim(0.8, 1.0); ax[1].set_title("B. Discrimination by feature set")
    ax[1].legend(fontsize=8)
    # univariate
    u = uni.sort_values("auroc").tail(12)
    barcol = [C["3d"] if "pps3d" in f else (C["1d"] if "pps1d" in f else C["a"]) for f in u["feature"]]
    ax[2].barh(u["feature"], u["auroc"], color=barcol)
    ax[2].set_xlim(0.5, 1.0); ax[2].set_xlabel("Univariate AUROC")
    ax[2].set_title("C. Single-feature power\n(red = 3D-PPS, blue = 1D control)")
    fig.suptitle("Figure 3. A purely structural score rivals top ML predictors and adds independent signal",
                 y=1.04, fontsize=12, fontweight="bold")
    fig.savefig(FIG / "fig3_model_performance.png", bbox_inches="tight"); plt.close(fig)


def fig_discordance():
    vus = pd.read_parquet(OUT / "vus_annotated.parquet")
    order = ["concordant-damaging", "discordant", "concordant-tolerated", "insufficient"]
    counts = vus["triage"].value_counts().reindex(order)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    bar_c = [C["P"], C["VUS"], C["B"], C["muted"]]
    ax[0].bar(range(len(order)), counts.values, color=bar_c)
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels([o.replace("-", "\n") for o in order], fontsize=9)
    ax[0].set_ylabel("VUS count")
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + 8, f"{v}\n({100*v/len(vus):.0f}%)", ha="center", fontsize=9)
    ax[0].set_title(f"A. Tool concordance across {len(vus)} VUS")
    # RCS distribution within discordant vs concordant
    for grp, c, lbl in [("concordant-tolerated", C["B"], "concordant-tolerated"),
                        ("discordant", C["VUS"], "discordant"),
                        ("concordant-damaging", C["P"], "concordant-damaging")]:
        ax[1].hist(vus.loc[vus.triage.eq(grp), "RCS"], bins=25, alpha=0.6, color=c, label=lbl)
    ax[1].set_xlabel("Reclassification Confidence Score (RCS)"); ax[1].set_ylabel("VUS count")
    ax[1].set_title("B. The framework resolves the discordant middle"); ax[1].legend(fontsize=8)
    fig.suptitle("Figure 4. 44% of VUS are discordant between tools — the clinically hardest set",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.savefig(FIG / "fig4_discordance.png", bbox_inches="tight"); plt.close(fig)


def fig_orphans_scatter():
    vus = pd.read_parquet(OUT / "vus_annotated.parquet")
    feat = pd.read_parquet(OUT / "features.parquet")
    orph = pd.read_csv(TAB / "structural_orphans.csv")
    orphset = set(orph["hgvs_g"])
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for lbl, c, s, al in [("B", C["B"], 26, .55), ("P", C["P"], 26, .5), ("VUS", C["VUS"], 14, .3)]:
        d = feat[feat.label.eq(lbl)]
        ax.scatter(d["pps1d_norm"], d["pps3d_norm"], s=s, c=c, alpha=al,
                   label={"B": "Benign", "P": "Pathogenic", "VUS": "VUS"}[lbl], edgecolors="none")
    o = vus[vus.hgvs_g.isin(orphset)]
    ax.scatter(o["pps1d_norm"], o["pps3d_norm"], s=90, facecolors="none",
               edgecolors=C["orphan"], linewidths=2, label=f"Structural orphan (n={len(o)})")
    ax.axhline(0.5, ls="--", c=C["muted"], lw=1); ax.axvline(0.3, ls="--", c=C["muted"], lw=1)
    ax.annotate("3D hotspot\nbut linear-isolated\n(only 3D sees these)", xy=(-0.5, 0.9),
                fontsize=9, color=C["orphan"], fontweight="bold")
    ax.set_xlabel("1D Proximity Score (sequence neighbours)")
    ax.set_ylabel("3D Pathogenic Proximity Score (folded structure)")
    ax.set_title("Figure 5. Structural orphans: pathogenic in 3D, invisible in 1D")
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(FIG / "fig5_orphans_scatter.png", bbox_inches="tight"); plt.close(fig)


def fig_positional():
    feat = pd.read_parquet(OUT / "features.parquet")
    orph = pd.read_csv(TAB / "structural_orphans.csv")
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axvspan(1, 780, color="#3a6ea5", alpha=0.08, label="Motor domain (1–780)")
    ax.axvspan(709, 777, color="#d1495b", alpha=0.12, label="Converter (709–777)")
    p = feat[feat.label.eq("P")]; b = feat[feat.label.eq("B")]
    ax.scatter(p["aapos"], p["pps3d_norm"], s=10, c=C["P"], alpha=.4, label="Pathogenic")
    ax.scatter(b["aapos"], b["pps3d_norm"], s=12, c=C["B"], alpha=.6, label="Benign")
    ax.scatter(orph["aapos"], orph["pps3d_norm"], s=70, facecolors="none",
               edgecolors=C["orphan"], linewidths=1.8, label="Structural orphan")
    ax.set_xlabel("MYH7 residue position"); ax.set_ylabel("3D-PPS")
    ax.set_title("Figure 6. Pathogenic structural constraint concentrates in the myosin motor domain")
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    fig.savefig(FIG / "fig6_positional.png", bbox_inches="tight"); plt.close(fig)


def fig_structure3d():
    """3D CA scatter coloured by pathogenic density — a structural view of the hotspot."""
    feat = pd.read_parquet(OUT / "features.parquet")
    ca = load_ca()
    xyz = np.array([ca[i][0] for i in sorted(ca)])
    resn = np.array(sorted(ca))
    # per-residue max pathogenic-proximity from variant table
    pr = feat.groupby("aapos")["path_density_3d"].max()
    dens = np.array([pr.get(int(r), 0.0) for r in resn])
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=dens, cmap="magma_r", s=6)
    pset = set(feat.loc[feat.label.eq("P"), "aapos"].astype(int))
    mask = np.array([int(r) in pset for r in resn])
    ax.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], c=C["P"], s=16,
               label="Known pathogenic residue")
    ax.set_title("Figure 7. MYH7 fold coloured by pathogenic density\n(dark = 3D hotspot)")
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Pathogenic density (3D)")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(FIG / "fig7_structure3d.png", bbox_inches="tight"); plt.close(fig)


def fig_reclassification():
    vus = pd.read_parquet(OUT / "vus_annotated.parquet")
    order = ["Likely pathogenic", "VUS", "Likely benign", "Benign"]
    counts = vus["provisional_class"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cc = [C["P"], C["VUS"], C["B"], "#1b4965"]
    ax[0].bar(range(len(order)), counts.values, color=cc)
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([o.replace(" ", "\n") for o in order])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + 6, int(v), ha="center")
    ax[0].set_ylabel("VUS count")
    ax[0].set_title("A. Provisional computational reclassification\n(ACMG/Bayesian points)")
    ax[1].hist(vus["acmg_points"], bins=range(-8, 12), color=C["a"], edgecolor="white")
    ax[1].axvline(6, ls="--", c=C["P"]); ax[1].axvline(-1, ls="--", c=C["B"])
    ax[1].set_xlabel("ACMG Bayesian points"); ax[1].set_ylabel("VUS count")
    ax[1].set_title("B. Evidence-strength distribution")
    fig.suptitle("Figure 8. 113 VUS reach Likely-Pathogenic on computational evidence alone",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.savefig(FIG / "fig8_reclassification.png", bbox_inches="tight"); plt.close(fig)


def main():
    fig_pps_by_label(); print("  fig2 ok")
    fig_model_performance(); print("  fig3 ok")
    fig_discordance(); print("  fig4 ok")
    fig_orphans_scatter(); print("  fig5 ok")
    fig_positional(); print("  fig6 ok")
    fig_structure3d(); print("  fig7 ok")
    fig_reclassification(); print("  fig8 ok")
    print(f"[figures] all saved -> {FIG}")


if __name__ == "__main__":
    main()
