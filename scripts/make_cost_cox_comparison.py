#!/usr/bin/env python3
"""Focused cost versus direct Cox comparison plot for NMF states."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "pancancer_ecosystem"
OUT = ROOT / "figures" / "pancancer_ecosystem_nature" / "focused"

COLORS = {
    "high cost + adverse": "#B23A32",
    "high cost + protective": "#4C6FAE",
    "high cost + neutral": "#8C8C8C",
    "mid cost / mixed": "#C6A15B",
    "stage-increasing high-cost adverse": "#B23A32",
    "stage-decreasing protective": "#4C6FAE",
}


def short_state(x: str) -> str:
    x = str(x)
    state = x.split(" | ", 1)[-1]
    repl = {
        "Proliferating T-cell (cell cycling)": "Prolif. T",
        "CD16+ NK-cell": "CD16 NK",
        "Exhausted CD8+ T-cell": "Exh. CD8",
        "Complete mesenchymal": "Epi mes.",
        "Cell cycling": "Cycling",
        "C1QC+ macrophage": "C1QC mac",
        "SPP1+ macrophage": "SPP1 mac",
        "NLRP3+ monocyte derived macrophage": "NLRP3 mono-mac",
        "Desmoplastic fibroblast": "Desmo fibro.",
        "Heat shock": "Heat shock",
        "Mast": "Mast",
        "Treg": "Treg",
        "Plasma cell": "Plasma",
    }
    label = repl.get(state, state[:20])
    comp = x.split(" | ", 1)[0]
    if label == "Cycling":
        return "Epi cycling" if comp == "epithelial" else "Mye cycling" if comp == "myeloid" else label
    if label == "Heat shock" and comp == "myeloid":
        return "Mye heat shock"
    return label


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.4,
            "axes.linewidth": 0.6,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )
    d = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")
    d = d[d["full_hr"].notna() & ~d["likely_lineage_only"].fillna(False)].copy()
    d["log_hr"] = np.log(d["full_hr"])
    d["plot_class"] = d["progression_cost_prognosis_class"].where(
        d["progression_cost_prognosis_class"].isin(["stage-increasing high-cost adverse", "stage-decreasing protective"]),
        d["cost_prognosis_class"],
    )
    d["color"] = d["plot_class"].map(COLORS).fillna("#BDBDBD")
    d["size"] = 22 + 0.00012 * d["weighted_score"].fillna(0)
    high_cut = d["cost_weighted_score_billion_usd"].quantile(0.66)
    rho, p = stats.spearmanr(d["cost_weighted_score_billion_usd"], d["log_hr"])

    fig, ax = plt.subplots(figsize=(3.4, 3.4), constrained_layout=False)
    ax.axvline(0, color="#888888", lw=0.7, ls="--")
    ax.axhline(high_cut, color="#888888", lw=0.7, ls=":")
    ax.scatter(
        d["log_hr"],
        d["cost_weighted_score_billion_usd"],
        s=d["size"],
        c=d["color"],
        edgecolor="#222222",
        linewidth=0.3,
        alpha=0.88,
    )

    labels = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "epithelial | Cell cycling",
        "myeloid | Cell cycling",
        "epithelial | Complete mesenchymal",
        "myeloid | Heat shock",
        "T_NK | Exhausted CD8+ T-cell",
        "T_NK | CD16+ NK-cell",
        "myeloid | SPP1+ macrophage",
        "mesenchymal | Desmoplastic fibroblast",
        "myeloid | Mast",
        "myeloid | C1QC+ macrophage",
    ]
    offsets = {
        "T_NK | Proliferating T-cell (cell cycling)": (0.006, 0.18),
        "epithelial | Cell cycling": (-0.058, -0.50),
        "myeloid | Cell cycling": (-0.045, 0.46),
        "epithelial | Complete mesenchymal": (0.008, 0.15),
        "myeloid | Heat shock": (0.008, -0.18),
        "T_NK | Exhausted CD8+ T-cell": (0.008, 0.15),
        "T_NK | CD16+ NK-cell": (0.008, 0.15),
        "myeloid | SPP1+ macrophage": (0.008, 0.20),
        "mesenchymal | Desmoplastic fibroblast": (0.008, -0.34),
        "myeloid | Mast": (0.008, 0.10),
        "myeloid | C1QC+ macrophage": (0.008, -0.25),
    }
    for _, r in d[d["state_key"].isin(labels)].iterrows():
        dx, dy = offsets.get(r["state_key"], (0.006, 0.0))
        ha = "right" if dx < 0 else "left"
        ax.text(
            r["log_hr"] + dx,
            r["cost_weighted_score_billion_usd"] + dy,
            short_state(r["state_key"]),
            fontsize=5.0,
            va="center",
            ha=ha,
        )

    ax.text(
        0.02,
        0.96,
        f"Spearman rho={rho:.2f}, p={p:.2g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
    )
    ax.text(0.01, high_cut + 0.35, "high-cost threshold", fontsize=5.0, color="#555555")
    ax.set_xlim(-0.12, 0.36)
    ax.set_ylim(2.5, 26.2)
    ax.set_xlabel("Direct TCGA signature log(HR)")
    ax.set_ylabel("Modeled NCI cost score (billion USD)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=2.2, width=0.55, pad=1.4)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.95)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"cost_vs_direct_cox_nmf_states.{ext}")
    plt.close(fig)

    out_table = d[
        [
            "state_key",
            "cost_weighted_score_billion_usd",
            "full_hr",
            "log_hr",
            "full_p",
            "weighted_score",
            "stage_beta_with_cancer_fixed_effect",
            "stage_fdr",
            "cost_prognosis_class",
            "progression_cost_prognosis_class",
        ]
    ].sort_values("cost_weighted_score_billion_usd", ascending=False)
    out_table.to_csv(OUT / "cost_vs_direct_cox_nmf_states_table.csv", index=False)
    print(f"wrote {OUT / 'cost_vs_direct_cox_nmf_states.pdf'}")
    print(f"wrote {OUT / 'cost_vs_direct_cox_nmf_states_table.csv'}")


if __name__ == "__main__":
    main()
