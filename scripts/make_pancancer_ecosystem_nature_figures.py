#!/usr/bin/env python3
"""Generate square Nature-style subfigures for pan-cancer ecosystem burden."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pancancer_ecosystem_zenodo10651059"
PROCESSED = ROOT / "data" / "processed" / "pancancer_ecosystem"
FIG = ROOT / "figures" / "pancancer_ecosystem_nature"
SUB = FIG / "subfigures"

COMP_COLORS = {
    "Malignant epithelial": "#4C6FAE",
    "Non-malignant epithelial": "#9B9B9B",
    "T cell": "#2A9D8F",
    "NK cell": "#E08E2E",
    "B cell": "#7A5EA8",
    "Plasma cell": "#8B6F47",
    "Macrophage": "#B23A32",
    "DC/pDC": "#637C8A",
    "Fibroblast": "#B8872C",
    "Endothelial": "#5E8C61",
    "Mast": "#C15A4A",
    "Other": "#C7C7C7",
}

NMF_COLORS = {
    "T_NK": "#2A9D8F",
    "myeloid": "#B23A32",
    "B_plasma": "#7A5EA8",
    "mesenchymal": "#B8872C",
    "epithelial": "#4C6FAE",
}

MODEL_LABELS = {
    "stratified_by_tcga_cancer__age_sex": "Base",
    "stratified_by_tcga_cancer__age_sex_stage_available": "Stage",
    "stratified_by_tcga_cancer__age_sex_purity_available": "Purity",
    "stratified_by_tcga_cancer__age_sex_stage_purity_available": "Full",
}

PRIMARY_NMF_COX_MODEL = "stratified_by_tcga_cancer__age_sex_stage_purity_available"

MONO_COLOR = "#4E6E8E"
MONO_DARK = "#2F4858"
MONO_LIGHT = "#D7E0E7"
SCHEME_FILL = "#F5F6F7"
ADVERSE_COLOR = "#B23A32"
PROTECTIVE_COLOR = "#4C6FAE"
NEUTRAL_COLOR = "#8C8C8C"


def add_point_legend(
    ax: plt.Axes,
    entries: list[tuple[str, str]],
    loc: str = "upper right",
    bbox_to_anchor: tuple[float, float] | None = None,
    fontsize: float = 4.8,
    ncol: int = 1,
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="#222222",
            markeredgewidth=0.25,
            markersize=4.0,
            label=label,
        )
        for label, color in entries
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        handletextpad=0.35,
        borderaxespad=0.2,
        ncol=ncol,
        columnspacing=0.8,
    )


def setup() -> None:
    SUB.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def square_fig(size: float = 2.55) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(size, size), constrained_layout=False)


def style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=2.2, width=0.55, pad=1.4)


def annotate_inside(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    dx: float = 0.0,
    dy: float = 0.0,
    fontsize: float = 5.1,
    ha: str = "left",
) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))
    xr = xmax - xmin
    yr = ymax - ymin
    xx = min(max(x + dx, xmin + 0.04 * xr), xmax - 0.04 * xr)
    yy = min(max(y + dy, ymin + 0.05 * yr), ymax - 0.05 * yr)
    visual_x = (xx - x0) / (x1 - x0) if x1 != x0 else 0.5
    if visual_x > 0.76:
        ha = "right"
    elif visual_x < 0.24:
        ha = "left"
    ax.text(xx, yy, label, fontsize=fontsize, va="center", ha=ha, clip_on=True)


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(SUB / f"{name}.{ext}")
    plt.close(fig)


def write_svg_outputs(name: str, svg: str) -> None:
    svg_path = SUB / f"{name}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(SUB / f"{name}.pdf"))
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(SUB / f"{name}.png"), output_width=1530, output_height=1530)


def load_nmf_umap(filename: str) -> pd.DataFrame:
    path = RAW / "NMF_h5ad" / filename
    adata = ad.read_h5ad(path, backed="r")
    try:
        xy = np.asarray(adata.obsm["X_umap"])
        obs = adata.obs[["Cell_state", "CancerAbbr"]].copy()
    finally:
        adata.file.close()
    out = pd.DataFrame(
        {
            "umap_1": xy[:, 0],
            "umap_2": xy[:, 1],
            "Cell_state": obs["Cell_state"].astype(str).to_numpy(),
            "CancerAbbr": obs["CancerAbbr"].astype(str).to_numpy(),
        }
    )
    return out


def fig1_umap_panel(
    filename: str,
    name: str,
    title: str,
    highlights: list[tuple[str, str, str]],
) -> None:
    d = load_nmf_umap(filename)
    fig, ax = square_fig()
    ax.scatter(d["umap_1"], d["umap_2"], s=1.2, color="#C9CED3", alpha=0.32, linewidths=0, rasterized=True)
    for state, label, color in highlights:
        sub = d[d["Cell_state"].eq(state)]
        if sub.empty:
            continue
        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            s=2.8,
            color=color,
            alpha=0.88,
            linewidths=0,
            rasterized=True,
            label=label,
        )
    xr = d["umap_1"].max() - d["umap_1"].min()
    yr = d["umap_2"].max() - d["umap_2"].min()
    ax.set_xlim(d["umap_1"].min() - 0.03 * xr, d["umap_1"].max() + 0.03 * xr)
    ax.set_ylim(d["umap_2"].min() - 0.03 * yr, d["umap_2"].max() + 0.03 * yr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.98, title, transform=ax.transAxes, va="top", ha="center", fontsize=7.2, weight="bold")
    ax.text(0.02, 0.05, "UMAP of atlas NMF cells", transform=ax.transAxes, va="bottom", ha="left", fontsize=5.2, color="#555555")
    add_point_legend(
        ax,
        [("Other states", "#C9CED3")] + [(label, color) for _state, label, color in highlights],
        loc="lower right",
        fontsize=4.45,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    save(fig, name)


def dotplot_stats(filename: str, states: list[tuple[str, str]], genes: list[str]) -> pd.DataFrame:
    path = RAW / "NMF_h5ad" / filename
    adata = ad.read_h5ad(path, backed="r")
    try:
        present_genes = [g for g in genes if g in adata.var_names]
        state_names = [s for s, _label in states]
        obs_state = adata.obs["Cell_state"].astype(str)
        row_mask = obs_state.isin(state_names).to_numpy()
        sub = adata[row_mask, present_genes].to_memory()
    finally:
        adata.file.close()

    x = sub.X
    if not sp.issparse(x):
        x = np.asarray(x)
    sub_state = sub.obs["Cell_state"].astype(str).to_numpy()
    rows = []
    for state, label in states:
        mask = sub_state == state
        if not mask.any():
            continue
        xm = x[mask]
        mean = np.asarray(xm.mean(axis=0)).ravel()
        if sp.issparse(xm):
            pct = np.asarray((xm > 0).mean(axis=0)).ravel() * 100
        else:
            pct = (xm > 0).mean(axis=0) * 100
        for gene, m, p in zip(present_genes, mean, pct):
            rows.append({"state": state, "label": label, "gene": gene, "mean_expr": float(m), "pct_expr": float(p)})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["scaled_mean"] = 0.0
    for gene, idx in d.groupby("gene", observed=True).groups.items():
        vals = d.loc[idx, "mean_expr"].to_numpy()
        sd = vals.std(ddof=0)
        scaled = vals - vals.mean()
        if sd > 0:
            scaled = scaled / sd
        d.loc[idx, "scaled_mean"] = np.clip(scaled, -1.5, 1.5)
    return d


def fig1_marker_dotplot(
    filename: str,
    name: str,
    title: str,
    states: list[tuple[str, str]],
    genes: list[str],
) -> None:
    d = dotplot_stats(filename, states, genes)
    fig, ax = square_fig()
    if d.empty:
        ax.text(0.5, 0.5, "No marker data", ha="center", va="center", fontsize=7)
        save(fig, name)
        return
    state_labels = [label for _state, label in states if label in set(d["label"])]
    genes_present = [g for g in genes if g in set(d["gene"])]
    xmap = {g: i for i, g in enumerate(genes_present)}
    ymap = {label: i for i, label in enumerate(reversed(state_labels))}
    d["x"] = d["gene"].map(xmap)
    d["y"] = d["label"].map(ymap)
    sizes = 5 + d["pct_expr"].clip(0, 100) * 0.75
    sc = ax.scatter(
        d["x"],
        d["y"],
        s=sizes,
        c=d["scaled_mean"],
        cmap="Blues",
        vmin=-1.5,
        vmax=1.5,
        edgecolor="#222222",
        linewidth=0.18,
    )
    ax.set_xticks(np.arange(len(genes_present)))
    ax.set_xticklabels(genes_present, rotation=45, ha="right", fontsize=5.3)
    ax.set_yticks(np.arange(len(state_labels)))
    ax.set_yticklabels(list(reversed(state_labels)), fontsize=5.8)
    ax.set_xlim(-0.65, len(genes_present) - 0.35)
    ax.set_ylim(-0.65, len(state_labels) - 0.35)
    ax.set_title(title, fontsize=7.2, pad=4, loc="center", weight="bold")
    ax.set_xlabel("Marker genes", fontsize=6.0, labelpad=0.5)
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.set_title("scaled\nmean", fontsize=4.6, pad=2)
    cbar.ax.tick_params(labelsize=4.4, length=1.5, width=0.45, pad=1)
    cbar.outline.set_linewidth(0.45)
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#5F86AF",
            markeredgecolor="#222222",
            markeredgewidth=0.18,
            markersize=np.sqrt(5 + pct * 0.75),
            label=f"{pct}%",
        )
        for pct in (25, 50, 75)
    ]
    ax.legend(
        handles=size_handles,
        title="% cells",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1.24, -0.03),
        fontsize=4.1,
        title_fontsize=4.5,
        handletextpad=0.25,
        borderaxespad=0.1,
    )
    fig.subplots_adjust(left=0.30, right=0.74, bottom=0.27, top=0.86)
    save(fig, name)


def svg_box(x: float, y: float, w: float, h: float, text: str, fill: str = SCHEME_FILL, stroke: str = "#222222") -> str:
    lines = text.split("\n")
    line_h = 13
    total_h = (len(lines) - 1) * line_h
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.1"/>']
    for i, line in enumerate(lines):
        yy = y + h / 2 - total_h / 2 + i * line_h + 3
        parts.append(
            f'<text x="{x + w / 2}" y="{yy}" text-anchor="middle" font-size="10.2" fill="#222222">{escape(line)}</text>'
        )
    return "\n".join(parts)


def svg_arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#222222") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.2" marker-end="url(#arrow)"/>'


def svg_template(body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="183.6pt" height="183.6pt" viewBox="0 0 255 255">
<defs>
  <marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="#222222"/>
  </marker>
  <style>
    text {{ font-family: DejaVu Sans, Arial, sans-serif; }}
    .small {{ font-size: 8.6px; fill: #555555; }}
    .title {{ font-size: 11.4px; font-weight: 600; fill: #222222; }}
  </style>
</defs>
<rect x="0" y="0" width="255" height="255" fill="white"/>
{body}
</svg>
'''


def short_state(x: str) -> str:
    x = str(x)
    comp = nmf_comp(x)
    state = x.replace("T_NK | ", "").replace("myeloid | ", "").replace("B_plasma | ", "")
    state = state.replace("mesenchymal | ", "").replace("epithelial | ", "")
    repl = {
        "Proliferating T-cell (cell cycling)": "Prolif. T",
        "Tissue resident memory T-cell": "TRM T",
        "Exhausted CD8+ T-cell": "Exh. CD8",
        "FCN1+ monocyte derived macrophage": "FCN1 mono-mac",
        "NLRP3+ monocyte derived macrophage": "NLRP3 mono-mac",
        "C1QC+ macrophage": "C1QC mac",
        "SPP1+ macrophage": "SPP1 mac",
        "CXCL9+ macrophage": "CXCL9 mac",
        "ISG15+ macrophage": "ISG15 mac",
        "Cell cycling": "Cycling",
        "Heat shock": "Heat shock",
        "CD16+ NK-cell": "CD16 NK",
        "XCL1+ NK-cell": "XCL1 NK",
        "Plasma cell": "Plasma",
        "Precursor B-cell": "Precursor B",
        "Complete mesenchymal": "Epi mes.",
        "Germinal center B-cell": "GC B",
        "PI16+ fibroblast": "PI16 fibro.",
        "Desmoplastic fibroblast": "Desmo fibro.",
        "Myofibroblast": "Myofibro.",
    }
    label = repl.get(state, state[:22])
    if comp == "B_plasma" and label in {"Cycling", "Heat shock"}:
        return "B " + label.lower()
    if comp == "myeloid" and label in {"Cycling", "Heat shock"}:
        return "Mye " + label.lower()
    if comp == "mesenchymal" and label == "Cycling":
        return "Mes cycling"
    if comp == "epithelial" and label == "Cycling":
        return "Epi cycling"
    if comp == "epithelial" and label == "Stress":
        return "Epi stress"
    return label


def nmf_comp(state_key: str) -> str:
    return str(state_key).split(" | ")[0]


def fig1a_design() -> None:
    body = [
        '<text class="title" x="127.5" y="21" text-anchor="middle">Pan-cancer ecosystem burden model</text>',
        svg_box(22, 45, 75, 44, "Zenodo atlas\n104 scRNA datasets\n3.49M tumor cells"),
        svg_box(158, 45, 75, 44, "Cancer burden\nGLOBOCAN\nWHO DALY/YLL"),
        svg_box(22, 123, 75, 44, "NMF states\n5 lineages\n98 modules"),
        svg_box(158, 123, 75, 44, "Modeled cost\nNCI site-level\ncare costs"),
        svg_box(80, 190, 95, 42, "TCGA validation\nCox HR + stage\nwithin-cancer z"),
        svg_arrow(97, 67, 158, 67),
        svg_arrow(59.5, 89, 59.5, 123),
        svg_arrow(195.5, 89, 195.5, 123),
        svg_arrow(80, 167, 100, 190),
        svg_arrow(175, 167, 155, 190),
        svg_arrow(127.5, 89, 127.5, 190),
        '<text class="small" x="127.5" y="107" text-anchor="middle">state representation x population/cost weights</text>',
    ]
    write_svg_outputs("fig1a_design", svg_template("\n".join(body)))


def fig1b_coverage() -> None:
    counts = pd.read_csv(PROCESSED / "ecosystem_sample_compartment_fractions_wide.csv")
    d = counts.groupby("Cancer type", observed=True).agg(n_samples=("sample_key", "nunique"), n_cells=("n_tumor_cells_total", "sum")).reset_index()
    d = d.sort_values("n_samples").tail(12)
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["n_samples"], color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Cancer type"])
    ax.set_xlabel("Tumor samples")
    ax.set_title("Atlas cancer coverage", fontsize=7.4, pad=4)
    style(ax)
    fig.subplots_adjust(left=0.30, right=0.96, bottom=0.16, top=0.88)
    save(fig, "fig1b_coverage")


def fig1c_data_layers() -> None:
    manifest = pd.read_csv(PROCESSED / "ecosystem_atlas_file_manifest.csv")
    samples = pd.read_csv(PROCESSED / "ecosystem_sample_compartment_fractions_wide.csv")
    states = pd.read_csv(PROCESSED / "ecosystem_nmf_state_priority_review.csv")
    tcga = pd.read_csv(PROCESSED / "tcga_nmf_state_signature_scores_survival_merged.csv.gz", usecols=["sample", "stage_ordinal"])
    rows = [
        ("Atlas datasets", len(manifest)),
        ("Tumor samples", samples["sample_key"].nunique()),
        ("Tumor cells", samples.drop_duplicates("sample_key")["n_tumor_cells_total"].sum() / 1e6),
        ("NMF states", states["state_key"].nunique()),
        ("TCGA samples", tcga["sample"].nunique() / 1000),
        ("TCGA staged", tcga["stage_ordinal"].notna().sum() / 1000),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    text = ["104", "1,066", "3.49M", "98", "11.0k", "7.37k"]
    fig, ax = square_fig()
    y = np.arange(len(rows))
    ax.barh(y, values, color=MONO_COLOR)
    for i, (v, t) in enumerate(zip(values, text)):
        ax.text(v * 1.03, i, t, va="center", fontsize=6.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Analysis coverage\n(log scale)")
    style(ax)
    fig.subplots_adjust(left=0.38, right=0.86, bottom=0.22, top=0.96)
    save(fig, "fig1c_data_layers")


def fig1d_epithelial_umap() -> None:
    fig1_umap_panel(
        "epi_NMF.h5ad",
        "fig1d_epithelial_umap",
        "Epithelial NMF states",
        [
            ("Cell cycling", "Cell cycling", ADVERSE_COLOR),
            ("Complete mesenchymal", "Mesenchymal", "#B8872C"),
            ("Stress", "Stress", PROTECTIVE_COLOR),
        ],
    )


def fig1e_tnk_umap() -> None:
    fig1_umap_panel(
        "tnk_NMF.h5ad",
        "fig1e_tnk_umap",
        "T/NK NMF states",
        [
            ("Proliferating T-cell (cell cycling)", "Prolif. T", ADVERSE_COLOR),
            ("Exhausted CD8+ T-cell", "Exh. CD8", PROTECTIVE_COLOR),
            ("Treg", "Treg", "#7A5EA8"),
        ],
    )


def fig1f_myeloid_umap() -> None:
    fig1_umap_panel(
        "myl_NMF.h5ad",
        "fig1f_myeloid_umap",
        "Myeloid NMF states",
        [
            ("Cell cycling", "Cycling", ADVERSE_COLOR),
            ("SPP1+ macrophage", "SPP1 mac", "#B8872C"),
            ("C1QC+ macrophage", "C1QC mac", PROTECTIVE_COLOR),
            ("Heat shock", "Heat shock", "#7A5EA8"),
        ],
    )


def fig1g_epithelial_dotplot() -> None:
    fig1_marker_dotplot(
        "epi_NMF.h5ad",
        "fig1g_epithelial_dotplot",
        "Epithelial state markers",
        [
            ("Cell cycling", "Cycling"),
            ("Complete mesenchymal", "Mesenchymal"),
            ("Stress", "Stress"),
        ],
        ["UBE2C", "TOP2A", "COL1A1", "DCN", "CXCL8", "KRT13"],
    )


def fig1h_tnk_dotplot() -> None:
    fig1_marker_dotplot(
        "tnk_NMF.h5ad",
        "fig1h_tnk_dotplot",
        "T/NK state markers",
        [
            ("Proliferating T-cell (cell cycling)", "Prolif. T"),
            ("Exhausted CD8+ T-cell", "Exh. CD8"),
            ("Treg", "Treg"),
        ],
        ["PCLAF", "UBE2C", "LAG3", "CD8A", "IL2RA", "FOXP3"],
    )


def fig1i_myeloid_dotplot() -> None:
    fig1_marker_dotplot(
        "myl_NMF.h5ad",
        "fig1i_myeloid_dotplot",
        "Myeloid state markers",
        [
            ("Cell cycling", "Cycling"),
            ("SPP1+ macrophage", "SPP1 mac"),
            ("C1QC+ macrophage", "C1QC mac"),
            ("Heat shock", "Heat shock"),
        ],
        ["PCLAF", "TOP2A", "SLC2A1", "HK2", "GPNMB", "CTSD", "HSPA6", "HSPA1A"],
    )


def fig1d_interpretation_scheme() -> None:
    body = [
        '<text class="title" x="127.5" y="21" text-anchor="middle">Interpreting expensive states</text>',
        svg_box(20, 47, 82, 40, "Modeled cost\nrepresentation"),
        svg_box(153, 47, 82, 40, "Direct Cox\nsurvival HR"),
        svg_box(20, 122, 82, 40, "Clinical stage\ntrend"),
        svg_box(153, 122, 82, 40, "Within-cancer\nvariation"),
        svg_box(68, 193, 119, 38, "Progression-cost-\nprognosis class"),
        svg_arrow(102, 67, 153, 67),
        svg_arrow(61, 87, 61, 122),
        svg_arrow(194, 87, 194, 122),
        svg_arrow(80, 162, 100, 193),
        svg_arrow(175, 162, 155, 193),
        '<text class="small" x="127.5" y="104" text-anchor="middle">high cost is interpreted only with outcome and stage</text>',
        '<circle cx="69" cy="181" r="5" fill="#B23A32"/><text class="small" x="79" y="184">adverse</text>',
        '<circle cx="136" cy="181" r="5" fill="#4C6FAE"/><text class="small" x="146" y="184">protective</text>',
    ]
    write_svg_outputs("fig1d_interpretation_scheme", svg_template("\n".join(body)))


def fig2a_major_mortality() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_major_compartment_priority_review.csv")
    d = d[~d["compartment"].isin(["Other", "Non-malignant epithelial"])].sort_values("weighted_score").tail(9)
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["weighted_score"] / 1e6, color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(d["compartment"])
    ax.set_xlabel("Global mortality score\n(million)")
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.22, top=0.96)
    save(fig, "fig2a_major_mortality")


def fig2b_major_cost() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_major_compartment_priority_review.csv")
    d = d[~d["compartment"].isin(["Other", "Non-malignant epithelial"])].sort_values("cost_weighted_score_billion_usd").tail(9)
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["cost_weighted_score_billion_usd"], color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(d["compartment"])
    ax.set_xlabel("Modeled US cost score\n(billion USD)")
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.22, top=0.96)
    save(fig, "fig2b_major_cost")


def fig2c_tcga_proxy() -> None:
    cox = pd.read_csv(PROCESSED / "tcga_cibersort_all_stratified_cox.csv")
    keep = [
        "macrophage_total",
        "Macrophages.M2",
        "Mast.cells.activated",
        "Monocytes",
        "T.cells.CD8",
        "T.cells.regulatory..Tregs.",
        "T.cells.follicular.helper",
        "Plasma.cells",
    ]
    labels = {
        "macrophage_total": "Mac total",
        "Macrophages.M2": "M2 mac",
        "Mast.cells.activated": "Mast act.",
        "Monocytes": "Monocytes",
        "T.cells.CD8": "CD8 T",
        "T.cells.regulatory..Tregs.": "Treg",
        "T.cells.follicular.helper": "Tfh",
        "Plasma.cells": "Plasma",
    }
    d = cox[cox["cell_type"].isin(keep)].copy()
    d["label"] = d["cell_type"].map(labels)
    d = d.set_index("cell_type").loc[keep].reset_index()
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.errorbar(
        d["hr_per_within_cancer_sd"],
        y,
        xerr=[d["hr_per_within_cancer_sd"] - d["ci95_low"], d["ci95_high"] - d["hr_per_within_cancer_sd"]],
        fmt="none",
        color="#555555",
        ecolor="#555555",
        elinewidth=0.8,
        capsize=2,
    )
    colors = [ADVERSE_COLOR if x > 1 and p < 0.05 else PROTECTIVE_COLOR if x < 1 and p < 0.05 else NEUTRAL_COLOR for x, p in zip(d["hr_per_within_cancer_sd"], d["p"])]
    ax.scatter(d["hr_per_within_cancer_sd"], y, s=18, c=colors, zorder=3)
    ax.axvline(1, color="#888888", lw=0.65, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.invert_yaxis()
    ax.set_xlabel("OS HR per\nwithin-cancer SD")
    style(ax)
    add_point_legend(
        ax,
        [("HR>1 p<0.05", ADVERSE_COLOR), ("HR<1 p<0.05", PROTECTIVE_COLOR), ("NS", NEUTRAL_COLOR)],
        loc="lower right",
        fontsize=4.25,
    )
    fig.subplots_adjust(left=0.34, right=0.96, bottom=0.22, top=0.96)
    save(fig, "fig2c_tcga_proxy")


def fig2d_major_cost_mortality() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_major_compartment_priority_review.csv")
    d = d[~d["compartment"].isin(["Other"])].copy()
    fig, ax = square_fig()
    ax.scatter(
        d["weighted_score"] / 1e6,
        d["cost_weighted_score_billion_usd"],
        s=34,
        color=MONO_COLOR,
        edgecolor="#222222",
        linewidth=0.3,
        alpha=0.85,
    )
    ax.set_xlim(0, d["weighted_score"].max() / 1e6 * 1.18)
    ax.set_ylim(0, d["cost_weighted_score_billion_usd"].max() * 1.10)
    label_map = {
        "T cell": "T cell",
        "Non-malignant epithelial": "Non-malignant\nepi.",
        "B cell": "B cell",
        "Malignant epithelial": "Malignant\nepi.",
        "Macrophage": "Macrophage",
        "Fibroblast": "Fibroblast",
    }
    offsets = {
        "T cell": (-0.07, 0.0),
        "Non-malignant epithelial": (0.05, 0.0),
        "B cell": (0.04, 0.0),
        "Malignant epithelial": (0.04, 0.0),
        "Macrophage": (0.04, 0.0),
        "Fibroblast": (0.04, 0.0),
    }
    for _, r in d[d["compartment"].isin(label_map)].iterrows():
        dx, dy = offsets.get(r["compartment"], (0.04, 0.0))
        annotate_inside(
            ax,
            r["weighted_score"] / 1e6,
            r["cost_weighted_score_billion_usd"],
            label_map[r["compartment"]],
            dx=dx,
            dy=dy,
            fontsize=4.5,
            ha="right" if dx < 0 else "left",
        )
    ax.set_xlabel("Global mortality score\n(million)")
    ax.set_ylabel("Modeled US cost score\n(billion USD)")
    style(ax)
    fig.subplots_adjust(left=0.22, right=0.94, bottom=0.24, top=0.96)
    save(fig, "fig2d_major_cost_mortality")


def fig2e_major_cost_proxy_hr() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_major_compartment_priority_review.csv")
    d = d[d["tcga_proxy"].notna()].copy()
    d["proxy_hr"] = d["hr_per_within_cancer_sd"]
    fig, ax = square_fig()
    ax.axvline(1, color="#888888", lw=0.65, ls="--")
    ax.scatter(
        d["proxy_hr"],
        d["cost_weighted_score_billion_usd"],
        s=38,
        color=MONO_COLOR,
        edgecolor="#222222",
        linewidth=0.3,
        alpha=0.85,
    )
    ax.set_xlim(0.925, 1.125)
    ax.set_ylim(0, d["cost_weighted_score_billion_usd"].max() * 1.10)
    label_map = {
        "T cell": "T cell",
        "B cell": "B cell",
        "Macrophage": "Mac",
        "Mast": "Mast",
        "NK cell": "NK",
        "Plasma cell": "Plasma",
        "DC/pDC": "DC/pDC",
    }
    offsets = {
        "T cell": (0.004, 0.0),
        "B cell": (0.004, 0.0),
        "Macrophage": (-0.006, 0.6),
        "Mast": (0.004, 0.0),
        "NK cell": (0.004, 0.0),
        "Plasma cell": (0.004, 0.0),
        "DC/pDC": (0.004, 0.0),
    }
    for _, r in d.iterrows():
        dx, dy = offsets.get(r["compartment"], (0.004, 0.0))
        annotate_inside(
            ax,
            r["proxy_hr"],
            r["cost_weighted_score_billion_usd"],
            label_map.get(r["compartment"], r["compartment"]),
            dx=dx,
            dy=dy,
            fontsize=4.7,
            ha="right" if dx < 0 else "left",
        )
    ax.set_xlabel("TCGA proxy OS HR")
    ax.set_ylabel("Modeled US cost score\n(billion USD)")
    style(ax)
    fig.subplots_adjust(left=0.22, right=0.94, bottom=0.22, top=0.96)
    save(fig, "fig2e_major_cost_proxy_hr")


def fig2f_major_variance_cost() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_major_compartment_priority_review.csv")
    d = d[~d["compartment"].isin(["Other", "Non-malignant epithelial"])].copy()
    d = d.sort_values("cost_weighted_score_billion_usd").tail(8)
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["between_cancer_variance_share"] * 100, color=MONO_LIGHT, label="Between")
    ax.barh(y, d["within_cancer_variance_share"] * 100, left=d["between_cancer_variance_share"] * 100, color=MONO_COLOR, label="Within")
    ax.set_yticks(y)
    ax.set_yticklabels(d["compartment"])
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Variance share (%)")
    style(ax)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.98, 0.03), handlelength=1.2, fontsize=5.0)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.19, top=0.96)
    save(fig, "fig2f_major_variance_cost")


def fig3a_state_raw() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_nmf_state_priority_review.csv")
    d = d[d["coverage_ok"] & ~d["likely_lineage_only"]].sort_values("weighted_score").tail(12)
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["weighted_score"] / 1000, color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels([short_state(x) for x in d["state_key"]])
    ax.set_xlabel("Global mortality score\n(thousand)")
    style(ax)
    fig.subplots_adjust(left=0.38, right=0.96, bottom=0.22, top=0.96)
    save(fig, "fig3a_nmf_raw")


def fig3b_state_adverse() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_nmf_state_priority_review.csv")
    d = d[
        d["coverage_ok"]
        & d["within_cancer_supported"]
        & ~d["likely_lineage_only"]
        & (d["direct_prognosis_weighted_global_mortality_score"] > 0)
    ].sort_values("direct_prognosis_weighted_global_mortality_score").tail(10)
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["direct_prognosis_weighted_global_mortality_score"] / 1000, color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels([short_state(x) for x in d["state_key"]])
    ax.set_xlabel("Direct Cox-weighted\nmortality score (thousand)")
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.24, top=0.96)
    save(fig, "fig3b_nmf_adverse")


def fig3c_raw_vs_adverse() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_nmf_state_priority_review.csv")
    d = d[d["coverage_ok"] & ~d["likely_lineage_only"]].copy()
    fig, ax = square_fig()
    ax.scatter(
        d["weighted_score"] / 1000,
        d["direct_prognosis_weighted_global_mortality_score"] / 1000,
        s=20,
        color=MONO_COLOR,
        alpha=0.78,
        edgecolor="#222222",
        linewidth=0.25,
    )
    label_states = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "epithelial | Cell cycling",
        "myeloid | Cell cycling",
        "myeloid | SPP1+ macrophage",
        "epithelial | Complete mesenchymal",
    ]
    ax.set_xlim(0, d["weighted_score"].max() / 1000 * 1.14)
    ax.set_ylim(-5, d["direct_prognosis_weighted_global_mortality_score"].max() / 1000 * 1.12)
    offsets = {
        "T_NK | Proliferating T-cell (cell cycling)": (-12, 0),
        "epithelial | Cell cycling": (8, 0),
        "myeloid | Cell cycling": (8, 0),
        "myeloid | SPP1+ macrophage": (8, 0),
        "epithelial | Complete mesenchymal": (8, 0),
    }
    for _, r in d[d["state_key"].isin(label_states)].iterrows():
        dx, dy = offsets.get(r["state_key"], (8, 0))
        annotate_inside(
            ax,
            r["weighted_score"] / 1000,
            r["direct_prognosis_weighted_global_mortality_score"] / 1000,
            short_state(r["state_key"]),
            dx=dx,
            dy=dy,
            fontsize=4.8,
            ha="right" if dx < 0 else "left",
        )
    ax.set_xlabel("Raw mortality score\n(thousand)")
    ax.set_ylabel("Direct Cox-weighted\nscore (thousand)")
    style(ax)
    fig.subplots_adjust(left=0.20, right=0.94, bottom=0.24, top=0.96)
    save(fig, "fig3c_raw_vs_adverse")


def fig3d_cost_prognosis_quadrant() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")
    d = d[~d["likely_lineage_only"].fillna(False)].copy()
    d = d[d["full_hr"].notna()].copy()
    median_cost = d["cost_weighted_score_billion_usd"].median()
    fig, ax = square_fig()
    ax.axvline(1, color="#888888", lw=0.65, ls="--")
    ax.axhline(median_cost, color="#888888", lw=0.65, ls=":")
    ax.scatter(
        d["full_hr"],
        d["cost_weighted_score_billion_usd"],
        s=34,
        color=MONO_COLOR,
        edgecolor="#222222",
        linewidth=0.25,
        alpha=0.82,
    )
    label_states = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "T_NK | Exhausted CD8+ T-cell",
        "epithelial | Cell cycling",
        "epithelial | Complete mesenchymal",
        "myeloid | Cell cycling",
        "myeloid | Heat shock",
        "myeloid | Mast",
    ]
    ax.set_xlim(0.88, max(1.38, d["full_hr"].max() * 1.04))
    ax.set_ylim(0, d["cost_weighted_score_billion_usd"].max() * 1.10)
    label_offsets = {
        "T_NK | Proliferating T-cell (cell cycling)": (-0.008, 0.3),
        "epithelial | Cell cycling": (-0.012, -0.5),
        "myeloid | Cell cycling": (-0.012, 0.2),
    }
    for _, r in d[d["state_key"].isin(label_states)].iterrows():
        dx, dy = label_offsets.get(r["state_key"], (0.008, 0.0))
        annotate_inside(ax, r["full_hr"], r["cost_weighted_score_billion_usd"], short_state(r["state_key"]), dx=dx, dy=dy, fontsize=4.9)
    ax.set_xlabel("Direct signature OS HR")
    ax.set_ylabel("Modeled US cost score\n(billion USD)")
    style(ax)
    fig.subplots_adjust(left=0.22, right=0.94, bottom=0.22, top=0.96)
    save(fig, "fig3d_cost_prognosis_quadrant")


def fig3e_cost_burden_hr_bubble() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")
    d = d[~d["likely_lineage_only"].fillna(False) & d["full_hr"].notna()].copy()
    d["abs_stage"] = d["stage_beta_with_cancer_fixed_effect"].fillna(0).clip(lower=0)
    sizes = 28 + 260 * d["abs_stage"]
    colors = np.log(d["full_hr"])
    fig, ax = square_fig()
    sc = ax.scatter(
        d["weighted_score"] / 1000,
        d["cost_weighted_score_billion_usd"],
        s=sizes,
        c=colors,
        cmap="RdBu_r",
        vmin=-0.10,
        vmax=0.30,
        edgecolor="#222222",
        linewidth=0.25,
        alpha=0.84,
    )
    ax.set_xlim(0, d["weighted_score"].max() / 1000 * 1.24)
    ax.set_ylim(0, d["cost_weighted_score_billion_usd"].max() * 1.10)
    for _, r in d.sort_values("cost_weighted_score_billion_usd", ascending=False).head(6).iterrows():
        annotate_inside(ax, r["weighted_score"] / 1000, r["cost_weighted_score_billion_usd"], short_state(r["state_key"]), dx=14, fontsize=4.8)
    ax.set_xlabel("Global mortality score\n(thousand)")
    ax.set_ylabel("Modeled US cost score\n(billion USD)")
    style(ax)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("log HR", rotation=90, labelpad=5)
    cbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.22, right=0.84, bottom=0.24, top=0.96)
    save(fig, "fig3e_cost_burden_hr_bubble")


def fig3f_cost_class_counts() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")
    d = d[~d["likely_lineage_only"].fillna(False)].copy()
    order = [
        "stage-increasing high-cost adverse",
        "high cost + protective",
        "high cost + neutral",
        "mid cost / mixed",
        "low cost + neutral",
    ]
    counts = d["progression_cost_prognosis_class"].value_counts().reindex(order).dropna()
    fig, ax = square_fig()
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels([x.replace("stage-increasing ", "stage-up\n").replace(" + ", " +\n") for x in counts.index], fontsize=5.7)
    ax.set_xlabel("NMF states")
    style(ax)
    fig.subplots_adjust(left=0.48, right=0.96, bottom=0.18, top=0.96)
    save(fig, "fig3f_cost_class_counts")


def fig4a_comp_variance() -> None:
    d = pd.read_csv(PROCESSED / "major_compartment_variance_decomposition.csv")
    d = d[~d["compartment"].isin(["Other", "Non-malignant epithelial"])].sort_values("within_cancer_variance_share")
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["within_cancer_variance_share"] * 100, color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(d["compartment"])
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Within-cancer variation (%)")
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.18, top=0.96)
    save(fig, "fig4a_comp_variance")


def fig4b_state_variance() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_nmf_state_priority_review.csv")
    keep = [
        "T_NK | Treg",
        "T_NK | Proliferating T-cell (cell cycling)",
        "epithelial | Cell cycling",
        "epithelial | Complete mesenchymal",
        "mesenchymal | Desmoplastic fibroblast",
        "myeloid | SPP1+ macrophage",
        "myeloid | Cell cycling",
        "myeloid | Heat shock",
    ]
    d = d[d["state_key"].isin(keep)].copy().sort_values("within_cancer_variance_share")
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["within_cancer_variance_share"] * 100, color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels([short_state(x) for x in d["state_key"]])
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Within-cancer variation (%)")
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.18, top=0.96)
    save(fig, "fig4b_state_variance")


def fig4c_leave_one() -> None:
    lo = pd.read_csv(PROCESSED / "nmf_state_global_mortality_leave_one_cancer_out.csv")
    keep = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "epithelial | Cell cycling",
        "myeloid | Cell cycling",
        "myeloid | SPP1+ macrophage",
        "epithelial | Complete mesenchymal",
        "mesenchymal | Desmoplastic fibroblast",
        "myeloid | Heat shock",
    ]
    d = lo[lo["state_key"].isin(keep)].copy()
    summary = d.groupby("state_key", observed=True).agg(rank_min=("rank_without_cancer", "min"), rank_max=("rank_without_cancer", "max"), rank_median=("rank_without_cancer", "median")).reset_index()
    summary = summary.sort_values("rank_median", ascending=False)
    fig, ax = square_fig()
    y = np.arange(len(summary))
    for i, r in enumerate(summary.itertuples(index=False)):
        ax.plot([r.rank_min, r.rank_max], [i, i], color=MONO_COLOR, lw=1.5)
        ax.scatter([r.rank_median], [i], color=MONO_DARK, s=18, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([short_state(x) for x in summary["state_key"]])
    ax.invert_xaxis()
    ax.set_xlabel("Leave-one-cancer rank\n(lower is stronger)")
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.24, top=0.96)
    save(fig, "fig4c_leave_one_rank")


def fig4d_stage_gradient_rank() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_state_stage_gradients.csv")
    show = pd.concat([d.sort_values("stage_beta_with_cancer_fixed_effect", ascending=False).head(8), d.sort_values("stage_beta_with_cancer_fixed_effect", ascending=True).head(2)])
    show = show.drop_duplicates("state_key").sort_values("stage_beta_with_cancer_fixed_effect")
    fig, ax = square_fig()
    y = np.arange(len(show))
    ax.barh(y, show["stage_beta_with_cancer_fixed_effect"], color=MONO_COLOR)
    ax.axvline(0, color="#888888", lw=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([short_state(x) for x in show["state_key"]])
    ax.set_xlabel("Stage beta\nwithin cancer")
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.22, top=0.96)
    save(fig, "fig4d_stage_gradient_rank")


def fig4e_stage_trajectories() -> None:
    means = pd.read_csv(PROCESSED / "ecosystem_state_stage_means_selected.csv")
    keep = [
        "mesenchymal | Desmoplastic fibroblast",
        "epithelial | Complete mesenchymal",
        "myeloid | SPP1+ macrophage",
        "epithelial | Cell cycling",
        "T_NK | Proliferating T-cell (cell cycling)",
        "myeloid | Mast",
    ]
    d = means[means["state_key"].isin(keep)].copy()
    fig, ax = square_fig()
    trajectory_colors = {
        state: color
        for state, color in zip(
            keep,
            ["#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#76B7B2", "#B07AA1"],
        )
    }
    for state in keep:
        sub = d[d["state_key"].eq(state)].sort_values("stage_ordinal")
        if sub.empty:
            continue
        ax.plot(
            sub["stage_ordinal"],
            sub["mean_within_cancer_z"],
            marker="o",
            ms=3.0,
            lw=1.0,
            color=trajectory_colors[state],
            label=short_state(state),
        )
    ax.axhline(0, color="#888888", lw=0.55, ls="--")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Clinical stage")
    ax.set_ylabel("Mean within-cancer z")
    style(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.00, 1.00), fontsize=4.7, handlelength=1.2)
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.20, top=0.96)
    save(fig, "fig4e_stage_trajectories")


def fig4f_state_axis_stage() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_state_axis_summary.csv")
    labels = {
        "epithelial_cycling_to_mesenchymal_stress": "Epi cycling-\nmesenchymal",
        "fibroblast_quiescent_to_desmoplastic": "Fibroblast\nactivation",
        "myeloid_resident_to_inflammatory_spp1_cycling": "Myeloid\ninflammatory",
        "tnk_cytotoxic_to_regulatory_proliferating": "T/NK\nactivation",
    }
    d["label"] = d["axis"].map(labels)
    d = d.sort_values("stage_beta_with_cancer_fixed_effect")
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["stage_beta_with_cancer_fixed_effect"], color=MONO_COLOR)
    ax.set_xlim(0, d["stage_beta_with_cancer_fixed_effect"].max() * 1.18)
    for i, r in enumerate(d.itertuples(index=False)):
        qtxt = "q<1e-20" if r.stage_fdr < 1e-20 else f"q={r.stage_fdr:.1e}"
        ax.text(
            r.stage_beta_with_cancer_fixed_effect - 0.002,
            i,
            qtxt,
            va="center",
            ha="right",
            fontsize=4.4,
            color="white" if r.stage_beta_with_cancer_fixed_effect > 0.05 else "#222222",
            clip_on=True,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Axis stage beta\nwithin cancer")
    style(ax)
    fig.subplots_adjust(left=0.42, right=0.94, bottom=0.22, top=0.96)
    save(fig, "fig4f_state_axis_stage")


def fig5a_direct_cox_forest() -> None:
    cox = pd.read_csv(PROCESSED / "tcga_nmf_state_signature_stratified_cox.csv")
    keep = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "epithelial | Cell cycling",
        "myeloid | Cell cycling",
        "myeloid | SPP1+ macrophage",
        "epithelial | Complete mesenchymal",
        "mesenchymal | Desmoplastic fibroblast",
        "myeloid | NLRP3+ monocyte derived macrophage",
        "myeloid | Heat shock",
        "epithelial | Stress",
        "myeloid | C1QC+ macrophage",
        "T_NK | Exhausted CD8+ T-cell",
        "myeloid | Mast",
    ]
    d = cox[(cox["model"] == PRIMARY_NMF_COX_MODEL) & cox["state_key"].isin(keep)].copy()
    d["state_key"] = pd.Categorical(d["state_key"], categories=keep, ordered=True)
    d = d.sort_values("state_key", ascending=False)
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.errorbar(
        d["hr_per_within_cancer_sd"],
        y,
        xerr=[
            d["hr_per_within_cancer_sd"] - d["ci95_low"],
            d["ci95_high"] - d["hr_per_within_cancer_sd"],
        ],
        fmt="none",
        color="#555555",
        ecolor="#555555",
        elinewidth=0.75,
        capsize=1.8,
        zorder=1,
    )
    ax.scatter(d["hr_per_within_cancer_sd"], y, s=18, color=MONO_COLOR, edgecolor="#222222", linewidth=0.25, zorder=3)
    ax.axvline(1, color="#888888", lw=0.65, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([short_state(x) for x in d["state_key"].astype(str)])
    ax.set_xlim(0.86, 1.46)
    ax.set_xlabel("Direct signature OS HR\nper within-cancer SD")
    style(ax)
    fig.subplots_adjust(left=0.40, right=0.96, bottom=0.23, top=0.96)
    save(fig, "fig5a_direct_nmf_cox")


def fig5b_direct_cox_models() -> None:
    cox = pd.read_csv(PROCESSED / "tcga_nmf_state_signature_stratified_cox.csv")
    keep_states = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "epithelial | Cell cycling",
        "myeloid | Cell cycling",
        "myeloid | SPP1+ macrophage",
        "epithelial | Complete mesenchymal",
        "mesenchymal | Desmoplastic fibroblast",
        "myeloid | NLRP3+ monocyte derived macrophage",
        "myeloid | Heat shock",
    ]
    keep_models = list(MODEL_LABELS)
    d = cox[cox["state_key"].isin(keep_states) & cox["model"].isin(keep_models)].copy()
    d["x"] = d["model"].map({m: i for i, m in enumerate(keep_models)})
    d["y"] = d["state_key"].map({s: i for i, s in enumerate(reversed(keep_states))})
    d["log_hr"] = np.log(d["hr_per_within_cancer_sd"])
    fig, ax = square_fig()
    sc = ax.scatter(
        d["x"],
        d["y"],
        c=d["log_hr"],
        s=np.where(d["p"] < 0.05, 42, 18),
        cmap="RdBu_r",
        vmin=-0.10,
        vmax=0.30,
        edgecolor="#222222",
        linewidth=0.25,
    )
    ax.set_xticks(range(len(keep_models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in keep_models], rotation=0, ha="center", fontsize=5.6)
    ax.set_yticks(range(len(keep_states)))
    ax.set_yticklabels([short_state(x) for x in reversed(keep_states)])
    ax.set_xlim(-0.75, len(keep_models) - 0.25)
    ax.set_ylim(-0.65, len(keep_states) - 0.35)
    ax.set_xlabel("Cox model")
    ax.set_ylabel("")
    style(ax)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.040, pad=0.030)
    cbar.set_label("")
    cbar.ax.set_title("log HR", fontsize=5.0, pad=2)
    cbar.ax.tick_params(labelsize=5.0, length=1.8, width=0.5, pad=1.0)
    cbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.39, right=0.80, bottom=0.21, top=0.94)
    save(fig, "fig5b_direct_nmf_model_sensitivity")


def fig5c_base_vs_full_hr() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")
    d = d[d["base_hr"].notna() & d["full_hr"].notna()].copy()
    fig, ax = square_fig()
    ax.scatter(
        np.log(d["base_hr"]),
        np.log(d["full_hr"]),
        s=32,
        color=MONO_COLOR,
        edgecolor="#222222",
        linewidth=0.25,
        alpha=0.84,
    )
    lim = max(abs(np.nanmin(np.log(d[["base_hr", "full_hr"]].values))), abs(np.nanmax(np.log(d[["base_hr", "full_hr"]].values)))) + 0.04
    ax.plot([-lim, lim], [-lim, lim], color="#888888", lw=0.65, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    offsets = {
        "T_NK | Proliferating T-cell (cell cycling)": (-0.040, 0.020),
        "epithelial | Cell cycling": (0.012, -0.024),
        "myeloid | SPP1+ macrophage": (0.010, 0.006),
        "myeloid | Mast": (0.010, -0.012),
    }
    for _, r in d[d["state_key"].isin(offsets)].iterrows():
        dx, dy = offsets[r["state_key"]]
        annotate_inside(
            ax,
            np.log(r["base_hr"]),
            np.log(r["full_hr"]),
            short_state(r["state_key"]),
            dx=dx,
            dy=dy,
            fontsize=4.5,
            ha="right" if dx < 0 else "left",
        )
    ax.set_xlabel("Base model log HR")
    ax.set_ylabel("Full model log HR")
    style(ax)
    fig.subplots_adjust(left=0.24, right=0.91, bottom=0.22, top=0.96)
    save(fig, "fig5c_base_vs_full_hr")


def fig5d_stage_beta_vs_hr() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")
    d = d[d["full_hr"].notna() & d["stage_beta_with_cancer_fixed_effect"].notna()].copy()
    fig, ax = square_fig()
    ax.axvline(0, color="#888888", lw=0.65, ls=":")
    ax.axhline(0, color="#888888", lw=0.65, ls="--")
    ax.scatter(
        d["stage_beta_with_cancer_fixed_effect"],
        np.log(d["full_hr"]),
        s=26 + 2.2 * d["cost_weighted_score_billion_usd"].fillna(0),
        color=MONO_COLOR,
        edgecolor="#222222",
        linewidth=0.25,
        alpha=0.85,
    )
    ax.set_xlim(d["stage_beta_with_cancer_fixed_effect"].min() - 0.030, d["stage_beta_with_cancer_fixed_effect"].max() + 0.065)
    ax.set_ylim(np.log(d["full_hr"]).min() - 0.045, np.log(d["full_hr"]).max() + 0.055)
    offsets = {
        "mesenchymal | Desmoplastic fibroblast": (-0.018, -0.026),
        "epithelial | Complete mesenchymal": (-0.010, 0.020),
        "myeloid | SPP1+ macrophage": (0.010, -0.010),
        "epithelial | Cell cycling": (-0.020, 0.014),
        "T_NK | Proliferating T-cell (cell cycling)": (-0.026, 0.036),
        "myeloid | Cell cycling": (-0.020, -0.026),
        "myeloid | Heat shock": (0.011, -0.020),
    }
    for _, r in d.sort_values("stage_beta_with_cancer_fixed_effect", ascending=False).head(7).iterrows():
        dx, dy = offsets.get(r["state_key"], (0.004, 0.0))
        annotate_inside(
            ax,
            r["stage_beta_with_cancer_fixed_effect"],
            np.log(r["full_hr"]),
            short_state(r["state_key"]),
            dx=dx,
            dy=dy,
            fontsize=4.0,
            ha="right" if dx < 0 else "left",
        )
    ax.set_xlabel("Stage beta\nwithin cancer")
    ax.set_ylabel("Direct signature log HR")
    style(ax)
    fig.subplots_adjust(left=0.24, right=0.91, bottom=0.22, top=0.96)
    save(fig, "fig5d_stage_beta_vs_hr")


def fig6a_major_daly() -> None:
    d = pd.read_csv(PROCESSED / "major_compartment_globocan_burden_scores.csv")
    d = d[
        d["source"].eq("WHO GHE 2021")
        & d["location"].eq("Global")
        & d["measure"].eq("daly")
        & ~d["compartment"].isin(["Other", "Non-malignant epithelial"])
    ].sort_values("weighted_score").tail(9)
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["weighted_score"] / 1e6, color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(d["compartment"])
    ax.set_xlabel("WHO DALY score\n(million)")
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.22, top=0.96)
    save(fig, "fig6a_major_daly")


def fig6b_state_daly() -> None:
    scores = pd.read_csv(PROCESSED / "nmf_state_globocan_burden_scores.csv")
    review = pd.read_csv(PROCESSED / "ecosystem_nmf_state_priority_review.csv")[["state_key", "coverage_ok", "likely_lineage_only"]]
    d = scores[
        scores["source"].eq("WHO GHE 2021")
        & scores["location"].eq("Global")
        & scores["measure"].eq("daly")
    ].merge(review, on="state_key", how="left")
    d = d[d["coverage_ok"].fillna(False) & ~d["likely_lineage_only"].fillna(False)].sort_values("weighted_score").tail(12)
    fig, ax = square_fig()
    y = np.arange(len(d))
    ax.barh(y, d["weighted_score"] / 1e6, color=MONO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels([short_state(x) for x in d["state_key"]])
    ax.set_xlabel("WHO DALY score\n(million)")
    style(ax)
    fig.subplots_adjust(left=0.38, right=0.96, bottom=0.22, top=0.96)
    save(fig, "fig6b_state_daly")


def fig6c_mortality_daly_rank() -> None:
    scores = pd.read_csv(PROCESSED / "nmf_state_globocan_burden_scores.csv")
    review = pd.read_csv(PROCESSED / "ecosystem_nmf_state_priority_review.csv")[["state_key", "coverage_ok", "likely_lineage_only"]]
    mort = scores[
        scores["source"].str.startswith("GLOBOCAN", na=False)
        & scores["location"].eq("Global")
        & scores["measure"].eq("mortality")
    ][["state_key", "rank_desc"]].rename(columns={"rank_desc": "mortality_rank"})
    daly = scores[
        scores["source"].eq("WHO GHE 2021")
        & scores["location"].eq("Global")
        & scores["measure"].eq("daly")
    ][["state_key", "rank_desc"]].rename(columns={"rank_desc": "daly_rank"})
    d = mort.merge(daly, on="state_key", how="inner").merge(review, on="state_key", how="left")
    d = d[d["coverage_ok"].fillna(False) & ~d["likely_lineage_only"].fillna(False)].copy()
    fig, ax = square_fig()
    ax.scatter(
        d["mortality_rank"],
        d["daly_rank"],
        s=20,
        color=MONO_COLOR,
        alpha=0.78,
        edgecolor="#222222",
        linewidth=0.25,
    )
    label_states = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "myeloid | C1QC+ macrophage",
        "myeloid | SPP1+ macrophage",
        "epithelial | Cell cycling",
        "mesenchymal | Desmoplastic fibroblast",
    ]
    lim = max(d["mortality_rank"].max(), d["daly_rank"].max()) + 3
    ax.plot([1, lim], [1, lim], color="#888888", lw=0.6, ls="--")
    ax.set_xlim(lim, -2)
    ax.set_ylim(lim, -2)
    offsets = {
        "T_NK | Proliferating T-cell (cell cycling)": (-2.8, 0.7),
        "myeloid | C1QC+ macrophage": (-3.0, 0.0),
        "myeloid | SPP1+ macrophage": (1.2, 0.0),
        "epithelial | Cell cycling": (1.2, 0.0),
        "mesenchymal | Desmoplastic fibroblast": (1.2, 0.0),
    }
    for _, r in d[d["state_key"].isin(label_states)].iterrows():
        dx, dy = offsets.get(r["state_key"], (1.0, 0.0))
        annotate_inside(
            ax,
            r["mortality_rank"],
            r["daly_rank"],
            short_state(r["state_key"]),
            dx=dx,
            dy=dy,
            fontsize=4.8,
            ha="right" if dx < 0 else "left",
        )
    ax.set_xlabel("GLOBOCAN mortality rank")
    ax.set_ylabel("WHO DALY rank")
    style(ax)
    fig.subplots_adjust(left=0.21, right=0.94, bottom=0.21, top=0.96)
    save(fig, "fig6c_mortality_daly_rank")


def fig6d_cost_daly_rank() -> None:
    scores = pd.read_csv(PROCESSED / "nmf_state_globocan_burden_scores.csv")
    review = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")[
        ["state_key", "rank_desc_cost", "likely_lineage_only", "full_hr", "cost_prognosis_class"]
    ]
    daly = scores[
        scores["source"].eq("WHO GHE 2021")
        & scores["location"].eq("Global")
        & scores["measure"].eq("daly")
    ][["state_key", "rank_desc"]].rename(columns={"rank_desc": "daly_rank"})
    d = daly.merge(review, on="state_key", how="inner")
    d = d[~d["likely_lineage_only"].fillna(False)].copy()
    fig, ax = square_fig()
    ax.scatter(
        d["daly_rank"],
        d["rank_desc_cost"],
        s=22,
        color=MONO_COLOR,
        edgecolor="#222222",
        linewidth=0.25,
        alpha=0.82,
    )
    label_states = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "T_NK | Exhausted CD8+ T-cell",
        "epithelial | Complete mesenchymal",
        "myeloid | Heat shock",
        "myeloid | C1QC+ macrophage",
    ]
    lim = max(d["daly_rank"].max(), d["rank_desc_cost"].max()) + 3
    ax.plot([1, lim], [1, lim], color="#888888", lw=0.6, ls="--")
    ax.set_xlim(lim, 0)
    ax.set_ylim(lim, 0)
    for _, r in d[d["state_key"].isin(label_states)].iterrows():
        annotate_inside(ax, r["daly_rank"], r["rank_desc_cost"], short_state(r["state_key"]), dx=-2.5, fontsize=4.8)
    ax.set_xlabel("WHO DALY rank")
    ax.set_ylabel("NCI cost rank")
    style(ax)
    fig.subplots_adjust(left=0.21, right=0.94, bottom=0.21, top=0.96)
    save(fig, "fig6d_cost_daly_rank")


def fig6e_low_cost_status() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")
    d = d[~d["likely_lineage_only"].fillna(False)].copy()
    d["cost_bin"] = np.select([d["high_cost"], d["low_cost"]], ["High cost", "Low cost"], default="Mid cost")
    d["prognosis"] = np.select([d["adverse"], d["protective"]], ["Adverse", "Protective"], default="Neutral/untested")
    tab = d.groupby(["cost_bin", "prognosis"], observed=True).size().unstack(fill_value=0)
    tab = tab.reindex(["Low cost", "Mid cost", "High cost"]).fillna(0)
    for col in ["Adverse", "Protective", "Neutral/untested"]:
        if col not in tab:
            tab[col] = 0
    fig, ax = square_fig()
    bottom = np.zeros(len(tab))
    colors = {"Adverse": "#B23A32", "Protective": "#4C6FAE", "Neutral/untested": "#BDBDBD"}
    for col in ["Adverse", "Protective", "Neutral/untested"]:
        ax.bar(np.arange(len(tab)), tab[col].values, bottom=bottom, color=colors[col], label=col)
        bottom += tab[col].values
    ax.set_xticks(np.arange(len(tab)))
    ax.set_xticklabels(tab.index, rotation=20, ha="right")
    ax.set_ylabel("NMF states")
    ax.set_xlabel("Modeled cost bin")
    style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=4.9)
    fig.subplots_adjust(left=0.21, right=0.96, bottom=0.27, top=0.96)
    save(fig, "fig6e_low_cost_status")


def fig6f_rank_concordance() -> None:
    scores = pd.read_csv(PROCESSED / "nmf_state_globocan_burden_scores.csv")
    review = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")[
        ["state_key", "rank_desc_cost", "likely_lineage_only"]
    ]
    mort = scores[
        scores["source"].str.startswith("GLOBOCAN", na=False)
        & scores["location"].eq("Global")
        & scores["measure"].eq("mortality")
    ][["state_key", "rank_desc"]].rename(columns={"rank_desc": "mortality_rank"})
    daly = scores[
        scores["source"].eq("WHO GHE 2021")
        & scores["location"].eq("Global")
        & scores["measure"].eq("daly")
    ][["state_key", "rank_desc"]].rename(columns={"rank_desc": "daly_rank"})
    d = mort.merge(daly, on="state_key").merge(review, on="state_key")
    d = d[~d["likely_lineage_only"].fillna(False)].copy()
    pairs = [
        ("Mortality-DALY", "mortality_rank", "daly_rank"),
        ("Mortality-cost", "mortality_rank", "rank_desc_cost"),
        ("DALY-cost", "daly_rank", "rank_desc_cost"),
    ]
    vals = []
    for label, a, b in pairs:
        vals.append((label, d[[a, b]].corr(method="spearman").iloc[0, 1]))
    fig, ax = square_fig()
    y = np.arange(len(vals))
    ax.barh(y, [v for _, v in vals], color=MONO_COLOR)
    for i, (_, v) in enumerate(vals):
        ax.text(min(v - 0.020, 1.03), i, f"{v:.2f}", va="center", fontsize=5.5, ha="right", color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([x for x, _ in vals])
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("Spearman rank correlation")
    style(ax)
    fig.subplots_adjust(left=0.40, right=0.93, bottom=0.20, top=0.96)
    save(fig, "fig6f_rank_concordance")


def main() -> None:
    setup()
    fig1a_design()
    fig1b_coverage()
    fig1c_data_layers()
    fig1d_epithelial_umap()
    fig1e_tnk_umap()
    fig1f_myeloid_umap()
    fig1g_epithelial_dotplot()
    fig1h_tnk_dotplot()
    fig1i_myeloid_dotplot()
    fig1d_interpretation_scheme()
    fig2a_major_mortality()
    fig2b_major_cost()
    fig2c_tcga_proxy()
    fig2d_major_cost_mortality()
    fig2e_major_cost_proxy_hr()
    fig2f_major_variance_cost()
    fig3a_state_raw()
    fig3b_state_adverse()
    fig3c_raw_vs_adverse()
    fig3d_cost_prognosis_quadrant()
    fig3e_cost_burden_hr_bubble()
    fig3f_cost_class_counts()
    fig4a_comp_variance()
    fig4b_state_variance()
    fig4c_leave_one()
    fig4d_stage_gradient_rank()
    fig4e_stage_trajectories()
    fig4f_state_axis_stage()
    fig5a_direct_cox_forest()
    fig5b_direct_cox_models()
    fig5c_base_vs_full_hr()
    fig5d_stage_beta_vs_hr()
    fig6a_major_daly()
    fig6b_state_daly()
    fig6c_mortality_daly_rank()
    fig6d_cost_daly_rank()
    fig6e_low_cost_status()
    fig6f_rank_concordance()
    print(f"wrote subfigures to {SUB}")


if __name__ == "__main__":
    main()
