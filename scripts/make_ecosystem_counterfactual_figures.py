#!/usr/bin/env python3
"""Generate Figure 7 counterfactual validation panels and LaTeX PDF."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "pancancer_ecosystem"
FIG = ROOT / "figures" / "pancancer_ecosystem_counterfactual"
SUB = FIG / "subfigures"

NMF_COLORS = {
    "T_NK": "#2A9D8F",
    "myeloid": "#B23A32",
    "B_plasma": "#7A5EA8",
    "mesenchymal": "#B8872C",
    "epithelial": "#4C6FAE",
}

CLASS_COLORS = {
    "stage-increasing high-cost adverse": "#B23A32",
    "high cost + protective": "#4C6FAE",
    "stage-decreasing protective": "#4C6FAE",
    "high cost + neutral": "#8C8C8C",
    "mid cost / mixed": "#C6A15B",
    "low cost + neutral": "#D7D7D7",
}

PRIORITY_STATES = [
    "T_NK | Proliferating T-cell (cell cycling)",
    "epithelial | Cell cycling",
    "myeloid | Cell cycling",
    "myeloid | SPP1+ macrophage",
    "mesenchymal | Desmoplastic fibroblast",
    "myeloid | Heat shock",
    "epithelial | Complete mesenchymal",
    "myeloid | NLRP3+ monocyte derived macrophage",
]

MONO_COLOR = "#4E6E8E"
MONO_DARK = "#2F4858"
MONO_LIGHT = "#D7E0E7"
SCHEME_FILL = "#F5F6F7"
OBSERVED_COLOR = "#B23A32"


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
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 5.5,
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


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(SUB / f"{name}.{ext}")
    plt.close(fig)


def annotate_inside(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    dx: float = 0,
    dy: float = 0,
    fontsize: float = 4.8,
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
    ax.text(xx, yy, label, fontsize=fontsize, ha=ha, va="center", clip_on=True)


def short_state(x: str) -> str:
    x = str(x)
    comp = x.split(" | ", 1)[0] if " | " in x else ""
    state = x.split(" | ", 1)[-1]
    repl = {
        "Proliferating T-cell (cell cycling)": "Prolif. T",
        "Exhausted CD8+ T-cell": "Exh. CD8",
        "CD16+ NK-cell": "CD16 NK",
        "Complete mesenchymal": "Epi mes.",
        "Cell cycling": "Cycling",
        "SPP1+ macrophage": "SPP1 mac",
        "NLRP3+ monocyte derived macrophage": "NLRP3 mono-mac",
        "Desmoplastic fibroblast": "Desmo fibro.",
        "Heat shock": "Heat shock",
        "Treg": "Treg",
        "Tfh": "Tfh",
        "Breast ductal": "Breast ductal",
        "Tissue resident memory T-cell": "TRM T",
    }
    label = repl.get(state, state[:20])
    if label == "Cycling":
        if comp == "epithelial":
            return "Epi cycling"
        if comp == "myeloid":
            return "Mye cycling"
        if comp == "B_plasma":
            return "B cycling"
    if label == "Heat shock" and comp == "myeloid":
        return "Mye heat shock"
    return label


def nmf_comp(state: str) -> str:
    return str(state).split(" | ", 1)[0]


def svg_box(x: float, y: float, w: float, h: float, text: str, fill: str = SCHEME_FILL) -> str:
    lines = text.split("\n")
    line_h = 13
    total_h = (len(lines) - 1) * line_h
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="#222222" stroke-width="1.1"/>']
    for i, line in enumerate(lines):
        yy = y + h / 2 - total_h / 2 + i * line_h + 3
        parts.append(f'<text x="{x + w / 2}" y="{yy}" text-anchor="middle" font-size="10.0">{escape(line)}</text>')
    return "\n".join(parts)


def svg_arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#222222" stroke-width="1.2" marker-end="url(#arrow)"/>'


def svg_template(body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="183.6pt" height="183.6pt" viewBox="0 0 255 255">
<defs>
  <marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="#222222"/>
  </marker>
  <style>
    text {{ font-family: DejaVu Sans, Arial, sans-serif; }}
    .title {{ font-size: 11.2px; font-weight: 600; fill: #222222; }}
    .small {{ font-size: 8.2px; fill: #555555; }}
  </style>
</defs>
<rect x="0" y="0" width="255" height="255" fill="white"/>
{body}
</svg>'''


def write_svg(name: str, svg: str) -> None:
    svg_path = SUB / f"{name}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(SUB / f"{name}.pdf"))
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(SUB / f"{name}.png"), output_width=1530, output_height=1530)


def fig7a_design() -> None:
    body = [
        '<text class="title" x="127.5" y="21" text-anchor="middle">Counterfactual validation layer</text>',
        svg_box(18, 44, 92, 38, "Observed\nstate score"),
        svg_box(145, 44, 92, 38, "Cancer-site\ncomposition test"),
        svg_box(18, 116, 92, 42, "Within-cancer\nlow-state baseline\nq25"),
        svg_box(145, 116, 92, 42, "TCGA risk\nPAF support"),
        svg_box(61, 191, 133, 40, "Validated modeled\nattributable burden"),
        svg_arrow(110, 63, 145, 63),
        svg_arrow(64, 82, 64, 116),
        svg_arrow(191, 82, 191, 116),
        svg_arrow(76, 158, 100, 191),
        svg_arrow(190, 158, 155, 191),
        '<text class="small" x="127.5" y="101" text-anchor="middle">hold cancer type and public weights fixed</text>',
        '<text class="small" x="127.5" y="176" text-anchor="middle">negative controls: label and weight permutations</text>',
    ]
    write_svg("fig7a_counterfactual_design", svg_template("\n".join(body)))


def fig7b_composition_rank() -> None:
    d = pd.read_csv(PROCESSED / "ecosystem_state_composition_balanced_scores.csv")
    d = d[d["endpoint"].eq("global_mortality") & ~d["likely_lineage_only"].fillna(False)].copy()
    rho, p = stats.spearmanr(d["observed_score_rank"], d["equal_site_score_rank"], nan_policy="omit")
    fig, ax = square_fig()
    colors = d["composition_class"].map(
        {
            "composition-robust": "#2A9D8F",
            "site-driven": "#B23A32",
            "biology-recurring": "#4C6FAE",
            "mixed": "#BDBDBD",
        }
    ).fillna("#BDBDBD")
    ax.scatter(d["observed_score_rank"], d["equal_site_score_rank"], s=21, c=colors, edgecolor="#222222", linewidth=0.25, alpha=0.84)
    lim = max(d["observed_score_rank"].max(), d["equal_site_score_rank"].max()) + 3
    ax.plot([1, lim], [1, lim], color="#888888", lw=0.6, ls="--")
    ax.set_xlim(lim, 0)
    ax.set_ylim(lim, 0)
    label_states = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "epithelial | Cell cycling",
        "myeloid | SPP1+ macrophage",
        "mesenchymal | Desmoplastic fibroblast",
    ]
    offsets = {
        "T_NK | Proliferating T-cell (cell cycling)": (11.0, -2.4),
        "epithelial | Cell cycling": (9.0, -3.8),
        "myeloid | Cell cycling": (-6.5, -5.5),
        "myeloid | SPP1+ macrophage": (9.0, 4.0),
        "mesenchymal | Desmoplastic fibroblast": (9.0, 2.0),
        "epithelial | Complete mesenchymal": (-5.0, -3.5),
    }
    for _, r in d[d["state_key"].isin(label_states)].iterrows():
        dx, dy = offsets.get(r["state_key"], (-2.8, 0.0))
        annotate_inside(ax, r["observed_score_rank"], r["equal_site_score_rank"], short_state(r["state_key"]), dx=dx, dy=dy, fontsize=4.3)
    ax.text(0.04, 0.96, f"rho={rho:.2f}\np={p:.1g}", transform=ax.transAxes, ha="left", va="top", fontsize=5.2)
    ax.set_xlabel("Observed mortality rank")
    ax.set_ylabel("Equal-site rank")
    style(ax)
    add_point_legend(
        ax,
        [
            ("Robust", "#2A9D8F"),
            ("Site-driven", "#B23A32"),
            ("Recurring", "#4C6FAE"),
            ("Mixed", "#BDBDBD"),
        ],
        loc="lower right",
        fontsize=4.15,
    )
    fig.subplots_adjust(left=0.21, right=0.94, bottom=0.21, top=0.96)
    save(fig, "fig7b_composition_rank")


def merged_counterfactual() -> pd.DataFrame:
    cf = pd.read_csv(PROCESSED / "ecosystem_state_within_cancer_counterfactual_scores.csv")
    paf = pd.read_csv(PROCESSED / "ecosystem_state_tcga_paf_counterfactual.csv")
    prog = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")
    d = cf.merge(
        paf[["state_key", "mean_paf_adverse_only", "paf_burden_mortality_score", "paf_cost_score_billion_usd"]],
        on="state_key",
        how="left",
    ).merge(
        prog[["state_key", "progression_cost_prognosis_class", "full_hr", "stage_beta_with_cancer_fixed_effect", "likely_lineage_only"]],
        on="state_key",
        how="left",
    )
    d["mean_paf_adverse_only"] = d["mean_paf_adverse_only"].fillna(0)
    d["integrated_adverse_counterfactual_score"] = d["reducible_global_mortality_score"] * d["mean_paf_adverse_only"]
    d["short_label"] = d["state_key"].map(short_state)
    d["plot_color"] = d["progression_cost_prognosis_class"].map(CLASS_COLORS).fillna(d["state_key"].map(lambda x: NMF_COLORS.get(nmf_comp(x), "#8C8C8C")))
    return d


def fig7c_reducible_cost() -> None:
    d = merged_counterfactual()
    show = d[~d["likely_lineage_only"].fillna(False)].sort_values("reducible_nci_cost_score_billion_usd").tail(10)
    fig, ax = square_fig()
    y = np.arange(len(show))
    ax.barh(y, show["reducible_nci_cost_score_billion_usd"], color=MONO_COLOR, alpha=0.94)
    ax.set_yticks(y)
    ax.set_yticklabels(show["short_label"])
    ax.set_xlabel("Within-cancer reducible\nmodeled cost score (B USD)")
    ax.set_xlim(0, show["reducible_nci_cost_score_billion_usd"].max() * 1.16)
    for i, r in enumerate(show.itertuples(index=False)):
        if getattr(r, "mean_paf_adverse_only", 0) > 0:
            ax.text(r.reducible_nci_cost_score_billion_usd + 0.25, i, "PAF+", va="center", fontsize=4.6, color=MONO_DARK)
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.94, bottom=0.22, top=0.96)
    save(fig, "fig7c_reducible_cost")


def fig7d_integrated_adverse() -> None:
    d = merged_counterfactual()
    show = d.sort_values("integrated_adverse_counterfactual_score").tail(9)
    fig, ax = square_fig()
    y = np.arange(len(show))
    ax.barh(y, show["integrated_adverse_counterfactual_score"] / 1000, color=MONO_COLOR, alpha=0.94)
    ax.set_yticks(y)
    ax.set_yticklabels(show["short_label"])
    ax.set_xlabel("Reducible mortality x\nTCGA adverse PAF (thousand)")
    ax.set_xlim(0, show["integrated_adverse_counterfactual_score"].max() / 1000 * 1.12)
    style(ax)
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.23, top=0.96)
    save(fig, "fig7d_integrated_adverse")


def fig7e_phase_validation() -> None:
    phase = pd.read_csv(PROCESSED / "ecosystem_state_phase_of_care_cost_scores.csv")
    prog = pd.read_csv(PROCESSED / "ecosystem_state_progression_cost_prognosis_map.csv")
    d = phase.merge(prog[["state_key", "progression_cost_prognosis_class"]], on="state_key", how="left")
    keep = [
        "epithelial | Cell cycling",
        "T_NK | Exhausted CD8+ T-cell",
        "myeloid | Mast",
        "mesenchymal | Desmoplastic fibroblast",
        "epithelial | Complete mesenchymal",
        "myeloid | SPP1+ macrophage",
        "myeloid | Heat shock",
        "T_NK | Proliferating T-cell (cell cycling)",
        "myeloid | Cell cycling",
    ]
    show = d[d["state_key"].isin(keep)].copy().sort_values("terminal_to_continuing_enrichment")
    fig, ax = square_fig()
    y = np.arange(len(show))
    ax.barh(y, show["terminal_to_continuing_enrichment"], color=MONO_COLOR)
    ax.axvline(d["terminal_to_continuing_enrichment"].median(), color="#888888", lw=0.65, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([short_state(x) for x in show["state_key"]])
    ax.set_xlabel("Last-year / continuing\nphase-cost enrichment")
    ax.set_xlim(0, max(16, show["terminal_to_continuing_enrichment"].max() * 1.12))
    style(ax)
    fig.subplots_adjust(left=0.40, right=0.96, bottom=0.23, top=0.96)
    save(fig, "fig7e_phase_validation")


def fig7f_negative_control() -> None:
    null = pd.read_csv(PROCESSED / "ecosystem_state_negative_control_null_distribution.csv")
    summary = pd.read_csv(PROCESSED / "ecosystem_state_negative_control_summary.csv")
    keep = [
        "T_NK | Proliferating T-cell (cell cycling)",
        "epithelial | Complete mesenchymal",
        "myeloid | SPP1+ macrophage",
        "mesenchymal | Desmoplastic fibroblast",
    ]
    null = null[null["control_type"].eq("cancer_weight_permutation") & null["state_key"].isin(keep)].copy()
    summary = summary[summary["control_type"].eq("cancer_weight_permutation") & summary["state_key"].isin(keep)].copy()
    order = keep[::-1]
    fig, ax = square_fig()
    data = [null[null["state_key"].eq(s)]["null_score"].to_numpy() / 1000 for s in order]
    parts = ax.violinplot(data, positions=np.arange(len(order)), vert=False, widths=0.62, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(MONO_LIGHT)
        pc.set_edgecolor("#888888")
        pc.set_alpha(0.95)
    obs_map = summary.set_index("state_key")["observed_score"].to_dict()
    p_map = summary.set_index("state_key")["empirical_p_ge_observed"].to_dict()
    for i, state in enumerate(order):
        obs = obs_map.get(state, np.nan) / 1000
        ax.scatter(obs, i, s=20, color=OBSERVED_COLOR, edgecolor="#222222", linewidth=0.25, zorder=3)
        p = p_map.get(state, np.nan)
        ax.text(ax.get_xlim()[1] * 0.97, i, f"p={p:.2f}", ha="right", va="center", fontsize=4.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([short_state(x) for x in order])
    ax.set_xlabel("Cancer-weight permutation\nnull score (thousand)")
    style(ax)
    add_point_legend(
        ax,
        [("Null distribution", MONO_LIGHT), ("Observed", OBSERVED_COLOR)],
        loc="lower right",
        fontsize=4.35,
    )
    fig.subplots_adjust(left=0.39, right=0.94, bottom=0.23, top=0.96)
    save(fig, "fig7f_negative_control")


def write_latex() -> None:
    tex = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=12mm]{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\definecolor{titlegray}{HTML}{222222}
\newcommand{\figuretitle}[2]{{\large\bfseries\color{titlegray}#1}\hfill{\bfseries #2}\par\vspace{2mm}\hrule\vspace{5mm}}
\newcommand{\panel}[3]{\begin{minipage}[t]{#1\linewidth}{\bfseries #2}\par\vspace{1mm}\includegraphics[width=\linewidth]{#3}\end{minipage}}
\newcommand{\legendtext}[1]{\vfill{\footnotesize #1}}
\begin{document}
\figuretitle{Figure 7}{Counterfactual validation of ecosystem burden}
\vspace{3mm}
\panel{0.31}{a}{subfigures/fig7a_counterfactual_design.pdf}
\hfill
\panel{0.31}{b}{subfigures/fig7b_composition_rank.pdf}
\hfill
\panel{0.31}{c}{subfigures/fig7c_reducible_cost.pdf}
\par\vspace{4mm}
\panel{0.31}{d}{subfigures/fig7d_integrated_adverse.pdf}
\hfill
\panel{0.31}{e}{subfigures/fig7e_phase_validation.pdf}
\hfill
\panel{0.31}{f}{subfigures/fig7f_negative_control.pdf}
\legendtext{\textbf{Figure 7.} Counterfactual and negative-control validation of modeled ecosystem burden. \textbf{a,} Framework for holding cancer type and public burden/cost weights fixed while comparing observed state representation with a within-cancer low-state baseline. \textbf{b,} Observed global-mortality rank versus equal-site rank, testing whether state priorities are dominated by cancer-site composition. \textbf{c,} Modeled reducible NCI cost score after shifting each cancer type to a low-state baseline; PAF+ marks states with adverse TCGA risk-counterfactual support. \textbf{d,} Integrated adverse counterfactual score combining within-cancer reducible mortality representation with TCGA adverse PAF. \textbf{e,} Phase-of-care validation using NCI per-patient annualized costs; bars show last-year-of-life to continuing-care enrichment. \textbf{f,} Cancer-weight permutation negative control for selected priority states; gray distributions are null scores and colored dots are observed scores. These are modeled validation quantities, not observed cell-state spending or causal treatment effects.}
\end{document}
"""
    (FIG / "ecosystem_counterfactual_figure7.tex").write_text(tex, encoding="utf-8")


def write_legends() -> None:
    text = """# Counterfactual validation figure legend

## Figure 7. Counterfactual validation of ecosystem burden

Counterfactual and negative-control validation of modeled ecosystem burden. **a,** Framework for holding cancer type and public burden/cost weights fixed while comparing observed state representation with a within-cancer low-state baseline. **b,** Observed global-mortality rank versus equal-site rank, testing whether state priorities are dominated by cancer-site composition. **c,** Modeled reducible NCI cost score after shifting each cancer type to a low-state baseline; PAF+ marks states with adverse TCGA risk-counterfactual support. **d,** Integrated adverse counterfactual score combining within-cancer reducible mortality representation with TCGA adverse PAF. **e,** Phase-of-care validation using NCI per-patient annualized costs; bars show last-year-of-life to continuing-care enrichment. **f,** Cancer-weight permutation negative control for selected priority states; gray distributions are null scores and colored dots are observed scores.

## Statistical notes

Within-cancer counterfactual scores use the 25th percentile of sample-level all-tumor state representation within each cancer type as the low-state baseline. TCGA PAF scores use existing cancer-stratified Cox coefficients adjusted for age, sex, stage and tumor purity where available, with within-cancer top-quartile/IQR contrasts. Phase-of-care scores use NCI per-patient annualized medical-service plus oral-prescription phase costs, so phase enrichment is a validation signal rather than a national aggregate cost estimate. Permutation controls preserve state representation and shuffle cancer burden weights. All cost and attribution statements are modeled/inferred, not observed patient-level cell-state spending.
"""
    (FIG / "ecosystem_counterfactual_figure_legends.md").write_text(text, encoding="utf-8")


def make_contact_sheet() -> None:
    paths = sorted(SUB.glob("*.png"))
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((360, 360))
        canvas = Image.new("RGB", (360, 392), "white")
        canvas.paste(im, (0, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 6), p.stem, fill=(0, 0, 0))
        draw.rectangle((0, 32, 359, 391), outline=(210, 0, 0), width=1)
        imgs.append(canvas)
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    out = Image.new("RGB", (cols * 360, rows * 392), "white")
    for i, im in enumerate(imgs):
        out.paste(im, ((i % cols) * 360, (i // cols) * 392))
    out.save(FIG / "qa_counterfactual_subfigures.png")


def scan_edges(strip: int = 18, threshold: int = 80) -> list[tuple[str, dict[str, int]]]:
    hits = []
    for p in sorted(SUB.glob("*.png")):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        strips = {
            "left": im.crop((0, 0, strip, h)),
            "right": im.crop((w - strip, 0, w, h)),
            "top": im.crop((0, 0, w, strip)),
            "bottom": im.crop((0, h - strip, w, h)),
        }
        counts = {k: sum(1 for r, g, b in s.getdata() if min(r, g, b) < 245) for k, s in strips.items()}
        if any(v > threshold for v in counts.values()):
            hits.append((p.name, counts))
    return hits


def main() -> None:
    setup()
    fig7a_design()
    fig7b_composition_rank()
    fig7c_reducible_cost()
    fig7d_integrated_adverse()
    fig7e_phase_validation()
    fig7f_negative_control()
    write_latex()
    write_legends()
    make_contact_sheet()
    hits = scan_edges()
    if hits:
        for name, counts in hits:
            print(f"edge-check {name}: {counts}")
    print(f"wrote counterfactual subfigures to {SUB}")
    print(f"wrote {FIG / 'ecosystem_counterfactual_figure7.tex'}")


if __name__ == "__main__":
    main()
