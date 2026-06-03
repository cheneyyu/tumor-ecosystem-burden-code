#!/usr/bin/env python3
"""Create signal-review tables for the pan-cancer ecosystem burden analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "pancancer_ecosystem"


MAJOR_PROXY = {
    "Macrophage": "macrophage_total",
    "Mast": "Mast.cells.activated",
    "DC/pDC": "Dendritic.cells.activated",
    "NK cell": "NK.cells.activated",
    "B cell": "B.cells.memory",
    "Plasma cell": "Plasma.cells",
    "T cell": "T.cells.CD8",
}


def state_proxy(state_key: str) -> str | None:
    comp, _, state = state_key.partition(" | ")
    lo = state.lower()
    if comp == "myeloid":
        if "mast" in lo:
            return "Mast.cells.activated"
        if "neutrophil" in lo:
            return "Neutrophils"
        if "dc" in lo or "dendritic" in lo or "langerhans" in lo or "pdc" in lo:
            return "Dendritic.cells.activated"
        if "monocyte" in lo:
            return "Monocytes"
        if "macrophage" in lo or "phagocyte" in lo or "cell cycling" in lo or "heat shock" in lo:
            return "macrophage_total"
    if comp == "T_NK":
        if "treg" in lo:
            return "T.cells.regulatory..Tregs."
        if "tfh" in lo:
            return "T.cells.follicular.helper"
        if "nk" in lo:
            return "NK.cells.activated"
        if "cd8" in lo or "resident" in lo or "exhausted" in lo or "intraepithelial" in lo:
            return "T.cells.CD8"
        if "th17" in lo or "ilc3" in lo:
            return "T.cells.CD4.memory.activated"
    if comp == "B_plasma":
        if "plasma" in lo:
            return "Plasma.cells"
        if "b-cell" in lo or "b cell" in lo or "germinal" in lo or "mature" in lo:
            return "B.cells.memory"
    return None


def top_contributions(terms: pd.DataFrame, feature_col: str, score_col: str, label_col: str = "globocan_label") -> pd.DataFrame:
    rows = []
    for feature, sub in terms.groupby(feature_col, observed=True):
        total = sub[score_col].sum()
        if not np.isfinite(total) or total <= 0:
            continue
        by_cancer = sub.groupby(label_col, observed=True)[score_col].sum().sort_values(ascending=False)
        top = by_cancer.head(3)
        rows.append(
            {
                feature_col: feature,
                "top_cancer": top.index[0],
                "top_cancer_pct": float(top.iloc[0] / total),
                "top3_cancers": "; ".join([f"{k} ({v / total:.0%})" for k, v in top.items()]),
            }
        )
    return pd.DataFrame(rows)


def make_major_review(cox: pd.DataFrame) -> pd.DataFrame:
    burden = pd.read_csv(OUT / "major_compartment_globocan_burden_scores.csv")
    burden = burden[(burden["location"].eq("Global")) & (burden["measure"].eq("mortality"))].copy()
    cost = pd.read_csv(OUT / "major_compartment_nci_cost_scores.csv")
    var = pd.read_csv(OUT / "major_compartment_variance_decomposition.csv")
    terms = pd.read_csv(OUT / "major_compartment_globocan_burden_terms.csv")
    terms = terms[(terms["location"].eq("Global")) & (terms["measure"].eq("mortality"))].copy()
    contrib = top_contributions(terms, "compartment", "weighted_representation")

    out = burden.merge(cost, on="compartment", how="left", suffixes=("", "_cost"))
    out = out.merge(var, on="compartment", how="left")
    out = out.merge(contrib, on="compartment", how="left")
    out["tcga_proxy"] = out["compartment"].map(MAJOR_PROXY)
    out = out.merge(
        cox[
            [
                "cell_type",
                "coef_log_hr_per_within_cancer_sd",
                "hr_per_within_cancer_sd",
                "ci95_low",
                "ci95_high",
                "p",
            ]
        ],
        left_on="tcga_proxy",
        right_on="cell_type",
        how="left",
    )
    out["adverse_loghr_sig"] = np.where(
        (out["coef_log_hr_per_within_cancer_sd"] > 0) & (out["p"] < 0.05),
        out["coef_log_hr_per_within_cancer_sd"],
        0.0,
    )
    out["prognosis_weighted_global_mortality_score"] = out["weighted_score"] * out["adverse_loghr_sig"]
    out["prognosis_weighted_cost_score"] = out["cost_weighted_score_billion_usd"] * out["adverse_loghr_sig"]
    out["coverage_ok"] = out["n_mapped_cancers"] >= 10
    out["not_single_cancer"] = out["top_cancer_pct"] <= 0.35
    out["within_cancer_supported"] = out["within_cancer_variance_share"] >= 0.5
    out.to_csv(OUT / "ecosystem_major_compartment_priority_review.csv", index=False)
    return out


def make_state_review(cox: pd.DataFrame) -> pd.DataFrame:
    burden = pd.read_csv(OUT / "nmf_state_globocan_burden_scores.csv")
    burden = burden[(burden["location"].eq("Global")) & (burden["measure"].eq("mortality"))].copy()
    cost = pd.read_csv(OUT / "nmf_state_nci_cost_scores.csv")
    var = pd.read_csv(OUT / "nmf_state_variance_decomposition.csv")
    cancer = pd.read_csv(OUT / "nmf_state_representation_by_cancer.csv")
    state_meta = cancer[["state_key", "nmf_compartment", "Cell_state", "state_category"]].drop_duplicates("state_key")
    terms = pd.read_csv(OUT / "nmf_state_globocan_burden_terms.csv")
    terms = terms[(terms["location"].eq("Global")) & (terms["measure"].eq("mortality"))].copy()
    contrib = top_contributions(terms, "state_key", "weighted_representation")

    out = burden.merge(cost, on="state_key", how="left", suffixes=("", "_cost"))
    out = out.merge(var, on="state_key", how="left")
    out = out.merge(state_meta, on="state_key", how="left")
    out = out.merge(contrib, on="state_key", how="left")
    out["tcga_proxy"] = out["state_key"].map(state_proxy)
    out = out.merge(
        cox[
            [
                "cell_type",
                "coef_log_hr_per_within_cancer_sd",
                "hr_per_within_cancer_sd",
                "ci95_low",
                "ci95_high",
                "p",
            ]
        ],
        left_on="tcga_proxy",
        right_on="cell_type",
        how="left",
    )
    out["adverse_loghr_sig"] = np.where(
        (out["coef_log_hr_per_within_cancer_sd"] > 0) & (out["p"] < 0.05),
        out["coef_log_hr_per_within_cancer_sd"],
        0.0,
    )
    out["prognosis_weighted_global_mortality_score"] = out["weighted_score"] * out["adverse_loghr_sig"]
    out["prognosis_weighted_cost_score"] = out["cost_weighted_score_billion_usd"] * out["adverse_loghr_sig"]
    direct_path = OUT / "tcga_nmf_state_signature_stratified_cox.csv"
    if direct_path.exists():
        direct = pd.read_csv(direct_path)
        direct = direct[direct["model"].eq("stratified_by_tcga_cancer__age_sex_stage_purity_available")].copy()
        direct = direct.rename(
            columns={
                "coef_log_hr_per_within_cancer_sd": "direct_coef_log_hr_per_within_cancer_sd",
                "hr_per_within_cancer_sd": "direct_hr_per_within_cancer_sd",
                "ci95_low": "direct_ci95_low",
                "ci95_high": "direct_ci95_high",
                "p": "direct_p",
                "n_samples": "direct_n_samples",
                "n_events": "direct_n_events",
                "n_cancer_types": "direct_n_cancer_types",
            }
        )
        out = out.merge(
            direct[
                [
                    "state_key",
                    "direct_n_samples",
                    "direct_n_events",
                    "direct_n_cancer_types",
                    "direct_coef_log_hr_per_within_cancer_sd",
                    "direct_hr_per_within_cancer_sd",
                    "direct_ci95_low",
                    "direct_ci95_high",
                    "direct_p",
                ]
            ],
            on="state_key",
            how="left",
        )
    else:
        out["direct_coef_log_hr_per_within_cancer_sd"] = np.nan
        out["direct_hr_per_within_cancer_sd"] = np.nan
        out["direct_p"] = np.nan
    out["direct_adverse_loghr_sig"] = np.where(
        (out["direct_coef_log_hr_per_within_cancer_sd"] > 0) & (out["direct_p"] < 0.05),
        out["direct_coef_log_hr_per_within_cancer_sd"],
        0.0,
    )
    out["direct_prognosis_weighted_global_mortality_score"] = out["weighted_score"] * out["direct_adverse_loghr_sig"]
    out["direct_prognosis_weighted_cost_score"] = out["cost_weighted_score_billion_usd"] * out["direct_adverse_loghr_sig"]
    out["coverage_ok"] = out["n_mapped_cancers"] >= 10
    out["not_single_cancer"] = out["top_cancer_pct"] <= 0.35
    out["within_cancer_supported"] = out["within_cancer_variance_share"] >= 0.5
    out["likely_lineage_only"] = out["state_category"].astype(str).eq("Epithelial: lineage/tumor")
    out.loc[out["state_key"].eq("epithelial | Complete mesenchymal"), "likely_lineage_only"] = False
    out.to_csv(OUT / "ecosystem_nmf_state_priority_review.csv", index=False)
    return out


def fmt_rows(df: pd.DataFrame, label_col: str, value_col: str, n: int = 8) -> list[str]:
    rows = []
    for r in df.sort_values(value_col, ascending=False).head(n).itertuples(index=False):
        rows.append(
            f"- {getattr(r, label_col)}: {getattr(r, value_col):,.1f}; "
            f"direct HR={getattr(r, 'direct_hr_per_within_cancer_sd', np.nan):.2f}; "
            f"proxy={getattr(r, 'tcga_proxy', '')}; "
            f"top cancers={getattr(r, 'top3_cancers', '')}"
        )
    return rows


def fmt_burden_rows(df: pd.DataFrame, label_col: str, n: int = 6) -> list[str]:
    rows = []
    for r in df.sort_values("weighted_score", ascending=False).head(n).itertuples(index=False):
        rows.append(f"- {getattr(r, label_col)}: {r.weighted_score:,.1f}; mapped cancers={r.n_mapped_cancers}")
    return rows


def write_review(major: pd.DataFrame, states: pd.DataFrame) -> None:
    major_raw = major.sort_values("weighted_score", ascending=False)
    major_adv = major[major["prognosis_weighted_global_mortality_score"] > 0].copy()
    state_raw = states[(states["coverage_ok"]) & (~states["likely_lineage_only"])].sort_values("weighted_score", ascending=False)
    state_proxy_adv = states[
        (states["coverage_ok"])
        & (states["within_cancer_supported"])
        & (~states["likely_lineage_only"])
        & (states["prognosis_weighted_global_mortality_score"] > 0)
    ].copy()
    state_direct_adv = states[
        (states["coverage_ok"])
        & (states["within_cancer_supported"])
        & (~states["likely_lineage_only"])
        & (states["direct_prognosis_weighted_global_mortality_score"] > 0)
    ].copy()
    major_scores = pd.read_csv(OUT / "major_compartment_globocan_burden_scores.csv")
    state_scores = pd.read_csv(OUT / "nmf_state_globocan_burden_scores.csv")
    state_flags = states[["state_key", "likely_lineage_only"]].drop_duplicates("state_key")
    state_scores = state_scores.merge(state_flags, on="state_key", how="left")
    major_daly = major_scores[
        major_scores["source"].eq("WHO GHE 2021")
        & major_scores["location"].eq("Global")
        & major_scores["measure"].eq("daly")
    ].copy()
    major_yll = major_scores[
        major_scores["source"].eq("WHO GHE 2021")
        & major_scores["location"].eq("Global")
        & major_scores["measure"].eq("yll")
    ].copy()
    state_daly = state_scores[
        state_scores["source"].eq("WHO GHE 2021")
        & state_scores["location"].eq("Global")
        & state_scores["measure"].eq("daly")
        & state_scores["n_mapped_cancers"].ge(10)
        & (~state_scores["likely_lineage_only"].fillna(False))
    ].copy()
    state_yll = state_scores[
        state_scores["source"].eq("WHO GHE 2021")
        & state_scores["location"].eq("Global")
        & state_scores["measure"].eq("yll")
        & state_scores["n_mapped_cancers"].ge(10)
        & (~state_scores["likely_lineage_only"].fillna(False))
    ].copy()
    lines = [
        "# Pan-cancer ecosystem signal review",
        "",
        "## Raw burden representation",
        "",
        "Major compartments are ranked by global mortality-weighted representation. This is not prognosis-weighted.",
        *fmt_rows(major_raw, "compartment", "weighted_score", 8),
        "",
        "NMF states are ranked after multiplying their within-compartment NMF fraction by atlas-derived compartment abundance. Epithelial lineage-only states are excluded from this review list.",
        *fmt_rows(state_raw, "state_key", "weighted_score", 12),
        "",
        "## WHO DALY/YLL sensitivity",
        "",
        "DALY/YLL sensitivity uses WHO Global Health Estimates 2021 public files, not a direct authenticated IHME GBD export. It is a sensitivity layer for disability-adjusted and fatal burden, not a replacement for patient-level costs.",
        "",
        "Global DALY-weighted major compartments:",
        *fmt_burden_rows(major_daly, "compartment", 6),
        "",
        "Global YLL-weighted major compartments:",
        *fmt_burden_rows(major_yll, "compartment", 6),
        "",
        "Global DALY-weighted NMF states, excluding epithelial lineage-only states:",
        *fmt_burden_rows(state_daly, "state_key", 8),
        "",
        "Global YLL-weighted NMF states, excluding epithelial lineage-only states:",
        *fmt_burden_rows(state_yll, "state_key", 8),
        "",
        "## Prognosis-weighted priority",
        "",
        "Major compartments still use broad TCGA CIBERSORT proxies because direct compartment signatures are not available for every major cell type.",
        *fmt_rows(major_adv, "compartment", "prognosis_weighted_global_mortality_score", 8),
        "",
        "Top NMF states after coverage, within-cancer support and direct TCGA NMF-signature adverse filtering:",
        *fmt_rows(state_direct_adv, "state_key", "direct_prognosis_weighted_global_mortality_score", 12),
        "",
        "For comparison, broad CIBERSORT proxy filtering prioritized:",
        *fmt_rows(state_proxy_adv, "state_key", "prognosis_weighted_global_mortality_score", 8),
        "",
        "## Working interpretation",
        "",
        "- Raw ecosystem representation is broad: T cell, malignant epithelial, fibroblast and macrophage compartments dominate global mortality-weighted burden.",
        "- WHO GHE DALY/YLL sensitivity preserves the broad ecosystem conclusion, with T cell, fibroblast, malignant epithelial and macrophage compartments remaining prominent.",
        "- Direct TCGA NMF-signature prognosis changes the state-level story: proliferating T, epithelial cycling, myeloid cycling, SPP1+ macrophage, epithelial mesenchymal-like, desmoplastic fibroblast, NLRP3+ monocyte-derived macrophage and heat-shock myeloid states are the strongest adverse-priority signals.",
        "- C1QC+ macrophage is a high raw burden state but is not directly adverse in the stage/purity-adjusted TCGA signature model; it should be framed as representation-heavy rather than a primary adverse-cost driver.",
        "- Mast is adverse using the broad activated-mast CIBERSORT proxy but protective in the direct mast-signature model, so it should be treated as unresolved rather than a lead claim.",
        "- This gives a defensible subcluster story spanning immune, stromal and malignant programs, not a TAM-only story.",
        "- Cost results remain modeled/inferred, not observed cell-type spending.",
    ]
    (OUT / "pancancer_ecosystem_signal_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cox = pd.read_csv(OUT / "tcga_cibersort_all_stratified_cox.csv")
    major = make_major_review(cox)
    states = make_state_review(cox)
    write_review(major, states)
    print(f"wrote {OUT / 'ecosystem_major_compartment_priority_review.csv'}")
    print(f"wrote {OUT / 'ecosystem_nmf_state_priority_review.csv'}")
    print(f"wrote {OUT / 'pancancer_ecosystem_signal_review.md'}")


if __name__ == "__main__":
    main()
