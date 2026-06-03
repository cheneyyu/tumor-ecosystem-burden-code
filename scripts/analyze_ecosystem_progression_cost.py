#!/usr/bin/env python3
"""Progression, cost and prognosis interpretation for pan-cancer NMF states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "pancancer_ecosystem"
PRIMARY_MODEL = "stratified_by_tcga_cancer__age_sex_stage_purity_available"

AXES = {
    "epithelial_cycling_to_mesenchymal_stress": [
        "epithelial | Cell cycling",
        "epithelial | Complete mesenchymal",
        "epithelial | Stress",
    ],
    "fibroblast_quiescent_to_desmoplastic": [
        "mesenchymal | PI16+ fibroblast",
        "mesenchymal | Myofibroblast",
        "mesenchymal | Desmoplastic fibroblast",
    ],
    "myeloid_resident_to_inflammatory_spp1_cycling": [
        "myeloid | C1QC+ macrophage",
        "myeloid | CXCL9+ macrophage",
        "myeloid | FCN1+ monocyte derived macrophage",
        "myeloid | NLRP3+ monocyte derived macrophage",
        "myeloid | SPP1+ macrophage",
        "myeloid | Cell cycling",
    ],
    "tnk_cytotoxic_to_regulatory_proliferating": [
        "T_NK | CD16+ NK-cell",
        "T_NK | Exhausted CD8+ T-cell",
        "T_NK | Treg",
        "T_NK | Proliferating T-cell (cell cycling)",
    ],
}


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def fdr(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    mask = p.notna()
    if mask.any():
        out.loc[mask] = multipletests(p.loc[mask], method="fdr_bh")[1]
    return out


def short_label(state_key: str) -> str:
    label = str(state_key).split(" | ", 1)[-1]
    repl = {
        "Proliferating T-cell (cell cycling)": "Prolif. T",
        "Tissue resident memory T-cell": "TRM T",
        "Exhausted CD8+ T-cell": "Exh. CD8",
        "FCN1+ monocyte derived macrophage": "FCN1 mono-mac",
        "NLRP3+ monocyte derived macrophage": "NLRP3 mono-mac",
        "C1QC+ macrophage": "C1QC mac",
        "SPP1+ macrophage": "SPP1 mac",
        "CXCL9+ macrophage": "CXCL9 mac",
        "Cell cycling": "Cycling",
        "Heat shock": "Heat shock",
        "CD16+ NK-cell": "CD16 NK",
        "Complete mesenchymal": "Epi mes.",
        "PI16+ fibroblast": "PI16 fibro.",
        "Desmoplastic fibroblast": "Desmo fibro.",
        "Myofibroblast": "Myofibro.",
    }
    label = repl.get(label, label)
    comp = str(state_key).split(" | ", 1)[0]
    if label == "Cycling":
        if comp == "epithelial":
            return "Epi cycling"
        if comp == "myeloid":
            return "Mye cycling"
    return label


def load_state_table() -> pd.DataFrame:
    review = pd.read_csv(OUT / "ecosystem_nmf_state_priority_review.csv")
    cox = pd.read_csv(OUT / "tcga_nmf_state_signature_stratified_cox.csv")
    cox_full = cox[cox["model"].eq(PRIMARY_MODEL)].copy()
    cox_base = cox[cox["model"].eq("stratified_by_tcga_cancer__age_sex")].copy()
    cols = [
        "state_key",
        "weighted_score",
        "cost_weighted_score_billion_usd",
        "n_mapped_cancers",
        "total_burden_covered",
        "rank_desc",
        "rank_desc_cost",
        "within_cancer_variance_share",
        "between_cancer_variance_share",
        "state_category",
        "likely_lineage_only",
    ]
    table = review[cols].copy()
    table = table.merge(
        cox_base[
            [
                "state_key",
                "coef_log_hr_per_within_cancer_sd",
                "hr_per_within_cancer_sd",
                "p",
            ]
        ].rename(
            columns={
                "coef_log_hr_per_within_cancer_sd": "base_log_hr",
                "hr_per_within_cancer_sd": "base_hr",
                "p": "base_p",
            }
        ),
        on="state_key",
        how="left",
    )
    table = table.merge(
        cox_full[
            [
                "state_key",
                "coef_log_hr_per_within_cancer_sd",
                "hr_per_within_cancer_sd",
                "ci95_low",
                "ci95_high",
                "p",
                "n_samples",
                "n_events",
                "n_cancer_types",
            ]
        ].rename(
            columns={
                "coef_log_hr_per_within_cancer_sd": "full_log_hr",
                "hr_per_within_cancer_sd": "full_hr",
                "ci95_low": "full_ci95_low",
                "ci95_high": "full_ci95_high",
                "p": "full_p",
                "n_samples": "full_n_samples",
                "n_events": "full_n_events",
                "n_cancer_types": "full_n_cancer_types",
            }
        ),
        on="state_key",
        how="left",
    )
    table["cost_percentile"] = table["cost_weighted_score_billion_usd"].rank(pct=True)
    table["mortality_percentile"] = table["weighted_score"].rank(pct=True)
    table["adverse"] = (table["full_log_hr"] > 0) & (table["full_p"] < 0.05)
    table["protective"] = (table["full_log_hr"] < 0) & (table["full_p"] < 0.05)
    table["high_cost"] = table["cost_percentile"] >= 0.66
    table["low_cost"] = table["cost_percentile"] <= 0.34
    conditions = [
        table["high_cost"] & table["adverse"],
        table["high_cost"] & table["protective"],
        table["high_cost"] & ~(table["adverse"] | table["protective"]),
        table["low_cost"] & table["adverse"],
        table["low_cost"] & table["protective"],
        table["low_cost"] & ~(table["adverse"] | table["protective"]),
    ]
    labels = [
        "high cost + adverse",
        "high cost + protective",
        "high cost + neutral",
        "low cost + adverse",
        "low cost + protective",
        "low cost + neutral",
    ]
    table["cost_prognosis_class"] = np.select(conditions, labels, default="mid cost / mixed")
    table["short_label"] = table["state_key"].map(short_label)
    table.to_csv(OUT / "ecosystem_state_cost_prognosis_quadrants.csv", index=False)
    return table


def stage_gradient(scores: pd.DataFrame, state_cols: list[str]) -> pd.DataFrame:
    dat = scores.dropna(subset=["stage_ordinal", "CancerType"]).copy()
    dat = dat[dat["stage_ordinal"].between(1, 4)]
    rows = []
    for state in state_cols:
        sub = dat[["CancerType", "stage_ordinal", state]].dropna().copy()
        if len(sub) < 500 or sub["CancerType"].nunique() < 5:
            continue
        sub["state_z"] = sub.groupby("CancerType", observed=True)[state].transform(zscore)
        sub = sub.dropna(subset=["state_z"])
        cancer_counts = sub.groupby("CancerType", observed=True)["stage_ordinal"].nunique()
        valid_cancers = cancer_counts[cancer_counts >= 2].index
        sub = sub[sub["CancerType"].isin(valid_cancers)].copy()
        if len(sub) < 400 or sub["CancerType"].nunique() < 5:
            continue
        try:
            model = smf.ols("state_z ~ stage_ordinal + C(CancerType)", data=sub).fit(cov_type="HC3")
            beta = float(model.params["stage_ordinal"])
            p = float(model.pvalues["stage_ordinal"])
        except Exception:
            beta = np.nan
            p = np.nan
        means = sub.groupby("stage_ordinal", observed=True)["state_z"].mean().to_dict()
        try:
            rho, spearman_p = stats.spearmanr(sub["stage_ordinal"], sub["state_z"], nan_policy="omit")
        except Exception:
            rho, spearman_p = np.nan, np.nan
        rows.append(
            {
                "state_key": state,
                "n_samples": len(sub),
                "n_cancer_types": sub["CancerType"].nunique(),
                "stage_beta_with_cancer_fixed_effect": beta,
                "stage_p": p,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(spearman_p) if np.isfinite(spearman_p) else np.nan,
                "stage1_mean_z": means.get(1.0, np.nan),
                "stage2_mean_z": means.get(2.0, np.nan),
                "stage3_mean_z": means.get(3.0, np.nan),
                "stage4_mean_z": means.get(4.0, np.nan),
                "stage4_minus_stage1_mean_z": means.get(4.0, np.nan) - means.get(1.0, np.nan),
            }
        )
    out = pd.DataFrame(rows)
    out["stage_fdr"] = fdr(out["stage_p"])
    out["stage_direction"] = np.where(
        (out["stage_beta_with_cancer_fixed_effect"] > 0) & (out["stage_fdr"] < 0.1),
        "increases with stage",
        np.where(
            (out["stage_beta_with_cancer_fixed_effect"] < 0) & (out["stage_fdr"] < 0.1),
            "decreases with stage",
            "no strong stage trend",
        ),
    )
    out = out.sort_values("stage_beta_with_cancer_fixed_effect", ascending=False)
    out.to_csv(OUT / "ecosystem_state_stage_gradients.csv", index=False)
    return out


def stage_means(scores: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    dat = scores.dropna(subset=["stage_ordinal", "CancerType"]).copy()
    dat = dat[dat["stage_ordinal"].between(1, 4)]
    rows = []
    for state in states:
        if state not in dat.columns:
            continue
        sub = dat[["CancerType", "stage_ordinal", state]].dropna().copy()
        sub["state_z"] = sub.groupby("CancerType", observed=True)[state].transform(zscore)
        sub = sub.dropna(subset=["state_z"])
        for stage, ssub in sub.groupby("stage_ordinal", observed=True):
            rows.append(
                {
                    "state_key": state,
                    "stage_ordinal": int(stage),
                    "mean_within_cancer_z": ssub["state_z"].mean(),
                    "sem_within_cancer_z": ssub["state_z"].sem(),
                    "n_samples": len(ssub),
                    "n_cancer_types": ssub["CancerType"].nunique(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ecosystem_state_stage_means_selected.csv", index=False)
    return out


def state_axes(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dat = scores.copy()
    rows = []
    sample_axis = dat[["sample", "CancerType", "stage_ordinal", "event", "duration", "absolute_purity"]].copy()
    for axis_name, states in AXES.items():
        available = [s for s in states if s in dat.columns]
        if len(available) < 2:
            continue
        z_cols = []
        for s in available:
            col = f"z__{s}"
            dat[col] = dat.groupby("CancerType", observed=True)[s].transform(zscore)
            z_cols.append(col)
        axis_score = dat[z_cols].mean(axis=1)
        axis_col = f"axis__{axis_name}"
        sample_axis[axis_col] = axis_score
        sub = dat[["CancerType", "stage_ordinal"]].copy()
        sub["axis_score"] = axis_score
        sub = sub.dropna(subset=["axis_score", "CancerType"])
        stage_sub = sub.dropna(subset=["stage_ordinal"])
        if len(stage_sub) >= 400 and stage_sub["CancerType"].nunique() >= 5:
            try:
                model = smf.ols("axis_score ~ stage_ordinal + C(CancerType)", data=stage_sub).fit(cov_type="HC3")
                beta = float(model.params["stage_ordinal"])
                p = float(model.pvalues["stage_ordinal"])
            except Exception:
                beta, p = np.nan, np.nan
        else:
            beta, p = np.nan, np.nan
        corr_rows = []
        for i, s1 in enumerate(available):
            for s2 in available[i + 1 :]:
                pair = dat[[s1, s2, "CancerType"]].dropna().copy()
                pair["s1_z"] = pair.groupby("CancerType", observed=True)[s1].transform(zscore)
                pair["s2_z"] = pair.groupby("CancerType", observed=True)[s2].transform(zscore)
                pair = pair.dropna(subset=["s1_z", "s2_z"])
                if len(pair) < 200:
                    continue
                rho, p_pair = stats.spearmanr(pair["s1_z"], pair["s2_z"])
                corr_rows.append((s1, s2, rho, p_pair, len(pair)))
        rows.append(
            {
                "axis": axis_name,
                "states": "; ".join(available),
                "n_states": len(available),
                "n_samples": int(sub["axis_score"].notna().sum()),
                "stage_beta_with_cancer_fixed_effect": beta,
                "stage_p": p,
                "mean_pairwise_spearman_rho": np.nanmean([x[2] for x in corr_rows]) if corr_rows else np.nan,
                "min_pairwise_spearman_rho": np.nanmin([x[2] for x in corr_rows]) if corr_rows else np.nan,
                "max_pairwise_spearman_rho": np.nanmax([x[2] for x in corr_rows]) if corr_rows else np.nan,
            }
        )
    axis_summary = pd.DataFrame(rows)
    axis_summary["stage_fdr"] = fdr(axis_summary["stage_p"])
    axis_summary.to_csv(OUT / "ecosystem_state_axis_summary.csv", index=False)
    sample_axis.to_csv(OUT / "ecosystem_state_axis_scores_tcga.csv.gz", index=False, compression="gzip")
    return axis_summary, sample_axis


def axis_correlation_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    selected = sorted({s for states in AXES.values() for s in states if s in scores.columns})
    dat = scores[["CancerType"] + selected].dropna(subset=["CancerType"]).copy()
    z = pd.DataFrame(index=dat.index)
    for s in selected:
        z[s] = dat.groupby("CancerType", observed=True)[s].transform(zscore)
    corr = z.corr(method="spearman")
    corr.to_csv(OUT / "ecosystem_state_axis_within_cancer_spearman.csv")
    return corr


def combine_interpretation(state_table: pd.DataFrame, stage: pd.DataFrame, axes: pd.DataFrame) -> pd.DataFrame:
    out = state_table.merge(
        stage[
            [
                "state_key",
                "stage_beta_with_cancer_fixed_effect",
                "stage_p",
                "stage_fdr",
                "stage_direction",
                "stage4_minus_stage1_mean_z",
            ]
        ],
        on="state_key",
        how="left",
    )
    axis_map = []
    for axis, states in AXES.items():
        for state in states:
            axis_map.append({"state_key": state, "state_axis": axis})
    out = out.merge(pd.DataFrame(axis_map), on="state_key", how="left")
    out["progression_cost_prognosis_class"] = out["cost_prognosis_class"]
    out.loc[
        out["stage_direction"].eq("increases with stage") & out["adverse"] & out["high_cost"],
        "progression_cost_prognosis_class",
    ] = "stage-increasing high-cost adverse"
    out.loc[
        out["stage_direction"].eq("decreases with stage") & out["protective"],
        "progression_cost_prognosis_class",
    ] = "stage-decreasing protective"
    out.to_csv(OUT / "ecosystem_state_progression_cost_prognosis_map.csv", index=False)
    return out


def write_summary(table: pd.DataFrame, stage: pd.DataFrame, axes: pd.DataFrame) -> None:
    def lines_for(df: pd.DataFrame, sort_col: str, n: int = 8, ascending: bool = False) -> list[str]:
        rows = []
        for r in df.sort_values(sort_col, ascending=ascending).head(n).itertuples(index=False):
            rows.append(
                f"- {r.short_label} (`{r.state_key}`): cost={r.cost_weighted_score_billion_usd:.2f}B, "
                f"HR={r.full_hr:.2f}, stage beta={getattr(r, 'stage_beta_with_cancer_fixed_effect', np.nan):.3f}, "
                f"class={r.progression_cost_prognosis_class}"
            )
        return rows

    high_cost_adv = table[table["progression_cost_prognosis_class"].eq("stage-increasing high-cost adverse")].copy()
    low_cost_adv = table[table["cost_prognosis_class"].eq("low cost + adverse")].copy()
    high_cost_neutral = table[table["cost_prognosis_class"].eq("high cost + neutral")].copy()
    protective = table[table["protective"]].copy()
    stage_up = table[table["stage_direction"].eq("increases with stage")].copy()
    stage_down = table[table["stage_direction"].eq("decreases with stage")].copy()

    md = [
        "# Ecosystem progression-cost-prognosis interpretation",
        "",
        "This analysis extends burden ranking without adding DepMap. It combines NCI modeled cost-weighted representation, direct TCGA NMF-signature survival effects, TCGA clinical stage gradients and within-cancer state axes.",
        "",
        "## Stage-increasing high-cost adverse states",
        *lines_for(high_cost_adv, "cost_weighted_score_billion_usd", 10),
        "",
        "## High-cost neutral/protective states",
        *lines_for(high_cost_neutral, "cost_weighted_score_billion_usd", 10),
        "",
        "## Low-cost adverse states",
        *lines_for(low_cost_adv, "full_log_hr", 10),
        "",
        "## Protective states",
        *lines_for(protective, "full_log_hr", 10, ascending=True),
        "",
        "## Strongest stage-up states",
        *lines_for(stage_up, "stage_beta_with_cancer_fixed_effect", 10),
        "",
        "## Strongest stage-down states",
        *lines_for(stage_down, "stage_beta_with_cancer_fixed_effect", 10, ascending=True),
        "",
        "## State axes",
    ]
    for r in axes.sort_values("stage_beta_with_cancer_fixed_effect", ascending=False).itertuples(index=False):
        md.append(
            f"- `{r.axis}`: stage beta={r.stage_beta_with_cancer_fixed_effect:.3f}, "
            f"FDR={r.stage_fdr:.3g}, mean pairwise rho={r.mean_pairwise_spearman_rho:.2f}; states={r.states}"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "- A high cost score should not be read as observed cell-state spending. It is modeled representation in high-cost cancer sites.",
        "- The most clinically useful class is stage-increasing high-cost adverse states, because it links disease progression, modeled economic burden and poor survival.",
        "- High-cost neutral or protective states likely reflect cancer-site prevalence, treatment intensity or longer survival rather than intrinsically adverse biology.",
        "- Low-cost adverse states are not benign; they may represent poor-prognosis biology in cancer sites with lower modeled care costs or weaker current treatment intensity.",
        "",
    ]
    (OUT / "ecosystem_progression_cost_prognosis_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    state_table = load_state_table()
    scores = pd.read_csv(OUT / "tcga_nmf_state_signature_scores_survival_merged.csv.gz")
    state_cols = [c for c in scores.columns if " | " in c]
    stage = stage_gradient(scores, state_cols)
    selected_states = sorted(
        set(
            [
                "T_NK | Proliferating T-cell (cell cycling)",
                "T_NK | CD16+ NK-cell",
                "T_NK | Exhausted CD8+ T-cell",
                "T_NK | Treg",
                "epithelial | Cell cycling",
                "epithelial | Complete mesenchymal",
                "epithelial | Stress",
                "mesenchymal | PI16+ fibroblast",
                "mesenchymal | Myofibroblast",
                "mesenchymal | Desmoplastic fibroblast",
                "myeloid | C1QC+ macrophage",
                "myeloid | CXCL9+ macrophage",
                "myeloid | Cell cycling",
                "myeloid | NLRP3+ monocyte derived macrophage",
                "myeloid | SPP1+ macrophage",
                "myeloid | Heat shock",
            ]
        )
    )
    stage_means(scores, selected_states)
    axes, _sample_axis = state_axes(scores)
    axis_correlation_matrix(scores)
    interpretation = combine_interpretation(state_table, stage, axes)
    write_summary(interpretation, stage, axes)
    print(f"wrote {OUT / 'ecosystem_state_progression_cost_prognosis_map.csv'}")
    print(f"wrote {OUT / 'ecosystem_state_stage_gradients.csv'}")
    print(f"wrote {OUT / 'ecosystem_state_axis_summary.csv'}")
    print(f"wrote {OUT / 'ecosystem_progression_cost_prognosis_summary.md'}")


if __name__ == "__main__":
    main()
