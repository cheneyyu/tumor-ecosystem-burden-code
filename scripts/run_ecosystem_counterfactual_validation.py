#!/usr/bin/env python3
"""Counterfactual and negative-control validation for tumor ecosystem burden."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "pancancer_ecosystem"
COST_RAW = ROOT / "data" / "raw" / "costs" / "nci_cost_tables_raw.csv"
PRIMARY_MODEL = "stratified_by_tcga_cancer__age_sex_stage_purity_available"
N_PERMUTATIONS = 500
RANDOM_SEED = 20260602

NMF_TO_ATLAS_COMPARTMENTS = {
    "epithelial": ["Malignant epithelial"],
    "myeloid": ["Macrophage", "DC/pDC", "Mast"],
    "T_NK": ["T cell", "NK cell"],
    "B_plasma": ["B cell", "Plasma cell"],
    "mesenchymal": ["Fibroblast", "Endothelial"],
}

GLOBOCAN_TO_NCI = {
    "Bladder": "Bladder",
    "Brain, CNS": "Brain",
    "Breast": "Female Breast",
    "Cervix uteri": "Cervix Uteri",
    "Colorectum": "Colorectal",
    "Kidney": "Kidney",
    "Leukaemia": "Leukemia",
    "Liver": "Liver",
    "Lip, oral cavity": "Oral Cavity",
    "Lung": "Lung",
    "Melanoma of skin": "Melanoma",
    "Multiple myeloma": "Myeloma",
    "NHL": "Non-Hodgkin Lymphoma",
    "Ovary": "Ovary",
    "Pancreas": "Pancreas",
    "Prostate": "Prostate",
    "Stomach": "Stomach",
    "Thyroid": "Thyroid",
    "Corpus uteri": "Uterus",
}

TCGA_TO_GLOBOCAN = {
    "BLCA": "Bladder",
    "BRCA": "Breast",
    "CESC": "Cervix uteri",
    "COAD": "Colorectum",
    "READ": "Colorectum",
    "DLBC": "NHL",
    "ESCA": "Oesophagus",
    "GBM": "Brain, CNS",
    "LGG": "Brain, CNS",
    "HNSC": "Lip, oral cavity",
    "KICH": "Kidney",
    "KIRC": "Kidney",
    "KIRP": "Kidney",
    "LAML": "Leukaemia",
    "LIHC": "Liver",
    "LUAD": "Lung",
    "LUSC": "Lung",
    "MESO": "Mesothelioma",
    "OV": "Ovary",
    "PAAD": "Pancreas",
    "PRAD": "Prostate",
    "SKCM": "Melanoma of skin",
    "STAD": "Stomach",
    "TGCT": "Testis",
    "THCA": "Thyroid",
    "UCEC": "Corpus uteri",
    "UCS": "Corpus uteri",
}

PRIORITY_STATES = [
    "T_NK | Proliferating T-cell (cell cycling)",
    "epithelial | Complete mesenchymal",
    "epithelial | Cell cycling",
    "myeloid | Cell cycling",
    "myeloid | Heat shock",
    "mesenchymal | Desmoplastic fibroblast",
    "myeloid | SPP1+ macrophage",
]

PROTECTIVE_CONTROL_STATES = [
    "T_NK | Exhausted CD8+ T-cell",
    "myeloid | Mast",
]


def short_state(x: str) -> str:
    comp, state = str(x).split(" | ", 1) if " | " in str(x) else ("", str(x))
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
        "Plasma cell": "Plasma",
        "Treg": "Treg",
    }
    label = repl.get(state, state[:22])
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


def clean_money(x: object) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"", "-", "nan", "NaN"}:
        return np.nan
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def rank_desc(s: pd.Series) -> pd.Series:
    return s.rank(method="min", ascending=False)


def load_global_mortality_terms() -> pd.DataFrame:
    terms = pd.read_csv(OUT / "nmf_state_globocan_burden_terms.csv")
    terms = terms[
        terms["source"].astype(str).str.startswith("GLOBOCAN", na=False)
        & terms["location"].eq("Global")
        & terms["measure"].eq("mortality")
    ].copy()
    terms["burden_total"] = pd.to_numeric(terms["burden_total"], errors="coerce")
    terms["state_representation_all_tumor"] = pd.to_numeric(terms["state_representation_all_tumor"], errors="coerce")
    terms["weighted_representation"] = terms["burden_total"] * terms["state_representation_all_tumor"]
    return terms


def load_cost_terms() -> pd.DataFrame:
    terms = pd.read_csv(OUT / "nmf_state_nci_cost_terms.csv")
    terms["cost_2020_billion_usd"] = pd.to_numeric(terms["cost_2020_billion_usd"], errors="coerce")
    terms["state_representation_all_tumor"] = pd.to_numeric(terms["state_representation_all_tumor"], errors="coerce")
    terms["weighted_cost_score_billion_usd"] = terms["cost_2020_billion_usd"] * terms["state_representation_all_tumor"]
    return terms


def summarize_weighted_terms(terms: pd.DataFrame, weight_col: str, score_col: str, endpoint: str) -> pd.DataFrame:
    rows = []
    mean_weight = terms[["CancerAbbr", weight_col]].drop_duplicates()[weight_col].mean()
    for state, sub in terms.groupby("state_key", observed=True):
        sub = sub.dropna(subset=[weight_col, "state_representation_all_tumor"]).copy()
        if sub.empty:
            continue
        rows.append(
            {
                "endpoint": endpoint,
                "state_key": state,
                "nmf_compartment": sub["nmf_compartment"].iloc[0],
                "observed_score": sub[score_col].sum(),
                "equal_site_score": sub["state_representation_all_tumor"].mean(),
                "balanced_weight_score": mean_weight * sub["state_representation_all_tumor"].sum(),
                "n_sites": sub["CancerAbbr"].nunique(),
                "total_weight_covered": sub[[weight_col, "CancerAbbr"]].drop_duplicates()[weight_col].sum(),
                "top_site": sub.sort_values(score_col, ascending=False)["CancerAbbr"].iloc[0],
                "top_site_score_share": sub[score_col].max() / sub[score_col].sum() if sub[score_col].sum() > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    for col in ["observed_score", "equal_site_score", "balanced_weight_score"]:
        out[f"{col}_rank"] = rank_desc(out[col])
        out[f"{col}_pct"] = out[col].rank(pct=True)
    out["composition_class"] = np.select(
        [
            (out["observed_score_pct"] >= 0.66) & (out["equal_site_score_pct"] >= 0.66),
            (out["observed_score_pct"] >= 0.66) & (out["equal_site_score_pct"] < 0.50),
            (out["observed_score_pct"] < 0.66) & (out["equal_site_score_pct"] >= 0.66),
        ],
        ["composition-robust", "site-driven", "biology-recurring"],
        default="mixed",
    )
    return out.sort_values(["endpoint", "observed_score_rank"])


def composition_balanced_scores() -> pd.DataFrame:
    mort = summarize_weighted_terms(load_global_mortality_terms(), "burden_total", "weighted_representation", "global_mortality")
    cost = summarize_weighted_terms(load_cost_terms(), "cost_2020_billion_usd", "weighted_cost_score_billion_usd", "nci_total_cost")
    out = pd.concat([mort, cost], ignore_index=True)
    review_cols = [
        "state_key",
        "state_category",
        "likely_lineage_only",
        "within_cancer_variance_share",
        "direct_prognosis_weighted_global_mortality_score",
    ]
    review = pd.read_csv(OUT / "ecosystem_nmf_state_priority_review.csv")[review_cols].drop_duplicates("state_key")
    out = out.merge(review, on="state_key", how="left")
    out["short_label"] = out["state_key"].map(short_state)
    out.to_csv(OUT / "ecosystem_state_composition_balanced_scores.csv", index=False)
    return out


def sample_level_state_representation() -> tuple[pd.DataFrame, pd.DataFrame]:
    nmf = pd.read_csv(OUT / "nmf_state_fraction_by_sample_long.csv")
    wide = pd.read_csv(OUT / "ecosystem_sample_compartment_fractions_wide.csv")
    comp_cols = []
    for nmf_comp, atlas_comps in NMF_TO_ATLAS_COMPARTMENTS.items():
        cols = [f"frac_all__{c}" for c in atlas_comps if f"frac_all__{c}" in wide.columns]
        wide[f"atlas_frac_for_{nmf_comp}"] = wide[cols].sum(axis=1) if cols else np.nan
        comp_cols.append(f"atlas_frac_for_{nmf_comp}")

    sample_meta = wide[
        ["Dataset", "Organ_origin", "Patient", "Cancer type", "globocan_label", "nci_cost_site", *comp_cols]
    ].copy()
    direct = (
        sample_meta.groupby(["Dataset", "Organ_origin", "Patient"], dropna=False, observed=True)[comp_cols]
        .mean()
        .reset_index()
    )
    direct["match_source"] = "dataset_patient"
    cancer = (
        sample_meta.groupby(["Cancer type"], dropna=False, observed=True)[comp_cols]
        .mean()
        .reset_index()
        .rename(columns={"Cancer type": "CancerAbbr"})
    )
    glob = (
        sample_meta.groupby(["globocan_label"], dropna=False, observed=True)[comp_cols]
        .mean()
        .reset_index()
    )

    x = nmf.merge(direct, on=["Dataset", "Organ_origin", "Patient"], how="left")
    for col in comp_cols:
        x = x.rename(columns={col: f"{col}__direct"})
    x = x.merge(cancer, on="CancerAbbr", how="left")
    for col in comp_cols:
        x = x.rename(columns={col: f"{col}__cancer"})
    x = x.merge(glob, on="globocan_label", how="left")
    for col in comp_cols:
        x = x.rename(columns={col: f"{col}__globocan"})

    values = []
    sources = []
    for r in x.itertuples(index=False):
        base = f"atlas_frac_for_{r.nmf_compartment}"
        direct_val = getattr(r, f"{base}__direct", np.nan)
        cancer_val = getattr(r, f"{base}__cancer", np.nan)
        glob_val = getattr(r, f"{base}__globocan", np.nan)
        if np.isfinite(direct_val):
            values.append(float(direct_val))
            sources.append("dataset_patient")
        elif np.isfinite(cancer_val):
            values.append(float(cancer_val))
            sources.append("cancer_mean")
        elif np.isfinite(glob_val):
            values.append(float(glob_val))
            sources.append("globocan_mean")
        else:
            values.append(np.nan)
            sources.append("missing")
    x["atlas_compartment_frac_all_tumor_sample"] = values
    x["compartment_fraction_source"] = sources
    x["state_representation_sample_all_tumor"] = (
        pd.to_numeric(x["frac_in_nmf_compartment"], errors="coerce") * x["atlas_compartment_frac_all_tumor_sample"]
    )
    keep = [
        "Patient_Organ_Tissue",
        "Dataset",
        "Organ_origin",
        "Patient",
        "CancerAbbr",
        "globocan_label",
        "nci_cost_site",
        "nmf_compartment",
        "Cell_state",
        "state_category",
        "state_key",
        "n_modules",
        "frac_in_nmf_compartment",
        "atlas_compartment_frac_all_tumor_sample",
        "state_representation_sample_all_tumor",
        "compartment_fraction_source",
    ]
    sample_state = x[keep].copy()
    match_summary = (
        sample_state.drop_duplicates(["Patient_Organ_Tissue", "nmf_compartment"])
        .groupby(["nmf_compartment", "compartment_fraction_source"], observed=True)
        .size()
        .reset_index(name="n_sample_compartment_pairs")
    )
    return sample_state, match_summary


def within_cancer_counterfactual_scores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_state, match_summary = sample_level_state_representation()
    rows = []
    group_cols = ["CancerAbbr", "globocan_label", "nci_cost_site", "nmf_compartment", "Cell_state", "state_category", "state_key"]
    for keys, sub in sample_state.dropna(subset=["state_representation_sample_all_tumor"]).groupby(group_cols, dropna=False, observed=True):
        vals = pd.to_numeric(sub["state_representation_sample_all_tumor"], errors="coerce").dropna()
        if vals.empty:
            continue
        q25 = vals.quantile(0.25)
        excess = np.maximum(vals - q25, 0)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_samples": int(vals.size),
                "mean_sample_representation": float(vals.mean()),
                "q25_low_state_baseline": float(q25),
                "mean_excess_above_q25": float(excess.mean()),
                "excess_fraction_of_mean": float(excess.mean() / vals.mean()) if vals.mean() > 0 else np.nan,
                "direct_match_fraction": float((sub["compartment_fraction_source"] == "dataset_patient").mean()),
            }
        )
        rows.append(row)
    cf = pd.DataFrame(rows)

    mort_weights = (
        load_global_mortality_terms()[["CancerAbbr", "globocan_label", "burden_total"]]
        .drop_duplicates(["CancerAbbr", "globocan_label"])
    )
    cost_weights = (
        load_cost_terms()[["CancerAbbr", "nci_cost_site", "cost_2020_billion_usd"]]
        .drop_duplicates(["CancerAbbr", "nci_cost_site"])
    )
    published_observed_mort = (
        load_global_mortality_terms()
        .groupby("state_key", observed=True)["weighted_representation"]
        .sum()
        .rename("published_observed_global_mortality_score")
        .reset_index()
    )
    published_observed_cost = (
        load_cost_terms()
        .groupby("state_key", observed=True)["weighted_cost_score_billion_usd"]
        .sum()
        .rename("published_observed_nci_cost_score_billion_usd")
        .reset_index()
    )

    terms = cf.merge(mort_weights, on=["CancerAbbr", "globocan_label"], how="left").merge(
        cost_weights, on=["CancerAbbr", "nci_cost_site"], how="left"
    )
    terms["counterfactual_reducible_mortality_score"] = terms["burden_total"] * terms["mean_excess_above_q25"]
    terms["counterfactual_reducible_cost_score_billion_usd"] = (
        terms["cost_2020_billion_usd"] * terms["mean_excess_above_q25"]
    )
    terms["sample_observed_mortality_score"] = terms["burden_total"] * terms["mean_sample_representation"]
    terms["sample_observed_cost_score_billion_usd"] = terms["cost_2020_billion_usd"] * terms["mean_sample_representation"]
    scores = (
        terms.groupby("state_key", observed=True)
        .agg(
            nmf_compartment=("nmf_compartment", "first"),
            state_category=("state_category", "first"),
            n_cancers=("CancerAbbr", "nunique"),
            n_samples=("n_samples", "sum"),
            mean_excess_fraction_of_mean=("excess_fraction_of_mean", "mean"),
            mean_direct_match_fraction=("direct_match_fraction", "mean"),
            reducible_global_mortality_score=("counterfactual_reducible_mortality_score", "sum"),
            reducible_nci_cost_score_billion_usd=("counterfactual_reducible_cost_score_billion_usd", "sum"),
            sample_observed_global_mortality_score=("sample_observed_mortality_score", "sum"),
            sample_observed_nci_cost_score_billion_usd=("sample_observed_cost_score_billion_usd", "sum"),
            total_burden_covered=("burden_total", "sum"),
            total_cost_covered_billion_usd=("cost_2020_billion_usd", "sum"),
        )
        .reset_index()
    )
    scores = scores.merge(published_observed_mort, on="state_key", how="left").merge(published_observed_cost, on="state_key", how="left")
    scores["reducible_fraction_of_observed_mortality"] = (
        scores["reducible_global_mortality_score"] / scores["sample_observed_global_mortality_score"]
    )
    scores["reducible_fraction_of_observed_cost"] = (
        scores["reducible_nci_cost_score_billion_usd"] / scores["sample_observed_nci_cost_score_billion_usd"]
    )
    scores["reducible_mortality_rank"] = rank_desc(scores["reducible_global_mortality_score"])
    scores["reducible_cost_rank"] = rank_desc(scores["reducible_nci_cost_score_billion_usd"])
    scores["short_label"] = scores["state_key"].map(short_state)

    terms.to_csv(OUT / "ecosystem_state_within_cancer_counterfactual_terms.csv", index=False)
    scores.to_csv(OUT / "ecosystem_state_within_cancer_counterfactual_scores.csv", index=False)
    match_summary.to_csv(OUT / "ecosystem_state_counterfactual_sample_match_summary.csv", index=False)
    return terms, scores, match_summary


def phase_cost_table() -> pd.DataFrame:
    raw = pd.read_csv(COST_RAW)
    raw = raw[raw["table_index"].isin([4, 5])].copy()
    raw["cancer_site"] = raw["Cancer Site"].replace(
        {
            "All Sites": "All sites",
            "Breast": "Female Breast",
            "Lung and Bronchus": "Lung",
        }
    )
    phase_cols = ["Initial care", "Continuing care", "Last year of life"]
    for col in phase_cols:
        raw[col] = raw[col].map(clean_money)
    phase = raw.melt(
        id_vars=["table_index", "cancer_site"],
        value_vars=phase_cols,
        var_name="phase",
        value_name="phase_cost_usd_per_patient",
    )
    phase["component"] = np.where(phase["table_index"].eq(4), "medical_services", "oral_prescription_drugs")
    phase = phase[phase["cancer_site"].ne("All sites")].copy()
    summed = (
        phase.groupby(["cancer_site", "phase"], observed=True)["phase_cost_usd_per_patient"]
        .sum(min_count=1)
        .reset_index()
    )
    return summed


def phase_of_care_cost_scores() -> pd.DataFrame:
    rep = pd.read_csv(OUT / "nmf_state_representation_by_cancer.csv")
    rep["phase_cost_site"] = rep["globocan_label"].map(GLOBOCAN_TO_NCI)
    phase = phase_cost_table()
    terms = rep.merge(phase, left_on="phase_cost_site", right_on="cancer_site", how="inner")
    terms["phase_weighted_score"] = terms["state_representation_all_tumor"] * terms["phase_cost_usd_per_patient"]
    scores = (
        terms.groupby(["state_key", "phase"], observed=True)
        .agg(
            nmf_compartment=("nmf_compartment", "first"),
            phase_weighted_score=("phase_weighted_score", "sum"),
            n_phase_cost_sites=("phase_cost_site", "nunique"),
            mean_phase_cost_usd_per_patient=("phase_cost_usd_per_patient", "mean"),
        )
        .reset_index()
    )
    wide = scores.pivot_table(index=["state_key", "nmf_compartment"], columns="phase", values="phase_weighted_score", observed=True).reset_index()
    wide.columns.name = None
    for col in ["Initial care", "Continuing care", "Last year of life"]:
        if col not in wide:
            wide[col] = np.nan
    wide = wide.rename(
        columns={
            "Initial care": "initial_care_phase_score",
            "Continuing care": "continuing_care_phase_score",
            "Last year of life": "last_year_of_life_phase_score",
        }
    )
    wide["terminal_to_continuing_enrichment"] = (
        wide["last_year_of_life_phase_score"] / wide["continuing_care_phase_score"].replace(0, np.nan)
    )
    wide["terminal_to_initial_enrichment"] = (
        wide["last_year_of_life_phase_score"] / wide["initial_care_phase_score"].replace(0, np.nan)
    )
    nsites = terms.groupby("state_key", observed=True)["phase_cost_site"].nunique().rename("n_phase_cost_sites").reset_index()
    wide = wide.merge(nsites, on="state_key", how="left")
    wide["terminal_enrichment_rank"] = rank_desc(wide["terminal_to_continuing_enrichment"])
    wide["last_year_phase_rank"] = rank_desc(wide["last_year_of_life_phase_score"])
    wide["short_label"] = wide["state_key"].map(short_state)
    wide.to_csv(OUT / "ecosystem_state_phase_of_care_cost_scores.csv", index=False)
    terms.to_csv(OUT / "ecosystem_state_phase_of_care_cost_terms.csv", index=False)
    return wide


def tcga_paf_counterfactual() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(OUT / "tcga_nmf_state_signature_scores_survival_merged.csv.gz")
    cox = pd.read_csv(OUT / "tcga_nmf_state_signature_stratified_cox.csv")
    cox = cox[cox["model"].eq(PRIMARY_MODEL)].copy()
    cox["coef_log_hr_per_within_cancer_sd"] = pd.to_numeric(cox["coef_log_hr_per_within_cancer_sd"], errors="coerce")
    cox["p"] = pd.to_numeric(cox["p"], errors="coerce")
    cox = cox.dropna(subset=["state_key", "coef_log_hr_per_within_cancer_sd"])

    burden = load_global_mortality_terms()[["globocan_label", "burden_total"]].drop_duplicates("globocan_label")
    total_cost = (
        pd.read_csv(ROOT / "data" / "raw" / "costs" / "nci_cost_tables_long.csv")
        .query("table_index == 1 and metric == '2020'")
        [["cancer_site", "value"]]
        .rename(columns={"value": "cost_2020_billion_usd"})
    )
    total_cost["cost_2020_billion_usd"] = pd.to_numeric(total_cost["cost_2020_billion_usd"], errors="coerce")
    rows = []
    state_cols = [c for c in scores.columns if " | " in c]
    for state in sorted(set(state_cols).intersection(set(cox["state_key"]))):
        beta = float(cox.loc[cox["state_key"].eq(state), "coef_log_hr_per_within_cancer_sd"].iloc[0])
        p = float(cox.loc[cox["state_key"].eq(state), "p"].iloc[0])
        sub = scores[["CancerType", state]].dropna().copy()
        sub = sub[sub["CancerType"].astype(str).isin(TCGA_TO_GLOBOCAN)].copy()
        if sub.empty:
            continue
        sub["score_z"] = sub.groupby("CancerType", observed=True)[state].transform(zscore)
        sub = sub.dropna(subset=["score_z"])
        for cancer, g in sub.groupby("CancerType", observed=True):
            if len(g) < 20:
                continue
            q25 = g["score_z"].quantile(0.25)
            q75 = g["score_z"].quantile(0.75)
            iqr = q75 - q25
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            high = g["score_z"] >= q75
            p_high = float(high.mean())
            hr_iqr = float(np.exp(beta * iqr))
            paf_raw = p_high * (hr_iqr - 1.0) / (p_high * (hr_iqr - 1.0) + 1.0)
            paf_adverse = max(paf_raw, 0.0) if beta > 0 and p < 0.05 else 0.0
            rows.append(
                {
                    "state_key": state,
                    "CancerType": cancer,
                    "globocan_label": TCGA_TO_GLOBOCAN[cancer],
                    "n_samples": len(g),
                    "q25_within_cancer_z": q25,
                    "q75_within_cancer_z": q75,
                    "iqr_within_cancer_z": iqr,
                    "p_high_top_quartile": p_high,
                    "cox_beta_per_sd": beta,
                    "cox_p": p,
                    "hr_iqr": hr_iqr,
                    "paf_raw": paf_raw,
                    "paf_adverse_only": paf_adverse,
                }
            )
    terms = pd.DataFrame(rows)
    if terms.empty:
        out = pd.DataFrame()
    else:
        glob_terms = (
            terms.groupby(["state_key", "globocan_label"], observed=True)
            .agg(
                mean_paf_raw=("paf_raw", "mean"),
                mean_paf_adverse_only=("paf_adverse_only", "mean"),
                n_tcga_samples=("n_samples", "sum"),
                n_tcga_cancer_types=("CancerType", "nunique"),
            )
            .reset_index()
        )
        glob_terms = glob_terms.merge(burden, on="globocan_label", how="left")
        glob_terms["nci_cost_site"] = glob_terms["globocan_label"].map(GLOBOCAN_TO_NCI)
        glob_terms = glob_terms.merge(total_cost, left_on="nci_cost_site", right_on="cancer_site", how="left")
        glob_terms["paf_burden_mortality_score"] = glob_terms["burden_total"] * glob_terms["mean_paf_adverse_only"]
        glob_terms["paf_cost_score_billion_usd"] = glob_terms["cost_2020_billion_usd"] * glob_terms["mean_paf_adverse_only"]
        out = (
            glob_terms.groupby("state_key", observed=True)
            .agg(
                n_globocan_sites=("globocan_label", "nunique"),
                n_tcga_cancer_types=("n_tcga_cancer_types", "sum"),
                n_tcga_samples=("n_tcga_samples", "sum"),
                mean_paf_raw=("mean_paf_raw", "mean"),
                mean_paf_adverse_only=("mean_paf_adverse_only", "mean"),
                paf_burden_mortality_score=("paf_burden_mortality_score", "sum"),
                paf_cost_score_billion_usd=("paf_cost_score_billion_usd", "sum"),
            )
            .reset_index()
        )
        out = out.merge(cox[["state_key", "coef_log_hr_per_within_cancer_sd", "hr_per_within_cancer_sd", "p"]], on="state_key", how="left")
        out["paf_burden_rank"] = rank_desc(out["paf_burden_mortality_score"])
        out["paf_cost_rank"] = rank_desc(out["paf_cost_score_billion_usd"])
        out["short_label"] = out["state_key"].map(short_state)
        glob_terms.to_csv(OUT / "ecosystem_state_tcga_paf_counterfactual_terms.csv", index=False)
    terms.to_csv(OUT / "ecosystem_state_tcga_paf_counterfactual_by_tcga_cancer.csv", index=False)
    out.to_csv(OUT / "ecosystem_state_tcga_paf_counterfactual.csv", index=False)
    return terms, out


def negative_controls(counterfactual_scores: pd.DataFrame, composition_scores: pd.DataFrame, paf_scores: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    terms = load_global_mortality_terms().dropna(subset=["state_representation_all_tumor", "burden_total"]).copy()
    terms = terms[~terms["likely_lineage_only"].fillna(False)] if "likely_lineage_only" in terms.columns else terms
    observed = terms.groupby("state_key", observed=True)["weighted_representation"].sum()
    top_states = [s for s in PRIORITY_STATES if s in observed.index]
    null_label = {s: [] for s in top_states}
    null_weight = {s: [] for s in top_states}
    null_rows = []

    base = terms[["CancerAbbr", "nmf_compartment", "state_key", "burden_total", "state_representation_all_tumor"]].copy()
    base["weighted_representation"] = base["burden_total"] * base["state_representation_all_tumor"]
    cancer_weights = base[["CancerAbbr", "burden_total"]].drop_duplicates()
    for _ in range(N_PERMUTATIONS):
        perm = base.copy()
        perm_keys = []
        for _, idx in perm.groupby("nmf_compartment", observed=True).groups.items():
            values = perm.loc[idx, "state_key"].to_numpy()
            perm_keys.extend(zip(idx, rng.permutation(values)))
        perm_map = pd.Series({idx: key for idx, key in perm_keys})
        perm["perm_state_key"] = perm.index.map(perm_map)
        scores = perm.groupby("perm_state_key", observed=True)["weighted_representation"].sum()
        for state in top_states:
            value = float(scores.get(state, 0.0))
            null_label[state].append(value)
            null_rows.append(
                {
                    "control_type": "state_label_permutation",
                    "endpoint": "global_mortality_observed_score",
                    "state_key": state,
                    "iteration": len(null_label[state]),
                    "null_score": value,
                }
            )

        shuffled = cancer_weights.copy()
        shuffled["perm_burden_total"] = rng.permutation(shuffled["burden_total"].to_numpy())
        perm_w = base.drop(columns=["burden_total"]).merge(shuffled[["CancerAbbr", "perm_burden_total"]], on="CancerAbbr", how="left")
        perm_w["perm_weighted"] = perm_w["perm_burden_total"] * perm_w["state_representation_all_tumor"]
        scores_w = perm_w.groupby("state_key", observed=True)["perm_weighted"].sum()
        for state in top_states:
            value = float(scores_w.get(state, 0.0))
            null_weight[state].append(value)
            null_rows.append(
                {
                    "control_type": "cancer_weight_permutation",
                    "endpoint": "global_mortality_observed_score",
                    "state_key": state,
                    "iteration": len(null_weight[state]),
                    "null_score": value,
                }
            )

    rows = []
    for control_type, nulls in [("state_label_permutation", null_label), ("cancer_weight_permutation", null_weight)]:
        for state, vals in nulls.items():
            arr = np.asarray(vals, dtype=float)
            obs = float(observed.get(state, 0.0))
            rows.append(
                {
                    "control_type": control_type,
                    "endpoint": "global_mortality_observed_score",
                    "state_key": state,
                    "short_label": short_state(state),
                    "observed_score": obs,
                    "null_mean": float(arr.mean()),
                    "null_p95": float(np.quantile(arr, 0.95)),
                    "empirical_p_ge_observed": float((np.sum(arr >= obs) + 1) / (len(arr) + 1)),
                    "passes_null_p05": bool(((np.sum(arr >= obs) + 1) / (len(arr) + 1)) < 0.05),
                    "note": "top adverse/progression state should exceed null",
                }
            )

    for state in PROTECTIVE_CONTROL_STATES:
        if state in set(paf_scores.get("state_key", [])):
            row = paf_scores[paf_scores["state_key"].eq(state)].iloc[0]
            rows.append(
                {
                    "control_type": "protective_stage_down_control",
                    "endpoint": "tcga_paf_adverse_only",
                    "state_key": state,
                    "short_label": short_state(state),
                    "observed_score": row.get("paf_burden_mortality_score", np.nan),
                    "null_mean": np.nan,
                    "null_p95": np.nan,
                    "empirical_p_ge_observed": np.nan,
                    "passes_null_p05": bool(row.get("paf_burden_mortality_score", 0.0) <= 1e-9),
                    "note": "protective/stage-down control should not produce adverse PAF burden",
                }
            )

    if "likely_lineage_only" in composition_scores.columns and not paf_scores.empty:
        comp = composition_scores[composition_scores["endpoint"].eq("global_mortality")][["state_key", "likely_lineage_only"]].drop_duplicates("state_key")
        integrated = counterfactual_scores.merge(
            paf_scores[["state_key", "mean_paf_adverse_only", "paf_burden_mortality_score"]],
            on="state_key",
            how="left",
        ).merge(comp, on="state_key", how="left")
        integrated["integrated_adverse_counterfactual_score"] = (
            integrated["reducible_global_mortality_score"] * integrated["mean_paf_adverse_only"].fillna(0)
        )
        top = integrated.sort_values("integrated_adverse_counterfactual_score", ascending=False).head(10)
        n_lineage_in_top = int(top["likely_lineage_only"].fillna(False).sum())
        rows.append(
            {
                "control_type": "lineage_only_control",
                "endpoint": "integrated_adverse_counterfactual_top10",
                "state_key": "lineage_only_states",
                "short_label": "Lineage-only",
                "observed_score": n_lineage_in_top,
                "null_mean": np.nan,
                "null_p95": np.nan,
                "empirical_p_ge_observed": np.nan,
                "passes_null_p05": bool(n_lineage_in_top == 0),
                "note": "lineage-only states should not enter the top integrated adverse counterfactual set",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ecosystem_state_negative_control_summary.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT / "ecosystem_state_negative_control_null_distribution.csv", index=False)
    return out


def write_summary(
    composition: pd.DataFrame,
    cf_scores: pd.DataFrame,
    paf: pd.DataFrame,
    phase: pd.DataFrame,
    neg: pd.DataFrame,
    match_summary: pd.DataFrame,
) -> None:
    prog = pd.read_csv(OUT / "ecosystem_state_progression_cost_prognosis_map.csv")
    prog = prog[["state_key", "progression_cost_prognosis_class", "full_hr", "full_p", "stage_beta_with_cancer_fixed_effect"]]
    merged = cf_scores.merge(prog, on="state_key", how="left").merge(
        paf[["state_key", "paf_burden_mortality_score", "paf_cost_score_billion_usd"]], on="state_key", how="left"
    ).merge(
        phase[["state_key", "terminal_to_continuing_enrichment"]], on="state_key", how="left"
    )
    top = merged.sort_values("reducible_nci_cost_score_billion_usd", ascending=False).head(8)
    phase_top = phase.merge(prog, on="state_key", how="left").sort_values("terminal_to_continuing_enrichment", ascending=False).head(8)
    neg_pass = neg["passes_null_p05"].fillna(False).sum() if not neg.empty else 0
    neg_total = len(neg)

    lines = [
        "# Ecosystem counterfactual validation summary",
        "",
        "This analysis adds modeled counterfactual and negative-control validation to the pan-cancer ecosystem burden framework. It estimates excess state representation above a within-cancer low-state baseline and should not be read as observed cell-state medical spending or a proven causal intervention.",
        "",
        "## Sample-level counterfactual representation matching",
        "",
        *[
            f"- {r.nmf_compartment}: {r.compartment_fraction_source} pairs={r.n_sample_compartment_pairs}"
            for r in match_summary.sort_values(["nmf_compartment", "compartment_fraction_source"]).itertuples(index=False)
        ],
        "",
        "## Top reducible modeled cost states",
        "",
    ]
    for r in top.itertuples(index=False):
        lines.append(
            f"- {short_state(r.state_key)} (`{r.state_key}`): reducible cost={r.reducible_nci_cost_score_billion_usd:.2f}B, "
            f"reducible mortality={r.reducible_global_mortality_score:,.0f}, "
            f"fraction of observed cost={r.reducible_fraction_of_observed_cost:.2f}, "
            f"PAF burden={getattr(r, 'paf_burden_mortality_score', np.nan):,.0f}, "
            f"terminal/continuing={getattr(r, 'terminal_to_continuing_enrichment', np.nan):.2f}, "
            f"class={getattr(r, 'progression_cost_prognosis_class', '')}"
        )
    lines += [
        "",
        "## Strongest terminal-care phase enrichment",
        "",
    ]
    for r in phase_top.itertuples(index=False):
        lines.append(
            f"- {short_state(r.state_key)} (`{r.state_key}`): terminal/continuing={r.terminal_to_continuing_enrichment:.2f}, "
            f"last-year phase score={r.last_year_of_life_phase_score:.1f}, class={getattr(r, 'progression_cost_prognosis_class', '')}"
        )
    lines += [
        "",
        "## Negative controls",
        "",
        f"- Passing rows: {int(neg_pass)} of {neg_total}.",
    ]
    for r in neg.sort_values(["control_type", "state_key"]).itertuples(index=False):
        ptxt = "NA" if pd.isna(r.empirical_p_ge_observed) else f"{r.empirical_p_ge_observed:.3g}"
        lines.append(
            f"- {r.control_type} / {short_state(r.state_key)}: observed={r.observed_score:.3g}, "
            f"null95={r.null_p95 if pd.notna(r.null_p95) else np.nan:.3g}, p={ptxt}, pass={r.passes_null_p05}; {r.note}"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- The within-cancer low-state counterfactual is an excess-representation model, not a treatment effect.",
        "- TCGA PAF scores use existing stage/purity-adjusted cancer-stratified Cox coefficients and within-cancer top-quartile/IQR contrasts.",
        "- NCI phase-of-care validation uses per-patient annualized phase costs, so terminal/continuing ratios are validation signals rather than national aggregate cost estimates.",
        "- If a state is high only under observed cancer-site weighting but not under equal-site or within-cancer counterfactual scoring, treat it as composition-driven.",
        "",
    ]
    (OUT / "ecosystem_counterfactual_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    composition = composition_balanced_scores()
    _terms, cf_scores, match_summary = within_cancer_counterfactual_scores()
    phase = phase_of_care_cost_scores()
    _paf_terms, paf = tcga_paf_counterfactual()
    neg = negative_controls(cf_scores, composition, paf)
    write_summary(composition, cf_scores, paf, phase, neg, match_summary)
    print(f"wrote {OUT / 'ecosystem_state_composition_balanced_scores.csv'}")
    print(f"wrote {OUT / 'ecosystem_state_within_cancer_counterfactual_scores.csv'}")
    print(f"wrote {OUT / 'ecosystem_state_tcga_paf_counterfactual.csv'}")
    print(f"wrote {OUT / 'ecosystem_state_phase_of_care_cost_scores.csv'}")
    print(f"wrote {OUT / 'ecosystem_state_negative_control_summary.csv'}")
    print(f"wrote {OUT / 'ecosystem_counterfactual_validation_summary.md'}")


if __name__ == "__main__":
    main()
