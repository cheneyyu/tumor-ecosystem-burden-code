#!/usr/bin/env python3
"""Build pan-cancer ecosystem burden tables from Zenodo 10651059.

This is the first non-TAM-specific rebuild for proj_eco.  It uses the
decompressed atlas and NMF h5ad files copied from the Zenodo 10651059 mirror,
then connects sample/cancer-level ecosystem representation to public
GLOBOCAN burden estimates and NCI modeled US cost estimates.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pancancer_ecosystem_zenodo10651059"
ATLAS = RAW / "atlas_dataset"
NMF = RAW / "NMF_h5ad"
BURDEN = ROOT / "data" / "raw" / "gbd_globocan" / "globocan_2022_burden_long.csv"
COST = ROOT / "data" / "raw" / "costs" / "nci_cost_tables_long.csv"
WHO_GHE = ROOT / "data" / "raw" / "who_ghe"
OUT = ROOT / "data" / "processed" / "pancancer_ecosystem"

COMPARTMENTS = [
    "Malignant epithelial",
    "Non-malignant epithelial",
    "T cell",
    "NK cell",
    "B cell",
    "Plasma cell",
    "Macrophage",
    "DC/pDC",
    "Fibroblast",
    "Endothelial",
    "Mast",
    "Other",
]

TME_COMPARTMENTS = [
    "T cell",
    "NK cell",
    "B cell",
    "Plasma cell",
    "Macrophage",
    "DC/pDC",
    "Fibroblast",
    "Endothelial",
    "Mast",
]

NMF_FILES = {
    "epi_NMF.h5ad": "epithelial",
    "myl_NMF.h5ad": "myeloid",
    "tnk_NMF.h5ad": "T_NK",
    "b_NMF.h5ad": "B_plasma",
    "mesenchymal_NMF.h5ad": "mesenchymal",
}

NMF_TO_ATLAS_COMPARTMENTS = {
    "epithelial": ["Malignant epithelial"],
    "myeloid": ["Macrophage", "DC/pDC", "Mast"],
    "T_NK": ["T cell", "NK cell"],
    "B_plasma": ["B cell", "Plasma cell"],
    "mesenchymal": ["Fibroblast", "Endothelial"],
}

ATLAS_TO_GLOBOCAN = {
    "ALL": "Leukaemia",
    "BLCA": "Bladder",
    "BRCA": "Breast",
    "CHOL": "Liver",
    "CLL": "Leukaemia",
    "CRC": "Colorectum",
    "GBM": "Brain, CNS",
    "HCC": "Liver",
    "HNSC": "Lip, oral cavity",
    "LC": "Lung",
    "LGG": "Brain, CNS",
    "MEL": "Melanoma of skin",
    "MM": "Multiple myeloma",
    "NB": "Oth. specified",
    "NET": "Pancreas",
    "NHL": "NHL",
    "OV": "Ovary",
    "PAAD": "Pancreas",
    "PRAD": "Prostate",
    "RCC": "Kidney",
    "SARC-EWING": "Oth. specified",
    "SARC-OST": "Oth. specified",
    "SARC-RHAB": "Oth. specified",
    "SARC-SYN": "Oth. specified",
    "SSCC": "Lip, oral cavity",
    "STAD": "Stomach",
    "THCA": "Thyroid",
    "UCEC": "Corpus uteri",
    "UVM": "Melanoma of skin",
    "WILM": "Kidney",
}

ATLAS_TO_NCI = {
    "ALL": "Leukemia",
    "BLCA": "Bladder",
    "BRCA": "Female Breast",
    "CLL": "Leukemia",
    "CRC": "Colorectal",
    "GBM": "Brain",
    "HNSC": "Oral Cavity",
    "LC": "Lung",
    "LGG": "Brain",
    "MEL": "Melanoma",
    "MM": "Myeloma",
    "NHL": "Non-Hodgkin Lymphoma",
    "OV": "Ovary",
    "PRAD": "Prostate",
    "RCC": "Kidney",
    "SSCC": "Oral Cavity",
    "THCA": "Thyroid",
    "UCEC": "Uterus",
    "UVM": "Melanoma",
    "WILM": "Kidney",
}

WHO_GHE_TO_GLOBOCAN = {
    "Mouth and oropharynx cancers": "Lip, oral cavity",
    "Stomach cancer": "Stomach",
    "Colon and rectum cancers": "Colorectum",
    "Liver cancer": "Liver",
    "Pancreas cancer": "Pancreas",
    "Trachea, bronchus, lung cancers": "Lung",
    "Melanoma and other skin cancers": "Melanoma of skin",
    "Breast cancer": "Breast",
    "Corpus uteri cancer": "Corpus uteri",
    "Ovary cancer": "Ovary",
    "Prostate cancer": "Prostate",
    "Kidney cancer": "Kidney",
    "Bladder cancer": "Bladder",
    "Brain and nervous system cancers": "Brain, CNS",
    "Thyroid cancer": "Thyroid",
    "Leukaemia": "Leukaemia",
    "Other malignant neoplasms": "Oth. specified",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def mode_or_na(s: pd.Series) -> object:
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


def compartment_from_celltype(celltype: object, cnv_status: object) -> str:
    ct = str(celltype)
    cnv = str(cnv_status).lower()
    if ct == "Epithelial":
        return "Malignant epithelial" if cnv == "tumor" else "Non-malignant epithelial"
    if ct == "T cell":
        return "T cell"
    if ct == "NK cell":
        return "NK cell"
    if ct == "B cell":
        return "B cell"
    if ct == "Plasma cell":
        return "Plasma cell"
    if ct == "Macrophage":
        return "Macrophage"
    if ct in {"Dendritic cell", "pDC"}:
        return "DC/pDC"
    if ct == "Fibroblast":
        return "Fibroblast"
    if ct == "Endothelial":
        return "Endothelial"
    if ct == "Mast":
        return "Mast"
    return "Other"


def state_category(compartment: str, state: object) -> str:
    s = str(state)
    lo = s.lower()
    if compartment == "T_NK":
        if "treg" in lo:
            return "T/NK: Treg"
        if "exhausted" in lo:
            return "T/NK: exhausted CD8"
        if "nk-cell" in lo or "nk cell" in lo:
            return "T/NK: NK"
        if "tfh" in lo or "th17" in lo or "ilc3" in lo:
            return "T/NK: helper/ILC"
        if "resident" in lo or "intraepithelial" in lo or "cd8" in lo:
            return "T/NK: tissue CD8"
        if "proliferating" in lo or "cell cycling" in lo:
            return "T/NK: proliferating"
        if "interferon" in lo:
            return "T/NK: IFN"
        return "T/NK: other"
    if compartment == "B_plasma":
        if "plasma" in lo:
            return "B/plasma: plasma"
        if "germinal" in lo or "mature" in lo:
            return "B/plasma: mature/GC"
        if "precursor" in lo or "lymphoblast" in lo:
            return "B/plasma: precursor"
        if "cell cycling" in lo:
            return "B/plasma: cycling"
        return "B/plasma: other"
    if compartment == "mesenchymal":
        if "endothelial" in lo or "aerocyte" in lo:
            return "Mesenchymal: endothelial"
        if "lymphatic" in lo:
            return "Mesenchymal: lymphatic"
        if "pericyte" in lo:
            return "Mesenchymal: pericyte"
        if "desmoplastic" in lo or "myofibroblast" in lo:
            return "Mesenchymal: myofibro/desmoplastic"
        if "inflammatory" in lo or "ccl19" in lo:
            return "Mesenchymal: inflammatory fibroblast"
        if "wound" in lo or "fibroblast" in lo:
            return "Mesenchymal: fibroblast"
        if "cell cycling" in lo:
            return "Mesenchymal: cycling"
        return "Mesenchymal: other"
    if compartment == "myeloid":
        if "macrophage" in lo or "monocyte" in lo or "phagocyte" in lo:
            return "Myeloid: macrophage/TAM"
        if "dc" in lo or "dendritic" in lo or "langerhans" in lo or "pdc" in lo:
            return "Myeloid: DC/pDC"
        if "mast" in lo or "neutrophil" in lo:
            return "Myeloid: mast/neutrophil"
        if "activation" in lo or "isg" in lo or "heat shock" in lo or "interferon" in lo:
            return "Myeloid: activation/stress"
        if "cell cycling" in lo:
            return "Myeloid: cycling"
        return "Myeloid: other"
    if compartment == "epithelial":
        if "cell cycling" in lo:
            return "Epithelial: cycling"
        if "squamous" in lo or "basal" in lo:
            return "Epithelial: basal/squamous"
        if "glandular" in lo or "ductal" in lo or "luminal" in lo:
            return "Epithelial: glandular/luminal"
        if "stress" in lo:
            return "Epithelial: stress"
        return "Epithelial: lineage/tumor"
    return f"{compartment}: other"


def read_atlas_sample_abundance(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_path = OUT / "ecosystem_sample_compartment_fractions_long.csv"
    wide_path = OUT / "ecosystem_sample_compartment_fractions_wide.csv"
    manifest_path = OUT / "ecosystem_atlas_file_manifest.csv"
    if sample_path.exists() and wide_path.exists() and manifest_path.exists() and not force:
        return pd.read_csv(sample_path), pd.read_csv(wide_path), pd.read_csv(manifest_path)

    count_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    files = sorted(ATLAS.glob("*.h5ad"))
    for i, path in enumerate(files, 1):
        print(f"[atlas {i}/{len(files)}] {path.name}", flush=True)
        a = ad.read_h5ad(path, backed="r")
        cols = ["Dataset", "Organ_origin", "Sample", "Patient", "Tissue", "Cancer type", "cnv_status", "Celltype"]
        obs = a.obs[cols].copy()
        obs["Tissue"] = obs["Tissue"].astype(str)
        obs["Cancer type"] = obs["Cancer type"].astype(str)
        obs["Celltype"] = obs["Celltype"].astype(str)
        obs["cnv_status"] = obs["cnv_status"].astype(str)
        obs["compartment"] = [compartment_from_celltype(c, z) for c, z in zip(obs["Celltype"], obs["cnv_status"])]
        tumor = obs[obs["Tissue"].eq("Tumor")].copy()
        tumor["sample_key"] = tumor["Dataset"].astype(str) + "|" + tumor["Sample"].astype(str)
        group_cols = ["sample_key", "Dataset", "Organ_origin", "Sample", "Patient", "Cancer type", "compartment"]
        counts = tumor.groupby(group_cols, observed=True).size().reset_index(name="n_cells")
        count_rows.append(counts)
        manifest_rows.append(
            {
                "dataset_file": path.name,
                "n_obs": int(a.n_obs),
                "n_vars": int(a.n_vars),
                "n_tumor_cells": int(len(tumor)),
                "n_samples_total": int(obs["Sample"].nunique()),
                "n_tumor_samples": int(tumor["sample_key"].nunique()),
                "organ_origin_values": "|".join(sorted(map(str, obs["Organ_origin"].dropna().unique()))),
                "tumor_cancer_type_values": "|".join(sorted(map(str, tumor["Cancer type"].dropna().unique()))),
            }
        )
        a.file.close()

    counts = pd.concat(count_rows, ignore_index=True)
    counts["globocan_label"] = counts["Cancer type"].map(ATLAS_TO_GLOBOCAN)
    counts["nci_cost_site"] = counts["Cancer type"].map(ATLAS_TO_NCI)
    counts = counts[counts["globocan_label"].notna()].copy()
    totals = counts.groupby("sample_key", observed=True)["n_cells"].transform("sum")
    tme_totals = counts[counts["compartment"].isin(TME_COMPARTMENTS)].groupby("sample_key", observed=True)["n_cells"].sum()
    counts["n_tumor_cells_total"] = totals
    counts["n_tme_cells_total"] = counts["sample_key"].map(tme_totals).fillna(0).astype(int)
    counts["frac_all_tumor"] = counts["n_cells"] / counts["n_tumor_cells_total"]
    counts["frac_tme"] = np.where(
        counts["compartment"].isin(TME_COMPARTMENTS) & (counts["n_tme_cells_total"] > 0),
        counts["n_cells"] / counts["n_tme_cells_total"],
        np.nan,
    )

    # Complete missing compartment rows per sample so sample means treat absence as zero.
    meta_cols = ["sample_key", "Dataset", "Organ_origin", "Sample", "Patient", "Cancer type", "globocan_label", "nci_cost_site", "n_tumor_cells_total", "n_tme_cells_total"]
    sample_meta = counts[meta_cols].drop_duplicates("sample_key")
    grid = sample_meta.assign(_k=1).merge(pd.DataFrame({"compartment": COMPARTMENTS, "_k": 1}), on="_k").drop(columns="_k")
    counts = grid.merge(
        counts[["sample_key", "compartment", "n_cells", "frac_all_tumor", "frac_tme"]],
        on=["sample_key", "compartment"],
        how="left",
    )
    counts["n_cells"] = counts["n_cells"].fillna(0).astype(int)
    counts["frac_all_tumor"] = counts["frac_all_tumor"].fillna(0.0)
    counts["frac_tme"] = np.where(
        counts["compartment"].isin(TME_COMPARTMENTS) & (counts["n_tme_cells_total"] > 0),
        counts["frac_tme"].fillna(0.0),
        np.nan,
    )

    wide = counts.pivot_table(
        index=["sample_key", "Dataset", "Organ_origin", "Sample", "Patient", "Cancer type", "globocan_label", "nci_cost_site", "n_tumor_cells_total", "n_tme_cells_total"],
        columns="compartment",
        values="frac_all_tumor",
        fill_value=0,
        observed=True,
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={c: f"frac_all__{c}" for c in COMPARTMENTS if c in wide.columns})

    counts.to_csv(sample_path, index=False)
    wide.to_csv(wide_path, index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    return counts, wide, manifest


def cancer_compartment_means(counts: pd.DataFrame) -> pd.DataFrame:
    out_path = OUT / "ecosystem_cancer_compartment_means.csv"
    if out_path.exists():
        return pd.read_csv(out_path)
    means = (
        counts.groupby(["Cancer type", "globocan_label", "nci_cost_site", "compartment"], dropna=False, observed=True)
        .agg(
            n_samples=("sample_key", "nunique"),
            n_cells=("n_cells", "sum"),
            mean_frac_all_tumor=("frac_all_tumor", "mean"),
            median_frac_all_tumor=("frac_all_tumor", "median"),
            mean_frac_tme=("frac_tme", "mean"),
            median_frac_tme=("frac_tme", "median"),
        )
        .reset_index()
    )
    means.to_csv(out_path, index=False)
    return means


def load_burden() -> pd.DataFrame:
    burden = pd.read_csv(BURDEN)
    burden = burden[burden["measure"].isin(["incidence", "mortality", "prevalence"])].copy()
    burden = burden[burden["cancer_short_label"].notna()].copy()
    burden["burden_total"] = pd.to_numeric(burden["total"], errors="coerce")
    burden["source"] = burden.get("source", "GLOBOCAN 2022")
    keep = [
        "source",
        "location",
        "measure",
        "cancer_short_label",
        "burden_total",
    ]
    return pd.concat([burden[keep], load_who_ghe_burden()], ignore_index=True)


def _who_cause_name(row: pd.Series) -> str | None:
    vals = []
    for value in row.iloc[1:6]:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text.endswith(".") and text[:-1].isdigit():
            continue
        if text in {"I.", "II.", "III.", "A.", "B.", "C."}:
            continue
        vals.append(text)
    if not vals:
        return None
    return max(vals, key=len)


def _read_who_ghe_global(path: Path, measure: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for sheet in ["Global 2021"]:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        for _, row in df.iterrows():
            cause = _who_cause_name(row)
            label = WHO_GHE_TO_GLOBOCAN.get(cause)
            if not label:
                continue
            rows.append(
                {
                    "source": "WHO GHE 2021",
                    "location": "Global",
                    "measure": measure,
                    "cancer_short_label": label,
                    "burden_total": pd.to_numeric(row.iloc[6], errors="coerce"),
                    "who_ghe_cause": cause,
                    "year": 2021,
                }
            )
    return pd.DataFrame(rows)


def _read_who_ghe_country(path: Path, measure: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="All ages", header=None)
    iso_row = 7
    wanted = {"CHN": "China", "USA": "United States"}
    iso_to_col = {str(v).strip(): i for i, v in enumerate(df.iloc[iso_row]) if str(v).strip() in wanted}
    rows = []
    for _, row in df.iterrows():
        if str(row.iloc[0]).strip() != "Persons":
            continue
        cause = _who_cause_name(row)
        label = WHO_GHE_TO_GLOBOCAN.get(cause)
        if not label:
            continue
        for iso, location in wanted.items():
            col = iso_to_col.get(iso)
            if col is None:
                continue
            value_thousand = pd.to_numeric(row.iloc[col], errors="coerce")
            rows.append(
                {
                    "source": "WHO GHE 2021",
                    "location": location,
                    "measure": measure,
                    "cancer_short_label": label,
                    "burden_total": value_thousand * 1000,
                    "who_ghe_cause": cause,
                    "year": 2021,
                }
            )
    return pd.DataFrame(rows)


def load_who_ghe_burden() -> pd.DataFrame:
    cache = OUT / "who_ghe_2021_cancer_burden_long.csv"
    if cache.exists():
        return pd.read_csv(cache)

    parts = [
        _read_who_ghe_global(WHO_GHE / "ghe2021_daly_global_new.xlsx", "daly"),
        _read_who_ghe_global(WHO_GHE / "ghe2021_yll_global_new.xlsx", "yll"),
        _read_who_ghe_country(WHO_GHE / "ghe2021_daly_bycountry_2021.xlsx", "daly"),
        _read_who_ghe_country(WHO_GHE / "ghe2021_yll_bycountry_2021.xlsx", "yll"),
    ]
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
    if not out.empty:
        out = out[out["burden_total"].notna()].copy()
        out.to_csv(cache, index=False)
    return out


def load_cost() -> pd.DataFrame:
    cost = pd.read_csv(COST)
    cost = cost[(cost["table_index"].eq(1)) & cost["metric"].astype(str).eq("2020")].copy()
    cost["cost_2020_billion_usd"] = pd.to_numeric(cost["value"], errors="coerce")
    cost = cost[cost["cost_2020_billion_usd"].notna()].copy()
    return cost[["cancer_site", "cost_2020_billion_usd"]]


def burden_scores_for_features(
    cancer_means: pd.DataFrame,
    feature_cols: list[str],
    feature_name_col: str,
    abundance_col: str,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    burden = load_burden()
    terms = cancer_means.merge(burden, left_on="globocan_label", right_on="cancer_short_label", how="inner")
    terms["weighted_representation"] = terms[abundance_col] * terms["burden_total"]
    terms_out = OUT / f"{prefix}_globocan_burden_terms.csv"
    terms.to_csv(terms_out, index=False)
    scores = (
        terms.groupby(["source", "location", "measure", feature_name_col], observed=True)
        .agg(
            weighted_score=("weighted_representation", "sum"),
            mean_abundance_across_mapped_cancers=(abundance_col, "mean"),
            n_mapped_cancers=("globocan_label", "nunique"),
            total_burden_covered=("burden_total", "sum"),
        )
        .reset_index()
    )
    scores["rank_desc"] = scores.groupby(["location", "measure"], observed=True)["weighted_score"].rank(method="min", ascending=False)
    scores.to_csv(OUT / f"{prefix}_globocan_burden_scores.csv", index=False)
    return scores, terms


def cost_scores_for_features(
    cancer_means: pd.DataFrame,
    feature_name_col: str,
    abundance_col: str,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost = load_cost()
    terms = cancer_means.merge(cost, left_on="nci_cost_site", right_on="cancer_site", how="inner")
    terms["weighted_cost_score_billion_usd"] = terms[abundance_col] * terms["cost_2020_billion_usd"]
    terms.to_csv(OUT / f"{prefix}_nci_cost_terms.csv", index=False)
    scores = (
        terms.groupby(feature_name_col, observed=True)
        .agg(
            cost_weighted_score_billion_usd=("weighted_cost_score_billion_usd", "sum"),
            mean_abundance_across_mapped_cancers=(abundance_col, "mean"),
            n_mapped_cancers=("nci_cost_site", "nunique"),
            total_cost_covered_billion_usd=("cost_2020_billion_usd", "sum"),
        )
        .reset_index()
    )
    scores["rank_desc"] = scores["cost_weighted_score_billion_usd"].rank(method="min", ascending=False)
    scores.to_csv(OUT / f"{prefix}_nci_cost_scores.csv", index=False)
    return scores, terms


def read_nmf_state_fractions(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_path = OUT / "nmf_state_fraction_by_sample_long.csv"
    cancer_path = OUT / "nmf_state_representation_by_cancer.csv"
    if long_path.exists() and cancer_path.exists() and not force:
        return pd.read_csv(long_path), pd.read_csv(cancer_path)

    rows = []
    for fname, compartment in NMF_FILES.items():
        path = NMF / fname
        print(f"[nmf] {fname}", flush=True)
        a = ad.read_h5ad(path, backed="r")
        obs = a.obs[["Patient_Organ_Tissue", "Tissue", "Organ_origin", "Patient", "Dataset", "CancerAbbr", "Cell_state"]].copy()
        obs = obs[obs["Tissue"].astype(str).eq("Tumor")].copy()
        obs["nmf_compartment"] = compartment
        obs["globocan_label"] = obs["CancerAbbr"].map(ATLAS_TO_GLOBOCAN)
        obs["nci_cost_site"] = obs["CancerAbbr"].map(ATLAS_TO_NCI)
        obs = obs[obs["globocan_label"].notna()].copy()
        obs["state_category"] = [state_category(compartment, s) for s in obs["Cell_state"]]
        counts = (
            obs.groupby(
                [
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
                ],
                dropna=False,
                observed=True,
            )
            .size()
            .reset_index(name="n_modules")
        )
        denom = counts.groupby(["Patient_Organ_Tissue", "nmf_compartment"], observed=True)["n_modules"].transform("sum")
        counts["frac_in_nmf_compartment"] = counts["n_modules"] / denom
        rows.append(counts)
        a.file.close()
    long = pd.concat(rows, ignore_index=True)
    long["state_key"] = long["nmf_compartment"].astype(str) + " | " + long["Cell_state"].astype(str)
    long.to_csv(long_path, index=False)

    cancer_state = (
        long.groupby(["CancerAbbr", "globocan_label", "nci_cost_site", "nmf_compartment", "Cell_state", "state_category", "state_key"], dropna=False, observed=True)
        .agg(
            n_samples=("Patient_Organ_Tissue", "nunique"),
            n_modules=("n_modules", "sum"),
            mean_frac_in_nmf_compartment=("frac_in_nmf_compartment", "mean"),
            median_frac_in_nmf_compartment=("frac_in_nmf_compartment", "median"),
        )
        .reset_index()
    )
    # Convert within-compartment NMF state fractions to a rough total tumor
    # representation by multiplying by atlas-derived compartment abundance.
    counts, _, _ = read_atlas_sample_abundance()
    comp_means = cancer_compartment_means(counts)
    comp_pivot = comp_means.pivot_table(
        index=["Cancer type", "globocan_label", "nci_cost_site"],
        columns="compartment",
        values="mean_frac_all_tumor",
        observed=True,
    ).reset_index()
    comp_pivot.columns.name = None
    for nmf_comp, comps in NMF_TO_ATLAS_COMPARTMENTS.items():
        present = [c for c in comps if c in comp_pivot.columns]
        comp_pivot[f"atlas_frac_for_{nmf_comp}"] = comp_pivot[present].sum(axis=1) if present else np.nan
    comp_cols = ["Cancer type", "globocan_label"] + [f"atlas_frac_for_{x}" for x in NMF_TO_ATLAS_COMPARTMENTS]
    cancer_state = cancer_state.merge(comp_pivot[comp_cols], left_on=["CancerAbbr", "globocan_label"], right_on=["Cancer type", "globocan_label"], how="left")
    cancer_state["atlas_compartment_frac_all_tumor"] = [
        row.get(f"atlas_frac_for_{row['nmf_compartment']}", np.nan) for _, row in cancer_state.iterrows()
    ]
    cancer_state["state_representation_all_tumor"] = cancer_state["mean_frac_in_nmf_compartment"] * cancer_state["atlas_compartment_frac_all_tumor"]
    cancer_state.to_csv(cancer_path, index=False)
    return long, cancer_state


def variance_decomposition(df: pd.DataFrame, feature_col: str, value_col: str, cancer_col: str, min_n: int = 10) -> pd.DataFrame:
    rows = []
    for feature, sub in df[[feature_col, value_col, cancer_col]].dropna().groupby(feature_col, observed=True):
        if len(sub) < min_n or sub[value_col].nunique() < 3:
            continue
        vals = pd.to_numeric(sub[value_col], errors="coerce")
        ok = vals.notna()
        sub = sub.loc[ok].copy()
        vals = vals.loc[ok]
        if len(sub) < min_n:
            continue
        grand = vals.mean()
        total_ss = float(((vals - grand) ** 2).sum())
        if total_ss <= 0:
            continue
        cancer_mean = sub.groupby(cancer_col, observed=True)[value_col].transform("mean")
        between_ss = float(((cancer_mean - grand) ** 2).sum())
        within_ss = float(((vals - cancer_mean) ** 2).sum())
        rows.append(
            {
                feature_col: feature,
                "n_samples": int(len(sub)),
                "n_cancers": int(sub[cancer_col].nunique()),
                "total_variance": float(vals.var(ddof=1)),
                "between_cancer_variance_share": between_ss / total_ss,
                "within_cancer_variance_share": within_ss / total_ss,
            }
        )
    return pd.DataFrame(rows)


def leave_one_cancer_out(terms: pd.DataFrame, feature_col: str, score_col: str, prefix: str) -> pd.DataFrame:
    base = terms.groupby(feature_col, observed=True)[score_col].sum().sort_values(ascending=False)
    top_features = base.index.tolist()
    rows = []
    cancer_col = "globocan_label"
    for cancer in sorted(terms[cancer_col].dropna().unique()):
        sub = terms[terms[cancer_col] != cancer]
        scores = sub.groupby(feature_col, observed=True)[score_col].sum()
        ranks = scores.rank(method="min", ascending=False)
        for feature in top_features:
            rows.append(
                {
                    feature_col: feature,
                    "removed_cancer": cancer,
                    "score_without_cancer": float(scores.get(feature, 0.0)),
                    "rank_without_cancer": float(ranks.get(feature, np.nan)),
                    "base_rank": float(base.rank(method="min", ascending=False).get(feature)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"{prefix}_leave_one_cancer_out.csv", index=False)
    return out


def run() -> None:
    setup()
    counts, wide, manifest = read_atlas_sample_abundance()
    comp_means = cancer_compartment_means(counts)
    comp_scores, comp_terms = burden_scores_for_features(
        comp_means,
        COMPARTMENTS,
        "compartment",
        "mean_frac_all_tumor",
        "major_compartment",
    )
    cost_scores_for_features(comp_means, "compartment", "mean_frac_all_tumor", "major_compartment")

    nmf_long, nmf_cancer = read_nmf_state_fractions()
    state_scores, state_terms = burden_scores_for_features(
        nmf_cancer,
        [],
        "state_key",
        "state_representation_all_tumor",
        "nmf_state",
    )
    cost_scores_for_features(nmf_cancer, "state_key", "state_representation_all_tumor", "nmf_state")

    comp_var = variance_decomposition(counts, "compartment", "frac_all_tumor", "globocan_label", min_n=20)
    comp_var.to_csv(OUT / "major_compartment_variance_decomposition.csv", index=False)
    nmf_var = variance_decomposition(nmf_long, "state_key", "frac_in_nmf_compartment", "globocan_label", min_n=30)
    nmf_var.to_csv(OUT / "nmf_state_variance_decomposition.csv", index=False)

    global_mort_comp = comp_terms[(comp_terms["location"].eq("Global")) & (comp_terms["measure"].eq("mortality"))].copy()
    leave_one_cancer_out(global_mort_comp, "compartment", "weighted_representation", "major_compartment_global_mortality")
    global_mort_state = state_terms[(state_terms["location"].eq("Global")) & (state_terms["measure"].eq("mortality"))].copy()
    leave_one_cancer_out(global_mort_state, "state_key", "weighted_representation", "nmf_state_global_mortality")

    write_summary(counts, manifest, comp_scores, state_scores, comp_var, nmf_var)


def top_lines(df: pd.DataFrame, label_col: str, value_col: str, n: int = 8) -> list[str]:
    out = []
    for r in df.sort_values(value_col, ascending=False).head(n).itertuples(index=False):
        label = getattr(r, label_col)
        val = getattr(r, value_col)
        mapped = getattr(r, "n_mapped_cancers", np.nan)
        out.append(f"- {label}: {val:,.1f} (mapped cancers={mapped})")
    return out


def write_summary(
    counts: pd.DataFrame,
    manifest: pd.DataFrame,
    comp_scores: pd.DataFrame,
    state_scores: pd.DataFrame,
    comp_var: pd.DataFrame,
    nmf_var: pd.DataFrame,
) -> None:
    global_mort_comp = comp_scores[(comp_scores["location"].eq("Global")) & (comp_scores["measure"].eq("mortality"))].copy()
    global_mort_state = state_scores[(state_scores["location"].eq("Global")) & (state_scores["measure"].eq("mortality"))].copy()
    lines = [
        "Pan-cancer ecosystem burden rebuild using Zenodo 10651059",
        "",
        "Atlas scope",
        f"- Atlas h5ad files scanned: {manifest['dataset_file'].nunique()}",
        f"- Mapped tumor samples: {counts['sample_key'].nunique()}",
        f"- Mapped tumor cancer labels: {counts['Cancer type'].nunique()}",
        f"- Tumor cells represented in mapped samples: {int(counts.drop_duplicates('sample_key')['n_tumor_cells_total'].sum()):,}",
        "",
        "Top major compartments by global mortality-weighted representation",
        *top_lines(global_mort_comp, "compartment", "weighted_score"),
        "",
        "Top NMF cell states by global mortality-weighted representation",
        *top_lines(global_mort_state, "state_key", "weighted_score"),
        "",
        "Major compartment variance structure",
    ]
    for r in comp_var.sort_values("within_cancer_variance_share", ascending=False).head(8).itertuples(index=False):
        lines.append(f"- {r.compartment}: within-cancer share={r.within_cancer_variance_share:.2f}, cancers={r.n_cancers}, samples={r.n_samples}")
    lines.extend(["", "NMF state variance structure"])
    for r in nmf_var.sort_values("within_cancer_variance_share", ascending=False).head(8).itertuples(index=False):
        lines.append(f"- {r.state_key}: within-cancer share={r.within_cancer_variance_share:.2f}, cancers={r.n_cancers}, samples={r.n_samples}")
    lines.extend(
        [
            "",
            "Interpretation boundary",
            "- Scores are burden- or cost-weighted representation scores from public atlas and public burden/cost estimates.",
            "- Cost scores are modeled/inferred and are not observed cell-type medical spending.",
            "- NMF state scores multiply within-compartment NMF state representation by atlas-derived compartment abundance before burden weighting.",
        ]
    )
    (OUT / "pancancer_ecosystem_burden_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
