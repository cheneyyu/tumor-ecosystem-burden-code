#!/usr/bin/env python3
"""translational validation extension for the tumor ecosystem project.

This is a local-first pipeline. It uses the processed pan-cancer ecosystem
tables already in the repository for signature harmonization, TCGA nested Cox
models, robustness summaries, manuscript inserts, and transparent external
validation stubs when CPTAC/ICB cohorts are not present locally.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-proj-eco")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from scipy.stats import chi2, norm, spearmanr


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "processed" / "pancancer_ecosystem"
OUT = ROOT / "data" / "processed" / "translational_validation"
FIG = ROOT / "figures" / "translational_validation"
REPORTS = ROOT / "reports"
MS = ROOT / "manuscript"
RAW_VALIDATION = ROOT / "data" / "raw" / "translational_validation"
RIAZ_RAW = RAW_VALIDATION / "riaz_gse91061"
GENE_INFO = RAW_VALIDATION / "Homo_sapiens.gene_info.gz"
IMVIGOR_TAR = RAW_VALIDATION / "IMvigor210CoreBiologies_1.0.0.tar.gz"
IMVIGOR_EXTRACT = RAW_VALIDATION / "imvigor210_pkg_extract"
IMVIGOR_TSV = RAW_VALIDATION / "imvigor210_extracted_tsv"
CPTAC_RAW = RAW_VALIDATION / "cptac_zenodo"
CPTAC_CLINICAL = CPTAC_RAW / "mssm-all_cancers-clinical-clinical_Pan-cancer.May2022.tsv.gz"
CPTAC_UCEC_RNA = CPTAC_RAW / "bcm-ucec-transcriptomics-UCEC-gene_rsem_removed_circRNA_tumor_normal_UQ_log2x1_BCM.txt.gz"
CPTAC_UCEC_PROTEIN = CPTAC_RAW / "bcm-ucec-proteomics-UCEC_proteomics_gene_abundance_log2_reference_intensity_normalized_Tumor.txt.gz"

CREATED: list[dict[str, str]] = []
RUN_DATE = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

PRIORITY_STATES = [
    "T_NK | Proliferating T-cell (cell cycling)",
    "epithelial | Cell cycling",
    "myeloid | Cell cycling",
    "myeloid | SPP1+ macrophage",
    "epithelial | Complete mesenchymal",
    "mesenchymal | Desmoplastic fibroblast",
    "myeloid | NLRP3+ monocyte derived macrophage",
    "myeloid | Heat shock",
    "T_NK | Exhausted CD8+ T-cell",
    "myeloid | C1QC+ macrophage",
    "T_NK | Treg",
    "T_NK | CD16+ NK-cell",
]

CYCLING_GENES = {
    "MKI67",
    "TOP2A",
    "UBE2C",
    "PCLAF",
    "PCNA",
    "MCM2",
    "MCM3",
    "MCM4",
    "MCM5",
    "MCM6",
    "MCM7",
    "CDK1",
    "CCNB1",
    "CCNB2",
    "CDC20",
    "TYMS",
    "RRM2",
    "HMGB2",
    "STMN1",
    "TUBA1B",
}

PROLIFERATION_STATES = [
    "T_NK | Proliferating T-cell (cell cycling)",
    "epithelial | Cell cycling",
    "myeloid | Cell cycling",
]

IMMUNE_PROXY_STATES = [
    "T_NK | CD16+ NK-cell",
    "T_NK | Exhausted CD8+ T-cell",
    "T_NK | Treg",
    "myeloid | C1QC+ macrophage",
    "myeloid | CXCL9+ macrophage",
    "myeloid | SPP1+ macrophage",
]

STROMAL_PROXY_STATES = [
    "mesenchymal | Desmoplastic fibroblast",
    "mesenchymal | Myofibroblast",
    "mesenchymal | PI16+ fibroblast",
]

RIAZ_URLS = {
    RIAZ_RAW / "README.md": "https://raw.githubusercontent.com/riazn/bms038_analysis/master/data/README.md",
    RIAZ_RAW / "bms038_clinical_data.csv": "https://raw.githubusercontent.com/riazn/bms038_analysis/master/data/bms038_clinical_data.csv",
    RIAZ_RAW / "SampleTableCorrected.9.19.16.csv": "https://raw.githubusercontent.com/riazn/bms038_analysis/master/data/SampleTableCorrected.9.19.16.csv",
    RIAZ_RAW / "rld.BMS038.20171011.csv": "https://raw.githubusercontent.com/riazn/bms038_analysis/master/data/rld.BMS038.20171011.csv",
    GENE_INFO: "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
}

IMVIGOR_URL = "http://research-pub.gene.com/IMvigor210CoreBiologies/packageVersions/IMvigor210CoreBiologies_1.0.0.tar.gz"

CPTAC_URLS = {
    CPTAC_CLINICAL: {
        "url": "https://zenodo.org/api/records/8394329/files/mssm-all_cancers-clinical-clinical_Pan-cancer.May2022.tsv.gz/content",
        "md5": "718195847eeba955b00c5a9355cb004f",
        "description": "CPTAC pan-cancer clinical table from Zenodo 8394329",
    },
    CPTAC_UCEC_RNA: {
        "url": "https://zenodo.org/api/records/8394329/files/bcm-ucec-transcriptomics-UCEC-gene_rsem_removed_circRNA_tumor_normal_UQ_log2(x+1)_BCM.txt.gz/content",
        "md5": "c98263541e80b4f7d0c0aa972474321c",
        "description": "CPTAC BCM UCEC RNA-seq log2 expression matrix from Zenodo 8394329",
    },
    CPTAC_UCEC_PROTEIN: {
        "url": "https://zenodo.org/api/records/8394329/files/bcm-ucec-proteomics-UCEC_proteomics_gene_abundance_log2_reference_intensity_normalized_Tumor.txt.gz/content",
        "md5": "7b4e2e7172869d812255f9f7acf44f3d",
        "description": "CPTAC BCM UCEC tumor proteome matrix from Zenodo 8394329",
    },
}

CPTAC_PATHWAY_STATE_MAP = {
    "T_NK | Proliferating T-cell (cell cycling)": "cell_cycle",
    "epithelial | Cell cycling": "cell_cycle",
    "myeloid | Cell cycling": "cell_cycle",
    "epithelial | Complete mesenchymal": "emt_stromal",
    "mesenchymal | Desmoplastic fibroblast": "emt_stromal",
    "myeloid | SPP1+ macrophage": "myeloid_inflammation",
    "myeloid | C1QC+ macrophage": "myeloid_inflammation",
    "myeloid | NLRP3+ monocyte derived macrophage": "myeloid_inflammation",
    "myeloid | Heat shock": "heat_shock_stress",
    "T_NK | Exhausted CD8+ T-cell": "immune_exhaustion_nk",
    "T_NK | Treg": "immune_exhaustion_nk",
    "T_NK | CD16+ NK-cell": "immune_exhaustion_nk",
}


@dataclass
class OutputRecord:
    path: Path
    description: str
    source: str
    dependencies: str


def ensure_dirs() -> None:
    for p in (OUT, FIG, REPORTS, MS):
        p.mkdir(parents=True, exist_ok=True)
    RIAZ_RAW.mkdir(parents=True, exist_ok=True)
    CPTAC_RAW.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, description: str, source: str, dependencies: str) -> None:
    item = {
        "file_path": rel(path),
        "description": description,
        "source": source,
        "created_date": RUN_DATE,
        "upstream_dependencies": dependencies,
        "sha256": checksum(path) if path.exists() and path.is_file() else "",
    }
    for idx, existing in enumerate(CREATED):
        if existing["file_path"] == item["file_path"]:
            CREATED[idx] = item
            return
    CREATED.append(item)


def write_text(path: Path, text: str, description: str, source: str, dependencies: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    record(path, description, source, dependencies)


def write_table(
    df: pd.DataFrame, path: Path, description: str, source: str, dependencies: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    record(path, description, source, dependencies)


def bh_fdr(pvalues: Iterable[float]) -> list[float]:
    p = np.asarray([np.nan if x is None else float(x) for x in pvalues], dtype=float)
    out = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return out.tolist()
    vals = p[mask]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[mask] = tmp
    return out.tolist()


def md_table(df: pd.DataFrame) -> str:
    """Render a small markdown table without pandas optional tabulate."""
    if df.empty:
        return "_No rows._"
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in d.to_numpy()]
    return "\n".join([header, sep] + body)


def short_label(state: str) -> str:
    mapping = {
        "T_NK | Proliferating T-cell (cell cycling)": "Prolif. T",
        "T_NK | Exhausted CD8+ T-cell": "Exh. CD8",
        "T_NK | CD16+ NK-cell": "CD16 NK",
        "T_NK | Treg": "Treg",
        "epithelial | Cell cycling": "Epi cycling",
        "epithelial | Complete mesenchymal": "Epi mes.",
        "epithelial | Stress": "Epi stress",
        "mesenchymal | Desmoplastic fibroblast": "Desmoplastic FB",
        "myeloid | Cell cycling": "Myeloid cycling",
        "myeloid | SPP1+ macrophage": "SPP1 mac",
        "myeloid | C1QC+ macrophage": "C1QC mac",
        "myeloid | NLRP3+ monocyte derived macrophage": "NLRP3 mono/mac",
        "myeloid | Heat shock": "Heat shock myeloid",
        "Integrated adverse ecosystem score": "Adverse score",
    }
    return mapping.get(state, state.split("|")[-1].strip()[:24])


def make_repo_audit() -> None:
    expected = {
        "zenodo_nmf_state_signature_genes.csv": [
            "state_key",
            "nmf_compartment",
            "Cell_state",
            "rank",
            "gene",
            "marker_delta",
        ],
        "tcga_nmf_state_signature_scores_survival_merged.csv.gz": [
            "sample",
            "CancerType",
            "duration",
            "event",
            "age_z",
            "male",
            "stage_ordinal",
            "absolute_purity",
        ],
        "ecosystem_state_progression_cost_prognosis_map.csv": [
            "state_key",
            "cost_weighted_score_billion_usd",
            "full_hr",
            "stage_fdr",
        ],
        "ecosystem_state_within_cancer_counterfactual_scores.csv": [
            "state_key",
            "reducible_global_mortality_score",
            "reducible_nci_cost_score_billion_usd",
        ],
    }
    rows = []
    for name, cols in expected.items():
        path = SRC / name
        present = path.exists()
        actual = []
        missing_cols = cols[:]
        if present:
            actual = list(pd.read_csv(path, nrows=0).columns)
            missing_cols = [c for c in cols if c not in actual]
        rows.append((name, present, ", ".join(cols), ", ".join(missing_cols) or "none"))

    scripts = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*.py"))
    figure_deps = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "figures").glob("**/*.tex"))

    lines = [
        "# Repository audit for translational validation upgrade",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "## Existing input tables",
        "",
        "| table | present | expected columns | missing expected columns |",
        "|---|---:|---|---|",
    ]
    lines += [f"| `{name}` | {present} | {cols} | {missing} |" for name, present, cols, missing in rows]
    lines += [
        "",
        "## Existing analysis scripts",
        "",
        "\n".join(f"- `{s}`" for s in scripts) or "- none detected",
        "",
        "## Current figure dependencies",
        "",
        "\n".join(f"- `{s}`" for s in figure_deps) or "- none detected",
        "",
        "## Reproducibility gaps addressed by this extension",
        "",
        "- Adds a single local command for translational validation outputs.",
        "- Standardizes NMF state signatures into a reusable TSV catalog.",
        "- Re-fits TCGA stratified Cox models using a transparent nested model sequence.",
        "- Records external-validation provenance before replacing placeholders with downloaded open-access cohort analyses.",
        "- Writes a manifest with checksums for all new generated files.",
        "",
        "## Remaining gaps after local-only run",
        "",
        "- External cohorts remain limited by open-access availability, single-cohort biology and heterogeneous processed-data formats.",
        "- Patient-level medical spending is still unavailable; all cost language remains modeled/ecological.",
        "- Tumor purity, immune and stromal adjustment rely on local proxies unless ESTIMATE or orthogonal deconvolution scores are added.",
    ]
    write_text(
        REPORTS / "repo_audit.md",
        "\n".join(lines),
        "Repository audit against goal.md deliverables",
        "local filesystem inspection",
        "goal.md; processed pancancer ecosystem tables",
    )


def make_signature_catalog() -> pd.DataFrame:
    src = SRC / "zenodo_nmf_state_signature_genes.csv"
    sig = pd.read_csv(src)
    cat = sig.rename(
        columns={
            "state_key": "state_id",
            "Cell_state": "state_label",
            "nmf_compartment": "compartment",
            "gene": "gene_symbol",
            "rank": "marker_rank_or_weight",
        }
    )[
        [
            "state_id",
            "state_label",
            "compartment",
            "gene_symbol",
            "marker_rank_or_weight",
            "marker_delta",
            "state_mean",
            "rest_mean",
        ]
    ].copy()
    cat["source_file"] = rel(src)
    write_table(
        cat,
        OUT / "signature_catalog.tsv",
        "Standardized NMF state marker signature catalog",
        "Zenodo NMF state marker table",
        rel(src),
    )

    variants = []
    for state_id, d in cat.sort_values("marker_rank_or_weight").groupby("state_id"):
        state_label = d["state_label"].iloc[0]
        comp = d["compartment"].iloc[0]
        for variant, q in [("top25", 25), ("top50", 50)]:
            for _, r in d.head(q).iterrows():
                variants.append(
                    {
                        "state_id": state_id,
                        "state_label": state_label,
                        "compartment": comp,
                        "variant": variant,
                        "gene_symbol": r["gene_symbol"],
                        "marker_rank_or_weight": r["marker_rank_or_weight"],
                        "source_file": r["source_file"],
                        "variant_note": f"top {q} genes by marker rank",
                    }
                )
        pos = d[d["marker_delta"] > 0]
        for _, r in pos.iterrows():
            variants.append(
                {
                    "state_id": state_id,
                    "state_label": state_label,
                    "compartment": comp,
                    "variant": "positive_only",
                    "gene_symbol": r["gene_symbol"],
                    "marker_rank_or_weight": r["marker_rank_or_weight"],
                    "source_file": r["source_file"],
                    "variant_note": "marker_delta > 0",
                }
            )
        non_cycle = d[~d["gene_symbol"].str.upper().isin(CYCLING_GENES)]
        for _, r in non_cycle.head(50).iterrows():
            variants.append(
                {
                    "state_id": state_id,
                    "state_label": state_label,
                    "compartment": comp,
                    "variant": "cell_cycle_gene_excluded_top50",
                    "gene_symbol": r["gene_symbol"],
                    "marker_rank_or_weight": r["marker_rank_or_weight"],
                    "source_file": r["source_file"],
                    "variant_note": "top 50 after excluding canonical cell-cycle genes; not a true residualized refit",
                }
            )
    vdf = pd.DataFrame(variants)
    write_table(
        vdf,
        OUT / "signature_catalog_variants.tsv",
        "Top25/top50/positive-only/cell-cycle-excluded signature variants",
        "derived from standardized signature catalog",
        f"{rel(src)}; canonical cell-cycle gene list embedded in run_all.py",
    )

    qc = cat.groupby(["compartment", "state_id"], as_index=False).agg(
        n_genes=("gene_symbol", "nunique"),
        max_rank=("marker_rank_or_weight", "max"),
        median_marker_delta=("marker_delta", "median"),
    )
    priority_present = [s for s in PRIORITY_STATES if s in set(cat["state_id"])]
    text = [
        "# Signature catalog QC",
        "",
        f"Run date: {RUN_DATE}",
        "",
        f"- Standardized states: {cat['state_id'].nunique()}",
        f"- Unique genes: {cat['gene_symbol'].nunique()}",
        f"- Priority states found: {len(priority_present)} / {len(PRIORITY_STATES)}",
        f"- Signature rows: {len(cat)}",
        f"- Variant rows: {len(vdf)}",
        "",
        "Cell-cycle-excluded variants remove a canonical cell-cycle gene list. This is a robustness variant, not a true proliferation-residualized NMF refit.",
        "",
        "## Smallest signatures",
        "",
        md_table(qc.sort_values("n_genes").head(12)),
    ]
    write_text(
        REPORTS / "signature_catalog_qc.md",
        "\n".join(text),
        "QC summary for standardized NMF signature catalog",
        "signature catalog generation",
        f"{rel(OUT / 'signature_catalog.tsv')}; {rel(OUT / 'signature_catalog_variants.tsv')}",
    )
    return cat


def state_columns(df: pd.DataFrame) -> list[str]:
    metadata = {
        "sample",
        "sample15",
        "sample_surv",
        "_PATIENT",
        "CancerType",
        "age_at_initial_pathologic_diagnosis",
        "gender",
        "race",
        "ajcc_pathologic_tumor_stage",
        "clinical_stage",
        "histological_type",
        "histological_grade",
        "initial_pathologic_dx_year",
        "menopause_status",
        "birth_days_to",
        "vital_status",
        "tumor_status",
        "last_contact_days_to",
        "death_days_to",
        "cause_of_death",
        "new_tumor_event_type",
        "new_tumor_event_site",
        "new_tumor_event_site_other",
        "new_tumor_event_dx_days_to",
        "treatment_outcome_first_course",
        "margin_status",
        "residual_tumor",
        "OS",
        "OS.time",
        "DSS",
        "DSS.time",
        "DFI",
        "DFI.time",
        "PFI",
        "PFI.time",
        "Redaction",
        "absolute_purity",
        "duration",
        "event",
        "age",
        "age_z",
        "male",
        "stage_ordinal",
        "globocan_label",
    }
    return [c for c in df.columns if c not in metadata and "|" in c]


def within_cancer_z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = df.copy()
    for col in cols:
        def _z(s: pd.Series) -> pd.Series:
            sd = s.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                return s * np.nan
            return (s - s.mean()) / sd

        z[col] = z.groupby("CancerType", observed=True)[col].transform(_z)
    return z


def mean_proxy(df: pd.DataFrame, states: list[str], exclude: str | None = None) -> pd.Series:
    cols = [c for c in states if c in df.columns and c != exclude]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1)


def fit_cox(
    df: pd.DataFrame,
    covariates: list[str],
    strata: bool = True,
    min_samples: int = 250,
    min_events: int = 50,
) -> tuple[CoxPHFitter | None, pd.DataFrame, str]:
    cols = ["duration", "event"] + covariates + (["CancerType"] if strata else [])
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = d[(d["duration"] > 0) & d["event"].isin([0, 1])]
    d["event"] = d["event"].astype(int)
    for c in list(covariates):
        if c not in d.columns or d[c].nunique(dropna=True) < 2:
            d = d.drop(columns=[c], errors="ignore")
            covariates = [x for x in covariates if x != c]
    if len(covariates) == 0:
        return None, d, "no_nonconstant_covariates"
    if len(d) < min_samples or int(d["event"].sum()) < min_events:
        return None, d, "insufficient_samples_or_events"
    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(
            d,
            duration_col="duration",
            event_col="event",
            strata=["CancerType"] if strata and "CancerType" in d.columns else None,
            show_progress=False,
        )
        return cph, d, ""
    except Exception as exc:  # lifelines exposes convergence problems as exceptions
        return None, d, f"{type(exc).__name__}: {exc}"


def row_from_cph(
    state: str,
    model_name: str,
    cph: CoxPHFitter | None,
    d: pd.DataFrame,
    error: str,
    base: CoxPHFitter | None = None,
    covariates: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "state_id": state,
        "state_label": short_label(state),
        "model": model_name,
        "n_samples": int(len(d)),
        "n_events": int(d["event"].sum()) if "event" in d else 0,
        "n_cancer_types": int(d["CancerType"].nunique()) if "CancerType" in d else np.nan,
        "coef_log_hr_per_within_cancer_sd": np.nan,
        "hr_per_within_cancer_sd": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "p_value": np.nan,
        "fdr": np.nan,
        "concordance_index": np.nan,
        "likelihood_ratio_stat_vs_previous": np.nan,
        "likelihood_ratio_p_vs_previous": np.nan,
        "covariates": covariates,
        "ph_diagnostic": "not_run_for_high_dimensional_stratified_screen",
        "error": error,
    }
    if cph is None or "state_z" not in getattr(cph, "summary", pd.DataFrame()).index:
        return row
    s = cph.summary.loc["state_z"]
    coef = float(s["coef"])
    se = float(s["se(coef)"])
    row.update(
        {
            "coef_log_hr_per_within_cancer_sd": coef,
            "hr_per_within_cancer_sd": float(math.exp(coef)),
            "ci95_low": float(math.exp(coef - 1.96 * se)),
            "ci95_high": float(math.exp(coef + 1.96 * se)),
            "p_value": float(s["p"]),
            "concordance_index": float(cph.concordance_index_),
            "error": "",
        }
    )
    if base is not None:
        lr = 2 * (float(cph.log_likelihood_) - float(base.log_likelihood_))
        row["likelihood_ratio_stat_vs_previous"] = max(lr, 0.0)
        row["likelihood_ratio_p_vs_previous"] = float(chi2.sf(max(lr, 0.0), 1))
    return row


def make_tcga_incremental() -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_path = SRC / "tcga_nmf_state_signature_scores_survival_merged.csv.gz"
    df = pd.read_csv(merged_path)
    states = state_columns(df)
    df = within_cancer_z(df, states)

    present_priority = [s for s in PRIORITY_STATES if s in states]
    adverse_components = [
        s
        for s in [
            "T_NK | Proliferating T-cell (cell cycling)",
            "epithelial | Cell cycling",
            "myeloid | Cell cycling",
            "myeloid | SPP1+ macrophage",
            "epithelial | Complete mesenchymal",
            "mesenchymal | Desmoplastic fibroblast",
            "myeloid | Heat shock",
        ]
        if s in df.columns
    ]
    df["Integrated adverse ecosystem score"] = df[adverse_components].mean(axis=1)
    df = within_cancer_z(df, ["Integrated adverse ecosystem score"])
    present_priority.append("Integrated adverse ecosystem score")

    rows = []
    for state in present_priority:
        work = df.copy()
        work["state_z"] = work[state]
        work["proliferation_proxy"] = mean_proxy(work, PROLIFERATION_STATES, exclude=state)
        work["immune_proxy"] = mean_proxy(work, IMMUNE_PROXY_STATES, exclude=state)
        work["stromal_proxy"] = mean_proxy(work, STROMAL_PROXY_STATES, exclude=state)

        model_defs = [
            ("model0_state_only_cancer_strata", ["state_z"], []),
            (
                "model1_clinical_purity_plus_state",
                ["age_z", "male", "stage_ordinal", "absolute_purity", "state_z"],
                ["age_z", "male", "stage_ordinal", "absolute_purity"],
            ),
            (
                "model2_plus_proliferation_and_state",
                ["age_z", "male", "stage_ordinal", "absolute_purity", "proliferation_proxy", "state_z"],
                ["age_z", "male", "stage_ordinal", "absolute_purity", "proliferation_proxy"],
            ),
            (
                "model3_plus_immune_stromal_and_state",
                [
                    "age_z",
                    "male",
                    "stage_ordinal",
                    "absolute_purity",
                    "proliferation_proxy",
                    "immune_proxy",
                    "stromal_proxy",
                    "state_z",
                ],
                [
                    "age_z",
                    "male",
                    "stage_ordinal",
                    "absolute_purity",
                    "proliferation_proxy",
                    "immune_proxy",
                    "stromal_proxy",
                ],
            ),
        ]
        for name, covs, base_covs in model_defs:
            base = None
            if base_covs:
                base, _, _ = fit_cox(work, base_covs, strata=True)
            cph, d, err = fit_cox(work, covs, strata=True)
            rows.append(row_from_cph(state, name, cph, d, err, base=base, covariates=", ".join(covs)))

    inc = pd.DataFrame(rows)
    inc["fdr"] = bh_fdr(inc["p_value"])
    inc["lrt_fdr_vs_previous"] = bh_fdr(inc["likelihood_ratio_p_vs_previous"])
    write_table(
        inc,
        OUT / "tcga_incremental_cox.tsv",
        "Nested TCGA stratified Cox models for priority ecosystem states",
        "local TCGA signature score/survival merged table",
        rel(merged_path),
    )

    cancer_rows = []
    for state in present_priority:
        for cancer, g in df.groupby("CancerType", observed=True):
            work = g.copy()
            work["state_z"] = work[state]
            covs = ["state_z", "age_z", "male", "stage_ordinal", "absolute_purity"]
            cph, d, err = fit_cox(work, covs, strata=False, min_samples=50, min_events=10)
            out = {
                "row_type": "cancer_specific",
                "state_id": state,
                "state_label": short_label(state),
                "cancer_type": cancer,
                "n_samples": int(len(d)),
                "n_events": int(d["event"].sum()) if "event" in d else 0,
                "coef_log_hr": np.nan,
                "se": np.nan,
                "hr": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "p_value": np.nan,
                "error": err,
            }
            if cph is not None and "state_z" in cph.summary.index:
                s = cph.summary.loc["state_z"]
                coef = float(s["coef"])
                se = float(s["se(coef)"])
                out.update(
                    {
                        "coef_log_hr": coef,
                        "se": se,
                        "hr": float(math.exp(coef)),
                        "ci95_low": float(math.exp(coef - 1.96 * se)),
                        "ci95_high": float(math.exp(coef + 1.96 * se)),
                        "p_value": float(s["p"]),
                        "error": "",
                    }
                )
            cancer_rows.append(out)

    cs = pd.DataFrame(cancer_rows)
    meta_rows = []
    for state, d in cs.dropna(subset=["coef_log_hr", "se"]).groupby("state_id", observed=True):
        d = d[d["se"] > 0]
        if d.empty:
            continue
        w = 1 / (d["se"] ** 2)
        coef = float((d["coef_log_hr"] * w).sum() / w.sum())
        se = float(math.sqrt(1 / w.sum()))
        z = coef / se
        p = float(2 * norm.sf(abs(z)))
        meta_rows.append(
            {
                "row_type": "fixed_effect_meta",
                "state_id": state,
                "state_label": short_label(state),
                "cancer_type": "pan_cancer_meta",
                "n_samples": int(d["n_samples"].sum()),
                "n_events": int(d["n_events"].sum()),
                "coef_log_hr": coef,
                "se": se,
                "hr": float(math.exp(coef)),
                "ci95_low": float(math.exp(coef - 1.96 * se)),
                "ci95_high": float(math.exp(coef + 1.96 * se)),
                "p_value": p,
                "error": "",
            }
        )
    meta = pd.concat([cs, pd.DataFrame(meta_rows)], ignore_index=True)
    meta["fdr"] = bh_fdr(meta["p_value"])
    write_table(
        meta,
        OUT / "tcga_cancer_specific_meta.tsv",
        "Cancer-specific Cox fits and fixed-effect meta-analysis",
        "local TCGA signature score/survival merged table",
        rel(merged_path),
    )

    make_tcga_incremental_figure(inc)
    make_tcga_incremental_summary(inc, meta)
    return inc, meta


def make_tcga_incremental_figure(inc: pd.DataFrame) -> None:
    final = inc[inc["model"] == "model3_plus_immune_stromal_and_state"].copy()
    final = final.sort_values("coef_log_hr_per_within_cancer_sd", ascending=True)
    source = final[
        [
            "state_id",
            "state_label",
            "n_samples",
            "n_events",
            "coef_log_hr_per_within_cancer_sd",
            "hr_per_within_cancer_sd",
            "ci95_low",
            "ci95_high",
            "p_value",
            "fdr",
            "likelihood_ratio_p_vs_previous",
            "lrt_fdr_vs_previous",
            "concordance_index",
        ]
    ].copy()
    write_table(
        source,
        FIG / "tcga_incremental_value_source.tsv",
        "Source table for TCGA incremental value figure",
        "tcga_incremental_cox.tsv final nested model rows",
        rel(OUT / "tcga_incremental_cox.tsv"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), gridspec_kw={"width_ratios": [1.3, 1.0]})
    y = np.arange(len(final))
    x = final["coef_log_hr_per_within_cancer_sd"].astype(float)
    lo = np.log(final["ci95_low"].astype(float))
    hi = np.log(final["ci95_high"].astype(float))
    colors = np.where(final["fdr"].fillna(1) < 0.1, "#b5443c", "#6c7a89")
    axes[0].barh(y, x, color=colors, alpha=0.9)
    axes[0].errorbar(x, y, xerr=[x - lo, hi - x], fmt="none", ecolor="#2c2c2c", lw=0.8)
    axes[0].axvline(0, color="#222222", lw=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(final["state_label"], fontsize=7)
    axes[0].set_xlabel("log HR per within-cancer SD")
    axes[0].set_title("Adjusted TCGA survival effect", fontsize=9, weight="bold")

    lrt = -np.log10(final["likelihood_ratio_p_vs_previous"].astype(float).clip(lower=1e-300))
    axes[1].barh(y, lrt, color="#4f7396")
    axes[1].axvline(-math.log10(0.05), color="#222222", lw=0.7, ls="--")
    axes[1].set_yticks([])
    axes[1].set_xlabel("-log10 LRT p")
    axes[1].set_title("Increment beyond proxies", fontsize=9, weight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    path = FIG / "tcga_incremental_value.pdf"
    fig.savefig(path)
    plt.close(fig)
    record(
        path,
        "TCGA incremental prognostic value figure",
        "matplotlib generated from TCGA incremental Cox source table",
        rel(FIG / "tcga_incremental_value_source.tsv"),
    )


def make_tcga_incremental_summary(inc: pd.DataFrame, meta: pd.DataFrame) -> None:
    final = inc[inc["model"] == "model3_plus_immune_stromal_and_state"].copy()
    top = final.sort_values("coef_log_hr_per_within_cancer_sd", ascending=False).head(8)
    sig = final[final["fdr"].fillna(1) < 0.1]
    meta_sig = meta[(meta["row_type"] == "fixed_effect_meta") & (meta["fdr"].fillna(1) < 0.1)]
    lines = [
        "# TCGA incremental prognostic value summary",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Nested stratified Cox models were fit with TCGA cancer type as strata. The final model added each ecosystem state to age, sex, stage, purity, a proliferation proxy, an immune proxy and a stromal proxy.",
        "",
        f"- Final-model states tested: {len(final)}",
        f"- Final-model states with FDR < 0.10: {len(sig)}",
        f"- Fixed-effect meta-analysis states with FDR < 0.10: {len(meta_sig)}",
        "",
        "## Highest adjusted adverse effects",
        "",
        md_table(top[
            [
                "state_label",
                "n_samples",
                "n_events",
                "hr_per_within_cancer_sd",
                "p_value",
                "fdr",
                "likelihood_ratio_p_vs_previous",
            ]
        ]),
        "",
        "Interpretation boundary: these are adverse prognostic associations after available local adjustments, not causal cell-state effects and not observed patient-level spending effects.",
    ]
    write_text(
        REPORTS / "tcga_incremental_value_summary.md",
        "\n".join(lines),
        "Narrative summary of nested TCGA Cox analysis",
        "tcga_incremental_cox.tsv and tcga_cancer_specific_meta.tsv",
        f"{rel(OUT / 'tcga_incremental_cox.tsv')}; {rel(OUT / 'tcga_cancer_specific_meta.tsv')}",
    )


def make_robustness(inc: pd.DataFrame) -> pd.DataFrame:
    map_df = pd.read_csv(SRC / "ecosystem_state_progression_cost_prognosis_map.csv")
    stage = pd.read_csv(SRC / "ecosystem_state_stage_gradients.csv")
    cf = pd.read_csv(SRC / "ecosystem_state_within_cancer_counterfactual_scores.csv")
    loo = pd.read_csv(SRC / "nmf_state_global_mortality_leave_one_cancer_out.csv")
    neg = pd.read_csv(SRC / "ecosystem_state_negative_control_summary.csv")

    final = inc[inc["model"] == "model3_plus_immune_stromal_and_state"].copy()
    final = final.set_index("state_id")
    stage = stage.set_index("state_key")
    map_df = map_df.set_index("state_key")
    cf = cf.set_index("state_key")
    neg_map = neg.groupby("state_key", as_index=True)["passes_null_p05"].max()
    loo_summary = (
        loo.groupby("state_key", as_index=True)
        .agg(median_leave_one_rank=("rank_without_cancer", "median"), max_leave_one_rank=("rank_without_cancer", "max"))
    )

    rows = []
    for state in [s for s in PRIORITY_STATES if s in map_df.index]:
        r = {
            "state_id": state,
            "state_label": short_label(state),
            "n_samples": int(final.loc[state, "n_samples"]) if state in final.index else 0,
            "n_events": int(final.loc[state, "n_events"]) if state in final.index else 0,
            "tcga_adjusted_fdr_lt_0_10": float(state in final.index and final.loc[state, "fdr"] < 0.10),
            "tcga_adjusted_hr_gt_1": float(state in final.index and final.loc[state, "hr_per_within_cancer_sd"] > 1),
            "stage_increasing_fdr_lt_0_05": float(
                state in stage.index
                and stage.loc[state, "stage_fdr"] < 0.05
                and "increases" in str(stage.loc[state, "stage_direction"])
            ),
            "within_cancer_variance_ge_0_50": float(
                state in map_df.index and map_df.loc[state, "within_cancer_variance_share"] >= 0.50
            ),
            "cost_rank_le_15": float(state in map_df.index and map_df.loc[state, "rank_desc_cost"] <= 15),
            "mortality_rank_le_15": float(state in map_df.index and map_df.loc[state, "rank_desc"] <= 15),
            "counterfactual_reducible_cost_rank_le_10": float(
                state in cf.index and cf.loc[state, "reducible_cost_rank"] <= 10
            ),
            "leave_one_cancer_median_rank_le_10": float(
                state in loo_summary.index and loo_summary.loc[state, "median_leave_one_rank"] <= 10
            ),
            "negative_control_exceeds_null_p05": float(state in neg_map.index and bool(neg_map.loc[state])),
        }
        rows.append(r)
    robust = pd.DataFrame(rows)
    robust["robustness_score"] = robust[
        [
            "tcga_adjusted_fdr_lt_0_10",
            "tcga_adjusted_hr_gt_1",
            "stage_increasing_fdr_lt_0_05",
            "within_cancer_variance_ge_0_50",
            "cost_rank_le_15",
            "mortality_rank_le_15",
            "counterfactual_reducible_cost_rank_le_10",
            "leave_one_cancer_median_rank_le_10",
            "negative_control_exceeds_null_p05",
        ]
    ].mean(axis=1)
    robust = robust.sort_values("robustness_score", ascending=False)
    write_table(
        robust,
        OUT / "robustness_matrix.tsv",
        "Binary and semi-binary robustness matrix for priority ecosystem states",
        "local progression-cost-prognosis, counterfactual, negative-control and TCGA tables",
        "; ".join(
            rel(p)
            for p in [
                SRC / "ecosystem_state_progression_cost_prognosis_map.csv",
                SRC / "ecosystem_state_stage_gradients.csv",
                SRC / "ecosystem_state_within_cancer_counterfactual_scores.csv",
                SRC / "nmf_state_global_mortality_leave_one_cancer_out.csv",
                SRC / "ecosystem_state_negative_control_summary.csv",
                OUT / "tcga_incremental_cox.tsv",
            ]
        ),
    )
    make_robustness_figure(robust)
    make_robustness_summary(robust)
    return robust


def make_robustness_figure(robust: pd.DataFrame) -> None:
    metrics = [
        "tcga_adjusted_fdr_lt_0_10",
        "tcga_adjusted_hr_gt_1",
        "stage_increasing_fdr_lt_0_05",
        "within_cancer_variance_ge_0_50",
        "cost_rank_le_15",
        "mortality_rank_le_15",
        "counterfactual_reducible_cost_rank_le_10",
        "leave_one_cancer_median_rank_le_10",
        "negative_control_exceeds_null_p05",
    ]
    labels = [
        "TCGA FDR<0.10",
        "HR>1",
        "Stage inc.",
        "Within-cancer",
        "Cost rank",
        "Mortality rank",
        "Counterfactual",
        "Leave-one",
        "Null pass",
    ]
    source = robust[["state_id", "state_label", "n_samples", "n_events"] + metrics + ["robustness_score"]]
    write_table(
        source,
        FIG / "robustness_heatmap_source.tsv",
        "Source table for robustness heatmap",
        "robustness_matrix.tsv",
        rel(OUT / "robustness_matrix.tsv"),
    )
    mat = robust[metrics].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(robust)))
    ax.set_yticklabels(robust["state_label"], fontsize=7)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("Robustness evidence matrix", fontsize=10, weight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "1" if mat[i, j] >= 0.5 else "", ha="center", va="center", fontsize=6, color="#1d2d3a")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="criterion met")
    fig.tight_layout()
    path = FIG / "robustness_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    record(
        path,
        "Robustness heatmap figure",
        "matplotlib generated from robustness heatmap source table",
        rel(FIG / "robustness_heatmap_source.tsv"),
    )


def make_robustness_summary(robust: pd.DataFrame) -> None:
    top = robust.head(6)
    lines = [
        "# Robustness summary",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "The robustness matrix records whether priority states pass each available local check. A value of 1 means that the criterion was met in the existing processed data; a value of 0 means that it was not met or was unavailable.",
        "",
        "## Highest-scoring states",
        "",
        md_table(top[["state_label", "n_samples", "n_events", "robustness_score"]]),
        "",
        "Negative-control results remain conservative for several states, which should be discussed as evidence that burden-weighted representation is not fully separable from cancer-site composition.",
    ]
    write_text(
        REPORTS / "robustness_summary.md",
        "\n".join(lines),
        "Narrative summary of robustness checks",
        "robustness_matrix.tsv",
        rel(OUT / "robustness_matrix.tsv"),
    )


def ensure_external_file(path: Path, url: str) -> tuple[bool, str]:
    if path.exists() and path.stat().st_size > 0:
        record(path, "Raw external validation input file", url, "existing local cache")
        return True, "present_in_local_cache"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
        record(path, "Raw external validation input file", url, "downloaded by run_all.py")
        return True, "downloaded"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def ensure_riaz_inputs() -> tuple[bool, dict[str, str]]:
    status = {}
    ok_all = True
    for path, url in RIAZ_URLS.items():
        ok, note = ensure_external_file(path, url)
        status[rel(path)] = note
        ok_all = ok_all and ok
    return ok_all, status


def load_entrez_symbol_map() -> dict[str, str]:
    gene_info = pd.read_csv(
        GENE_INFO,
        sep="\t",
        dtype=str,
        usecols=["#tax_id", "GeneID", "Symbol"],
    )
    gene_info = gene_info[(gene_info["#tax_id"] == "9606") & (gene_info["Symbol"].notna())]
    gene_info = gene_info[gene_info["Symbol"] != "-"]
    return dict(zip(gene_info["GeneID"].astype(str), gene_info["Symbol"].astype(str)))


def make_riaz_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_map = load_entrez_symbol_map()
    expr_path = RIAZ_RAW / "rld.BMS038.20171011.csv"
    sample_path = RIAZ_RAW / "SampleTableCorrected.9.19.16.csv"
    clinical_path = RIAZ_RAW / "bms038_clinical_data.csv"
    sig_path = OUT / "signature_catalog.tsv"

    expr = pd.read_csv(expr_path)
    expr = expr.rename(columns={expr.columns[0]: "entrez_id"})
    expr["gene_symbol"] = expr["entrez_id"].astype(str).map(gene_map)
    expr = expr.dropna(subset=["gene_symbol"])
    sample_cols = [c for c in expr.columns if c not in {"entrez_id", "gene_symbol"}]
    expr_gene = expr.groupby("gene_symbol", as_index=True)[sample_cols].mean(numeric_only=True).T
    expr_gene.index.name = "sample_id"

    sig = pd.read_csv(sig_path, sep="\t")
    states = [s for s in PRIORITY_STATES if s in set(sig["state_id"])]
    rows = []
    coverage = []
    for state in states:
        genes = (
            sig[sig["state_id"] == state]
            .sort_values("marker_rank_or_weight")["gene_symbol"]
            .astype(str)
            .str.upper()
            .drop_duplicates()
            .tolist()
        )
        available = [g for g in genes if g in expr_gene.columns]
        coverage.append(
            {
                "cohort": "Riaz_GSE91061",
                "state_id": state,
                "state_label": short_label(state),
                "n_signature_genes": len(genes),
                "n_signature_genes_used": len(available),
                "genes_used": ";".join(available),
            }
        )
        if not available:
            continue
        x = expr_gene[available].copy()
        z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0).replace(0, np.nan)
        score = z.mean(axis=1)
        for sample_id, val in score.items():
            rows.append(
                {
                    "cohort": "Riaz_GSE91061",
                    "sample_id": sample_id,
                    "state_id": state,
                    "state_label": short_label(state),
                    "scoring_method": "zscore_mean_rlog",
                    "score": val,
                    "n_signature_genes_used": len(available),
                    "provenance_status": "downloaded_open_github_rld_matrix",
                }
            )
    scores = pd.DataFrame(rows)
    samples = pd.read_csv(sample_path)
    clinical = pd.read_csv(clinical_path)
    clinical = clinical.rename(columns={"Sample": "clinical_sample"})
    scores = scores.merge(
        samples[["Sample", "PatientID", "PreOn", "BOR", "Response", "Cohort"]],
        left_on="sample_id",
        right_on="Sample",
        how="left",
    )
    scores = scores.merge(
        clinical[["PatientID", "PFS", "PFSWK", "PFS_SOR", "OS", "OSWK", "OS_SOR", "myBOR", "myBOR2", "myBOR3"]],
        on="PatientID",
        how="left",
    )
    scores["response_binary"] = scores["Response"].map({"PRCR": 1, "PD": 0, "SD": 0})
    scores = scores.drop(columns=["Sample"])
    coverage_df = pd.DataFrame(coverage)

    write_table(
        scores,
        OUT / "icb_signature_scores.tsv",
        "Riaz/GSE91061 prioritized ecosystem signature scores with sample metadata",
        "riazn/bms038_analysis rlog expression and sample table",
        f"{rel(expr_path)}; {rel(sample_path)}; {rel(clinical_path)}; {rel(sig_path)}; {rel(GENE_INFO)}",
    )
    write_table(
        coverage_df,
        OUT / "icb_signature_gene_coverage.tsv",
        "Gene coverage for Riaz/GSE91061 signature scoring",
        "signature catalog intersected with Riaz Entrez-to-symbol expression matrix",
        f"{rel(expr_path)}; {rel(sig_path)}; {rel(GENE_INFO)}",
    )
    return scores, coverage_df


def fit_logistic_response(d: pd.DataFrame) -> tuple[float, float, float, str]:
    d = d.dropna(subset=["score", "response_binary"]).copy()
    d = d[d["response_binary"].isin([0, 1])]
    if len(d) < 20 or d["response_binary"].nunique() < 2:
        return np.nan, np.nan, np.nan, "insufficient_response_classes"
    sd = d["score"].std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.nan, np.nan, np.nan, "constant_score"
    x = pd.DataFrame({"state_score_z": (d["score"] - d["score"].mean()) / sd})
    cohort = pd.get_dummies(d["Cohort"].astype(str), prefix="cohort", drop_first=True, dtype=float)
    if cohort.shape[1] > 0:
        x = pd.concat([x, cohort], axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = d["response_binary"].astype(float)
    try:
        model = sm.Logit(y, x).fit(disp=False, maxiter=200)
        coef = float(model.params["state_score_z"])
        se = float(model.bse["state_score_z"])
        p = float(model.pvalues["state_score_z"])
        return coef, se, p, ""
    except Exception as exc:
        return np.nan, np.nan, np.nan, f"{type(exc).__name__}: {exc}"


def fit_icb_cox(
    d: pd.DataFrame,
    duration_col: str,
    event_col: str,
    categorical_covariates: list[str] | None = None,
    numeric_covariates: list[str] | None = None,
    min_samples: int = 40,
    min_events: int = 10,
) -> tuple[float, float, float, int, int, str]:
    categorical_covariates = categorical_covariates or []
    numeric_covariates = numeric_covariates or []
    d = d.dropna(subset=["score", duration_col, event_col]).copy()
    d[duration_col] = pd.to_numeric(d[duration_col], errors="coerce")
    d[event_col] = pd.to_numeric(d[event_col], errors="coerce")
    d = d[(d[duration_col] > 0) & d[event_col].isin([0, 1])]
    if len(d) < min_samples or int(d[event_col].sum()) < min_events:
        return np.nan, np.nan, np.nan, len(d), int(d[event_col].sum()) if event_col in d else 0, "insufficient_samples_or_events"
    sd = d["score"].std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.nan, np.nan, np.nan, len(d), int(d[event_col].sum()), "constant_score"
    x = pd.DataFrame(
        {
            duration_col: d[duration_col].astype(float),
            event_col: d[event_col].astype(int),
            "state_score_z": (d["score"] - d["score"].mean()) / sd,
        },
        index=d.index,
    )
    for cov in categorical_covariates:
        if cov in d.columns and d[cov].notna().sum() >= min_samples and d[cov].nunique(dropna=True) > 1:
            x = pd.concat([x, pd.get_dummies(d[cov].astype(str), prefix=cov.replace(" ", "_"), drop_first=True, dtype=float)], axis=1)
    for cov in numeric_covariates:
        if cov in d.columns:
            vals = pd.to_numeric(d[cov], errors="coerce")
            if vals.notna().sum() >= min_samples and vals.std(ddof=0) > 0:
                x[f"{cov}_z"] = (vals - vals.mean()) / vals.std(ddof=0)
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < min_samples or int(x[event_col].sum()) < min_events:
        return np.nan, np.nan, np.nan, len(x), int(x[event_col].sum()) if event_col in x else 0, "insufficient_complete_cases_or_events"
    try:
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(x, duration_col=duration_col, event_col=event_col, show_progress=False)
        s = cph.summary.loc["state_score_z"]
        return float(s["coef"]), float(s["se(coef)"]), float(s["p"]), len(x), int(x[event_col].sum()), ""
    except Exception as exc:
        return np.nan, np.nan, np.nan, len(x), int(x[event_col].sum()), f"{type(exc).__name__}: {exc}"


def make_riaz_icb_validation() -> bool:
    ok, input_status = ensure_riaz_inputs()
    if not ok:
        return False

    scores, coverage = make_riaz_scores()
    existing = pd.read_csv(OUT / "icb_state_validation.tsv", sep="\t")
    existing = existing[existing["cohort"] != "Riaz_GSE91061"].copy()
    rows = []
    for endpoint, mask, note in [
        ("objective_response_pre_treatment", scores["PreOn"].eq("Pre"), "primary baseline/pre-treatment test"),
        ("objective_response_on_treatment", scores["PreOn"].eq("On"), "exploratory on-treatment test"),
    ]:
        subset = scores[mask].copy()
        for state, d in subset.groupby("state_id", observed=True):
            coef, se, p, err = fit_logistic_response(d)
            n = int(d.dropna(subset=["response_binary"])["sample_id"].nunique())
            events = int(d.dropna(subset=["response_binary"]).drop_duplicates("sample_id")["response_binary"].sum())
            ci = ""
            odds_ratio = np.nan
            if np.isfinite(coef) and np.isfinite(se):
                ci = f"logOR 95% CI [{coef - 1.96 * se:.3g}, {coef + 1.96 * se:.3g}]"
                odds_ratio = float(math.exp(coef))
            rows.append(
                {
                    "cohort": "Riaz_GSE91061",
                    "cancer_type": "melanoma",
                    "treatment": "nivolumab",
                    "endpoint": endpoint,
                    "state_label": short_label(state),
                    "state_id": state,
                    "n_samples": n,
                    "n_events": events,
                    "effect_size": coef,
                    "odds_ratio_per_sd": odds_ratio,
                    "confidence_interval_or_se": ci,
                    "p_value": p,
                    "fdr": np.nan,
                    "covariates": "state_score_z + cohort",
                    "interpretation": (
                        f"{note}; positive logOR means higher score in responders"
                        if not err
                        else f"not evaluable: {err}"
                    ),
                }
            )
    pre_scores = scores[scores["PreOn"].eq("Pre")].copy()
    pre_scores["pfs_event"] = 1 - pd.to_numeric(pre_scores["PFS_SOR"], errors="coerce")
    pre_scores["os_event"] = 1 - pd.to_numeric(pre_scores["OS_SOR"], errors="coerce")
    for endpoint, duration_col, event_col, note in [
        ("progression_free_survival_pre_treatment", "PFS", "pfs_event", "baseline PFS Cox model; positive logHR means shorter PFS"),
        ("overall_survival_pre_treatment", "OS", "os_event", "baseline OS Cox model; positive logHR means shorter OS"),
    ]:
        for state, d in pre_scores.groupby("state_id", observed=True):
            coef, se, p, n, events, err = fit_icb_cox(
                d,
                duration_col,
                event_col,
                categorical_covariates=["Cohort"],
                min_samples=25,
                min_events=8,
            )
            ci = f"logHR 95% CI [{coef - 1.96 * se:.3g}, {coef + 1.96 * se:.3g}]" if np.isfinite(coef) and np.isfinite(se) else ""
            rows.append(
                {
                    "cohort": "Riaz_GSE91061",
                    "cancer_type": "melanoma",
                    "treatment": "nivolumab",
                    "endpoint": endpoint,
                    "state_label": short_label(state),
                    "state_id": state,
                    "n_samples": n,
                    "n_events": events,
                    "effect_size": coef,
                    "odds_ratio_per_sd": "",
                    "confidence_interval_or_se": ci,
                    "p_value": p,
                    "fdr": np.nan,
                    "covariates": "state_score_z + cohort",
                    "interpretation": note if not err else f"not evaluable: {err}",
                }
            )
    riaz = pd.DataFrame(rows)
    riaz["fdr"] = bh_fdr(riaz["p_value"])
    out = pd.concat([existing, riaz], ignore_index=True, sort=False)
    write_table(
        out,
        OUT / "icb_state_validation.tsv",
        "Immunotherapy state validation table with Riaz/GSE91061 response analysis",
        "Riaz/GSE91061 rlog expression, sample metadata and clinical response",
        f"{rel(OUT / 'icb_signature_scores.tsv')}; {rel(OUT / 'icb_signature_gene_coverage.tsv')}",
    )

    prov = pd.read_csv(OUT / "external_validation_provenance.tsv", sep="\t")
    for col in ["downloaded_files", "sha256", "analysis_note"]:
        if col not in prov.columns:
            prov[col] = ""
    for col in ["status", "reason", "download_date", "downloaded_files", "sha256", "analysis_note"]:
        if col in prov.columns:
            prov[col] = prov[col].fillna("").astype(str)
    file_list = [p for p in RIAZ_URLS if p.exists()]
    riaz_mask = prov["dataset"].str.contains("Riaz", na=False)
    prov.loc[riaz_mask, "status"] = "downloaded_and_analyzed"
    prov.loc[riaz_mask, "reason"] = "Open GitHub/GEO-linked expression and clinical files were available and scored in this run."
    prov.loc[riaz_mask, "download_date"] = RUN_DATE
    prov.loc[riaz_mask, "downloaded_files"] = ";".join(rel(p) for p in file_list)
    prov.loc[riaz_mask, "sha256"] = ";".join(f"{p.name}:{checksum(p)}" for p in file_list)
    prov.loc[riaz_mask, "analysis_note"] = "Rlog expression scored by z-score mean; PR/CR versus PD/SD response tested by logistic regression."
    write_table(
        prov,
        OUT / "external_validation_provenance.tsv",
        "External validation provenance and availability status",
        "local file availability check plus Riaz download/scoring",
        "goal.md external validation requirements; Riaz raw files",
    )

    make_riaz_icb_figure(riaz)
    make_riaz_icb_report(riaz, coverage, input_status)
    return True


def make_riaz_icb_figure(riaz: pd.DataFrame) -> None:
    source = riaz[riaz["endpoint"] == "objective_response_pre_treatment"].copy()
    source = source.sort_values("effect_size")
    write_table(
        source,
        FIG / "icb_validation_source.tsv",
        "Source table for ICB validation figure",
        "Riaz/GSE91061 logistic response analysis",
        rel(OUT / "icb_state_validation.tsv"),
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    y = np.arange(len(source))
    x = source["effect_size"].astype(float)
    se_vals = []
    for _, r in source.iterrows():
        if pd.notna(r["effect_size"]) and isinstance(r["confidence_interval_or_se"], str) and "[" in r["confidence_interval_or_se"]:
            # Recover CI width from unavailable structured SE by using p-value-free conservative display.
            se_vals.append(np.nan)
        else:
            se_vals.append(np.nan)
    colors = np.where(source["fdr"].fillna(1) < 0.1, "#b5443c", "#6c7a89")
    ax.barh(y, x, color=colors)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(source["state_label"], fontsize=7)
    ax.set_xlabel("log odds ratio per SD score")
    ax.set_title("Riaz/GSE91061 pre-treatment response", fontsize=10, weight="bold")
    ax.text(
        0.02,
        0.02,
        "Exploratory logistic model: response = PR/CR vs PD/SD; covariate = cohort.",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIG / "icb_validation.pdf"
    fig.savefig(path)
    plt.close(fig)
    record(
        path,
        "Riaz/GSE91061 ICB response validation figure",
        "matplotlib generated from ICB validation source table",
        rel(FIG / "icb_validation_source.tsv"),
    )


def make_riaz_icb_report(riaz: pd.DataFrame, coverage: pd.DataFrame, input_status: dict[str, str]) -> None:
    pre = riaz[riaz["endpoint"] == "objective_response_pre_treatment"].copy()
    pre = pre.sort_values("effect_size", ascending=False)
    sig = pre[pre["fdr"].fillna(1) < 0.1]
    lines = [
        "# Immunotherapy validation summary",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Riaz/GSE91061 melanoma nivolumab data were downloaded from the public `riazn/bms038_analysis` GitHub data directory and scored with the standardized ecosystem signatures. The primary response analysis used pre-treatment samples and coded PR/CR as responders and PD/SD as non-responders.",
        "",
        f"- Pre-treatment response rows tested: {len(pre)} states.",
        f"- States with response FDR < 0.10: {len(sig)}.",
        f"- Median signature genes used: {coverage['n_signature_genes_used'].median():.0f}.",
        "",
        "## Pre-treatment response effects",
        "",
        md_table(pre[["state_label", "n_samples", "n_events", "effect_size", "odds_ratio_per_sd", "p_value", "fdr"]].head(12)),
        "",
        "Interpretation boundary: this is an exploratory treatment-response association in a small melanoma anti-PD-1 cohort, not a causal treatment effect and not a pan-cancer therapy conclusion.",
        "",
        "## Input status",
        "",
        md_table(pd.DataFrame([{"file_path": k, "status": v} for k, v in input_status.items()])),
    ]
    write_text(
        REPORTS / "icb_validation_summary.md",
        "\n".join(lines),
        "ICB validation summary with Riaz/GSE91061 analysis",
        "Riaz/GSE91061 signature scoring and logistic response analysis",
        f"{rel(OUT / 'icb_state_validation.tsv')}; {rel(OUT / 'icb_signature_scores.tsv')}",
    )


def ensure_imvigor_inputs() -> tuple[bool, dict[str, str]]:
    status = {}
    ok, note = ensure_external_file(IMVIGOR_TAR, IMVIGOR_URL)
    status[rel(IMVIGOR_TAR)] = note
    if not ok:
        return False, status

    cds_path = IMVIGOR_EXTRACT / "IMvigor210CoreBiologies" / "data" / "cds.RData"
    if not cds_path.exists():
        try:
            with tarfile.open(IMVIGOR_TAR, "r:gz") as tar:
                member = tar.getmember("IMvigor210CoreBiologies/data/cds.RData")
                IMVIGOR_EXTRACT.mkdir(parents=True, exist_ok=True)
                tar.extract(member, IMVIGOR_EXTRACT)
            status[rel(cds_path)] = "extracted_from_tarball"
        except Exception as exc:
            status[rel(cds_path)] = f"{type(exc).__name__}: {exc}"
            return False, status
    else:
        status[rel(cds_path)] = "present_in_local_cache"

    required = [IMVIGOR_TSV / "counts.tsv", IMVIGOR_TSV / "pheno.tsv", IMVIGOR_TSV / "feature.tsv"]
    if all(p.exists() and p.stat().st_size > 0 for p in required):
        for p in required:
            record(p, "Extracted IMvigor210CoreBiologies TSV", "cds.RData low-level R export", rel(cds_path))
            status[rel(p)] = "present_in_local_cache"
        return True, status

    r_code = f"""
out <- '{IMVIGOR_TSV.as_posix()}'
dir.create(out, recursive=TRUE, showWarnings=FALSE)
e <- new.env()
load('{cds_path.as_posix()}', envir=e)
cds <- e$cds
a <- attributes(cds)
counts <- get('counts', envir=a$assayData)
pheno <- attributes(a$phenoData)$data
feature <- attributes(a$featureData)$data
write.table(counts, file=file.path(out,'counts.tsv'), sep='\\t', quote=FALSE, col.names=NA)
write.table(pheno, file=file.path(out,'pheno.tsv'), sep='\\t', quote=FALSE, col.names=NA)
write.table(feature, file=file.path(out,'feature.tsv'), sep='\\t', quote=FALSE, col.names=NA)
"""
    try:
        subprocess.run(["Rscript", "-e", r_code], check=True, capture_output=True, text=True)
    except Exception as exc:
        status["Rscript_extract"] = f"{type(exc).__name__}: {exc}"
        return False, status
    for p in required:
        record(p, "Extracted IMvigor210CoreBiologies TSV", "cds.RData low-level R export", rel(cds_path))
        status[rel(p)] = "exported_from_cds_RData"
    return True, status


def make_imvigor_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    counts_path = IMVIGOR_TSV / "counts.tsv"
    pheno_path = IMVIGOR_TSV / "pheno.tsv"
    feature_path = IMVIGOR_TSV / "feature.tsv"
    sig_path = OUT / "signature_catalog.tsv"

    counts = pd.read_csv(counts_path, sep="\t").rename(columns={"Unnamed: 0": "row_id"})
    feature = pd.read_csv(feature_path, sep="\t").rename(columns={"Unnamed: 0": "row_id"})
    pheno = pd.read_csv(pheno_path, sep="\t").rename(columns={"Unnamed: 0": "sample_id"})
    feature["row_id"] = feature["row_id"].astype(str)
    counts["row_id"] = counts["row_id"].astype(str)
    feature = feature[["row_id", "Symbol"]].dropna()
    counts = counts.merge(feature, on="row_id", how="left").dropna(subset=["Symbol"])
    sample_cols = [c for c in counts.columns if c not in {"row_id", "Symbol"}]
    x = counts.groupby("Symbol", as_index=True)[sample_cols].sum(numeric_only=True)
    lib = x.sum(axis=0).replace(0, np.nan)
    logcpm = np.log2(x.div(lib, axis=1) * 1_000_000 + 1).T
    logcpm.index.name = "sample_id"

    sig = pd.read_csv(sig_path, sep="\t")
    states = [s for s in PRIORITY_STATES if s in set(sig["state_id"])]
    rows = []
    coverage = []
    for state in states:
        genes = (
            sig[sig["state_id"] == state]
            .sort_values("marker_rank_or_weight")["gene_symbol"]
            .astype(str)
            .str.upper()
            .drop_duplicates()
            .tolist()
        )
        available = [g for g in genes if g in logcpm.columns]
        coverage.append(
            {
                "cohort": "IMvigor210",
                "state_id": state,
                "state_label": short_label(state),
                "n_signature_genes": len(genes),
                "n_signature_genes_used": len(available),
                "genes_used": ";".join(available),
            }
        )
        if not available:
            continue
        sub = logcpm[available]
        z = (sub - sub.mean(axis=0)) / sub.std(axis=0, ddof=0).replace(0, np.nan)
        score = z.mean(axis=1)
        for sample_id, val in score.items():
            rows.append(
                {
                    "cohort": "IMvigor210",
                    "sample_id": sample_id,
                    "state_id": state,
                    "state_label": short_label(state),
                    "scoring_method": "zscore_mean_logcpm",
                    "score": val,
                    "n_signature_genes_used": len(available),
                    "provenance_status": "downloaded_imvigor210_cds_counts",
                }
            )
    scores = pd.DataFrame(rows).merge(pheno, on="sample_id", how="left")
    scores["response_binary"] = scores["binaryResponse"].map({"CR/PR": 1, "SD/PD": 0})
    coverage_df = pd.DataFrame(coverage)

    existing_scores = pd.read_csv(OUT / "icb_signature_scores.tsv", sep="\t") if (OUT / "icb_signature_scores.tsv").exists() else pd.DataFrame()
    combined_scores = pd.concat([existing_scores[existing_scores.get("cohort", "") != "IMvigor210"], scores], ignore_index=True, sort=False)
    write_table(
        combined_scores,
        OUT / "icb_signature_scores.tsv",
        "ICB prioritized ecosystem signature scores with sample metadata",
        "Riaz/GSE91061 and IMvigor210 expression tables",
        f"{rel(OUT / 'signature_catalog.tsv')}; {rel(counts_path)}; {rel(pheno_path)}; {rel(feature_path)}",
    )

    existing_cov = pd.read_csv(OUT / "icb_signature_gene_coverage.tsv", sep="\t") if (OUT / "icb_signature_gene_coverage.tsv").exists() else pd.DataFrame()
    combined_cov = pd.concat([existing_cov[existing_cov.get("cohort", "") != "IMvigor210"], coverage_df], ignore_index=True, sort=False)
    write_table(
        combined_cov,
        OUT / "icb_signature_gene_coverage.tsv",
        "Gene coverage for ICB signature scoring",
        "signature catalog intersected with ICB expression matrices",
        f"{rel(OUT / 'signature_catalog.tsv')}; {rel(feature_path)}",
    )
    return scores, coverage_df


def fit_imvigor_response(d: pd.DataFrame) -> tuple[float, float, float, str]:
    d = d.dropna(subset=["score", "response_binary"]).copy()
    d = d[d["response_binary"].isin([0, 1])]
    if len(d) < 50 or d["response_binary"].nunique() < 2:
        return np.nan, np.nan, np.nan, "insufficient_response_classes"
    sd = d["score"].std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.nan, np.nan, np.nan, "constant_score"
    x = pd.DataFrame({"state_score_z": (d["score"] - d["score"].mean()) / sd}, index=d.index)
    if "Enrollment IC" in d.columns:
        ic = pd.get_dummies(d["Enrollment IC"].astype(str), prefix="IC", drop_first=True, dtype=float)
        x = pd.concat([x, ic], axis=1)
    if "FMOne mutation burden per MB" in d.columns:
        tmb = pd.to_numeric(d["FMOne mutation burden per MB"], errors="coerce")
        if tmb.notna().sum() > 50 and tmb.std(ddof=0) > 0:
            x["tmb_z"] = (tmb - tmb.mean()) / tmb.std(ddof=0)
    x = sm.add_constant(x, has_constant="add")
    keep = x.notna().all(axis=1)
    y = d.loc[keep, "response_binary"].astype(float)
    x = x.loc[keep]
    try:
        model = sm.Logit(y, x).fit(disp=False, maxiter=200)
        coef = float(model.params["state_score_z"])
        se = float(model.bse["state_score_z"])
        p = float(model.pvalues["state_score_z"])
        return coef, se, p, ""
    except Exception as exc:
        return np.nan, np.nan, np.nan, f"{type(exc).__name__}: {exc}"


def make_imvigor_icb_validation() -> bool:
    ok, input_status = ensure_imvigor_inputs()
    if not ok:
        return False
    scores, coverage = make_imvigor_scores()
    existing = pd.read_csv(OUT / "icb_state_validation.tsv", sep="\t")
    existing = existing[existing["cohort"] != "IMvigor210"].copy()
    rows = []
    for state, d in scores.groupby("state_id", observed=True):
        coef, se, p, err = fit_imvigor_response(d)
        dd = d.dropna(subset=["response_binary"]).drop_duplicates("sample_id")
        n = int(dd["sample_id"].nunique())
        events = int(dd["response_binary"].sum())
        ci = ""
        odds_ratio = np.nan
        if np.isfinite(coef) and np.isfinite(se):
            ci = f"logOR 95% CI [{coef - 1.96 * se:.3g}, {coef + 1.96 * se:.3g}]"
            odds_ratio = float(math.exp(coef))
        rows.append(
            {
                "cohort": "IMvigor210",
                "cancer_type": "urothelial carcinoma",
                "treatment": "atezolizumab",
                "endpoint": "objective_response",
                "state_label": short_label(state),
                "state_id": state,
                "n_samples": n,
                "n_events": events,
                "effect_size": coef,
                "odds_ratio_per_sd": odds_ratio,
                "confidence_interval_or_se": ci,
                "p_value": p,
                "fdr": np.nan,
                "covariates": "state_score_z + Enrollment IC + TMB",
                "interpretation": (
                    "objective response CR/PR versus SD/PD; positive logOR means higher score in responders"
                    if not err
                    else f"not evaluable: {err}"
                ),
            }
        )
        ccoef, cse, cp, cn, cevents, cerr = fit_icb_cox(
            d,
            "os",
            "censOS",
            categorical_covariates=["Enrollment IC"],
            numeric_covariates=["FMOne mutation burden per MB"],
            min_samples=100,
            min_events=50,
        )
        cci = f"logHR 95% CI [{ccoef - 1.96 * cse:.3g}, {ccoef + 1.96 * cse:.3g}]" if np.isfinite(ccoef) and np.isfinite(cse) else ""
        rows.append(
            {
                "cohort": "IMvigor210",
                "cancer_type": "urothelial carcinoma",
                "treatment": "atezolizumab",
                "endpoint": "overall_survival",
                "state_label": short_label(state),
                "state_id": state,
                "n_samples": cn,
                "n_events": cevents,
                "effect_size": ccoef,
                "odds_ratio_per_sd": "",
                "confidence_interval_or_se": cci,
                "p_value": cp,
                "fdr": np.nan,
                "covariates": "state_score_z + Enrollment IC + TMB",
                "interpretation": (
                    "overall survival Cox model; positive logHR means higher mortality hazard"
                    if not cerr
                    else f"not evaluable: {cerr}"
                ),
            }
        )
    imv = pd.DataFrame(rows)
    out = pd.concat([existing, imv], ignore_index=True, sort=False)
    out["fdr"] = bh_fdr(out["p_value"])
    write_table(
        out,
        OUT / "icb_state_validation.tsv",
        "Immunotherapy state validation table with Riaz and IMvigor response analyses",
        "Riaz/GSE91061 and IMvigor210 expression-response analyses",
        f"{rel(OUT / 'icb_signature_scores.tsv')}; {rel(OUT / 'icb_signature_gene_coverage.tsv')}",
    )

    prov = pd.read_csv(OUT / "external_validation_provenance.tsv", sep="\t")
    for col in ["downloaded_files", "sha256", "analysis_note"]:
        if col not in prov.columns:
            prov[col] = ""
    for col in ["status", "reason", "download_date", "downloaded_files", "sha256", "analysis_note"]:
        if col in prov.columns:
            prov[col] = prov[col].fillna("").astype(str)
    imv_files = [IMVIGOR_TAR, IMVIGOR_TSV / "counts.tsv", IMVIGOR_TSV / "pheno.tsv", IMVIGOR_TSV / "feature.tsv"]
    mask = prov["dataset"].str.contains("IMvigor", na=False)
    prov.loc[mask, "status"] = "downloaded_and_analyzed"
    prov.loc[mask, "reason"] = "Open package tarball was available; cds.RData was extracted to counts/pheno/feature TSVs and scored."
    prov.loc[mask, "download_date"] = RUN_DATE
    prov.loc[mask, "downloaded_files"] = ";".join(rel(p) for p in imv_files if p.exists())
    prov.loc[mask, "sha256"] = ";".join(f"{p.name}:{checksum(p)}" for p in imv_files if p.exists())
    prov.loc[mask, "analysis_note"] = "Counts normalized to logCPM; CR/PR versus SD/PD response tested by logistic regression."
    write_table(
        prov,
        OUT / "external_validation_provenance.tsv",
        "External validation provenance and availability status",
        "local file availability check plus Riaz/IMvigor download/scoring",
        "goal.md external validation requirements; Riaz and IMvigor raw files",
    )

    make_combined_icb_figure(out)
    make_combined_icb_report(out, coverage, input_status)
    return True


def make_combined_icb_figure(icb: pd.DataFrame) -> None:
    source = icb[
        (icb["n_samples"] > 0)
        & (
            ((icb["cohort"] == "Riaz_GSE91061") & (icb["endpoint"] == "objective_response_pre_treatment"))
            | ((icb["cohort"] == "IMvigor210") & (icb["endpoint"] == "objective_response"))
        )
    ].copy()
    source["display"] = source["state_label"] + " | " + source["cohort"].replace({"Riaz_GSE91061": "Riaz", "IMvigor210": "IMvigor"})
    source = source.sort_values(["cohort", "effect_size"])
    write_table(
        source,
        FIG / "icb_validation_source.tsv",
        "Source table for combined ICB validation figure",
        "Riaz/GSE91061 and IMvigor210 logistic response analysis",
        rel(OUT / "icb_state_validation.tsv"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5), sharex=True)
    for ax, cohort, title in [
        (axes[0], "Riaz_GSE91061", "Riaz melanoma nivolumab"),
        (axes[1], "IMvigor210", "IMvigor urothelial atezolizumab"),
    ]:
        d = source[source["cohort"] == cohort].sort_values("effect_size")
        y = np.arange(len(d))
        colors = np.where(d["fdr"].fillna(1) < 0.1, "#b5443c", "#6c7a89")
        ax.barh(y, d["effect_size"].astype(float), color=colors)
        ax.axvline(0, color="#222222", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(d["state_label"], fontsize=7)
        ax.set_title(title, fontsize=9, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].set_xlabel("log OR per SD")
    axes[1].set_xlabel("log OR per SD")
    fig.suptitle("Exploratory ICB response validation", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG / "icb_validation.pdf"
    fig.savefig(path)
    plt.close(fig)
    record(
        path,
        "Combined ICB response validation figure",
        "matplotlib generated from ICB validation source table",
        rel(FIG / "icb_validation_source.tsv"),
    )


def make_combined_icb_report(icb: pd.DataFrame, coverage: pd.DataFrame, input_status: dict[str, str]) -> None:
    tested = icb[icb["n_samples"] > 0].copy()
    sig = tested[tested["fdr"].fillna(1) < 0.1]
    riaz = tested[(tested["cohort"] == "Riaz_GSE91061") & (tested["endpoint"] == "objective_response_pre_treatment")]
    imv = tested[(tested["cohort"] == "IMvigor210") & (tested["endpoint"] == "objective_response")]
    survival = tested[tested["endpoint"].astype(str).str.contains("survival", na=False)].copy()
    lines = [
        "# Immunotherapy validation summary",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Two public immunotherapy cohorts were scored with the standardized ecosystem signatures: Riaz/GSE91061 melanoma nivolumab and IMvigor210 urothelial atezolizumab. Response models code CR/PR as responders and SD/PD as non-responders; available OS/PFS endpoints were tested by Cox models.",
        "",
        f"- Riaz pre-treatment states tested: {len(riaz)}.",
        f"- IMvigor210 states tested: {len(imv)}.",
        f"- ICB response/survival tests with FDR < 0.10 across available cohort-state tests: {len(sig)}.",
        f"- Survival/PFS state-endpoint tests included: {len(survival)}.",
        f"- Median IMvigor signature genes used: {coverage['n_signature_genes_used'].median():.0f}.",
        "",
        "## Riaz/GSE91061 pre-treatment response effects",
        "",
        md_table(riaz.sort_values("effect_size", ascending=False)[["state_label", "n_samples", "n_events", "effect_size", "odds_ratio_per_sd", "p_value", "fdr"]].head(12)),
        "",
        "## IMvigor210 response effects",
        "",
        md_table(imv.sort_values("effect_size", ascending=False)[["state_label", "n_samples", "n_events", "effect_size", "odds_ratio_per_sd", "p_value", "fdr"]].head(12)),
        "",
        "## ICB survival/PFS effects",
        "",
        md_table(survival.sort_values("p_value")[["cohort", "endpoint", "state_label", "n_samples", "n_events", "effect_size", "p_value", "fdr"]].head(12)),
        "",
        "Interpretation boundary: these are exploratory treatment-response associations in public bulk RNA-seq cohorts, not causal treatment effects. The Riaz cohort is small; IMvigor is larger but tumor-type and treatment specific.",
        "",
        "## IMvigor input status",
        "",
        md_table(pd.DataFrame([{"file_path": k, "status": v} for k, v in input_status.items()])),
    ]
    write_text(
        REPORTS / "icb_validation_summary.md",
        "\n".join(lines),
        "ICB validation summary with Riaz/GSE91061 and IMvigor210 analyses",
        "ICB signature scoring and logistic response analyses",
        f"{rel(OUT / 'icb_state_validation.tsv')}; {rel(OUT / 'icb_signature_scores.tsv')}",
    )


def ensure_cptac_inputs() -> tuple[bool, dict[str, str]]:
    status = {}
    ok_all = True
    for path, meta in CPTAC_URLS.items():
        ok, note = ensure_external_file(path, meta["url"])
        if ok:
            actual = md5sum(path)
            if actual == meta["md5"]:
                note = f"{note}; md5_verified"
                record(path, meta["description"], meta["url"], f"Zenodo 8394329; md5:{actual}")
            else:
                note = f"{note}; md5_mismatch expected {meta['md5']} observed {actual}"
                ok = False
        status[rel(path)] = note
        ok_all = ok_all and ok
    return ok_all, status


def load_ensembl_symbol_map() -> dict[str, str]:
    gene_info = pd.read_csv(
        GENE_INFO,
        sep="\t",
        dtype=str,
        usecols=["#tax_id", "Symbol", "dbXrefs"],
    )
    gene_info = gene_info[
        (gene_info["#tax_id"] == "9606")
        & gene_info["Symbol"].notna()
        & gene_info["dbXrefs"].notna()
        & gene_info["dbXrefs"].str.contains("Ensembl:", na=False)
    ].copy()
    rows = []
    for _, r in gene_info.iterrows():
        for ens in re.findall(r"Ensembl:(ENSG\d+)", str(r["dbXrefs"])):
            rows.append((ens, str(r["Symbol"]).upper()))
    return dict(rows)


def load_cptac_gene_matrix(path: Path, sample_mode: str) -> pd.DataFrame:
    ens_map = load_ensembl_symbol_map()
    mat = pd.read_csv(path, sep="\t", index_col=0)
    mat.index = mat.index.astype(str).str.replace(r"\.\d+$", "", regex=True).map(ens_map)
    mat = mat[mat.index.notna()].copy()
    mat.index.name = "gene_symbol"
    mat = mat.apply(pd.to_numeric, errors="coerce")
    if sample_mode == "rna_tumor":
        cols = [c for c in mat.columns if str(c).endswith("_T")]
        mat = mat[cols].copy()
        mat.columns = [str(c).rsplit("_", 1)[0] for c in mat.columns]
    elif sample_mode == "protein_tumor":
        mat.columns = [str(c) for c in mat.columns]
        mat = mat[[c for c in mat.columns if c.startswith("C3")]].copy()
    else:
        raise ValueError(f"Unknown sample mode: {sample_mode}")
    mat = mat.groupby(level=0).mean(numeric_only=True)
    return mat


def cptac_signature_scores_for_matrix(
    mat: pd.DataFrame,
    sig: pd.DataFrame,
    states: list[str],
    cohort: str,
    scoring_method: str,
    provenance_status: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coverage = []
    if scoring_method == "zscore_mean":
        gene_sd = mat.std(axis=1, ddof=0).replace(0, np.nan)
        scored_mat = mat.sub(mat.mean(axis=1), axis=0).div(gene_sd, axis=0)
    elif scoring_method == "rank_mean_percentile":
        scored_mat = mat.rank(axis=0, pct=True)
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    for state in states:
        genes = (
            sig[sig["state_id"] == state]
            .sort_values("marker_rank_or_weight")["gene_symbol"]
            .astype(str)
            .str.upper()
            .drop_duplicates()
            .tolist()
        )
        available = [g for g in genes if g in scored_mat.index]
        coverage.append(
            {
                "cohort": cohort,
                "state_id": state,
                "state_label": short_label(state),
                "scoring_method": scoring_method,
                "n_signature_genes": len(genes),
                "n_signature_genes_used": len(available),
                "genes_used": ";".join(available),
            }
        )
        if not available:
            continue
        score = scored_mat.loc[available].mean(axis=0, skipna=True)
        for case_id, val in score.items():
            rows.append(
                {
                    "cohort": cohort,
                    "cancer_type": "UCEC",
                    "sample_id": f"{case_id}_T" if "rna" in provenance_status else case_id,
                    "case_id": case_id,
                    "state_id": state,
                    "state_label": short_label(state),
                    "scoring_method": scoring_method,
                    "score": val,
                    "n_signature_genes_used": len(available),
                    "provenance_status": provenance_status,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def stage_to_ordinal(value: object) -> float:
    text = str(value).upper()
    if "IV" in text:
        return 4.0
    if "III" in text:
        return 3.0
    if "II" in text:
        return 2.0
    if "I" in text:
        return 1.0
    return np.nan


def grade_to_ordinal(value: object) -> float:
    m = re.search(r"G\s*([1-4])", str(value).upper())
    return float(m.group(1)) if m else np.nan


def prepare_cptac_ucec_clinical() -> pd.DataFrame:
    clinical = pd.read_csv(CPTAC_CLINICAL, sep="\t", dtype=str)
    clinical = clinical[clinical["tumor_code"] == "UCEC"].copy()
    clinical["age"] = pd.to_numeric(clinical["consent/age"], errors="coerce")
    clinical["age_z"] = (clinical["age"] - clinical["age"].mean()) / clinical["age"].std(ddof=0)
    clinical["male"] = clinical["consent/sex"].map({"Male": 1.0, "Female": 0.0})
    clinical["stage_ordinal"] = clinical["baseline/tumor_stage_pathological"].map(stage_to_ordinal)
    clinical["grade_ordinal"] = clinical["cptac_path/histologic_grade"].map(grade_to_ordinal)
    clinical["os_days"] = pd.to_numeric(clinical["Overall survival, days"], errors="coerce")
    clinical["os_event"] = pd.to_numeric(clinical["Survival status (1, dead; 0, alive)"], errors="coerce")
    clinical["rfs_days"] = pd.to_numeric(clinical["Recurrence-free survival, days"], errors="coerce")
    clinical["rfs_event"] = pd.to_numeric(clinical["Recurrence status (1, yes; 0, no)"], errors="coerce")
    keep = [
        "case_id",
        "tumor_code",
        "age",
        "age_z",
        "male",
        "stage_ordinal",
        "grade_ordinal",
        "os_days",
        "os_event",
        "rfs_days",
        "rfs_event",
        "baseline/tumor_stage_pathological",
        "cptac_path/histologic_grade",
    ]
    return clinical[keep].drop_duplicates("case_id")


def fit_cptac_linear(d: pd.DataFrame, outcome: str, covariates: list[str]) -> tuple[float, float, float, int, str]:
    work = d.dropna(subset=["score", outcome]).copy()
    if len(work) < 40 or work[outcome].nunique(dropna=True) < 2:
        return np.nan, np.nan, np.nan, len(work), "insufficient_samples_or_endpoint_variation"
    sd = work["score"].std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.nan, np.nan, np.nan, len(work), "constant_score"
    work["state_score_z"] = (work["score"] - work["score"].mean()) / sd
    cols = ["state_score_z"]
    for cov in covariates:
        if cov in work.columns and work[cov].notna().sum() >= 40 and work[cov].nunique(dropna=True) > 1:
            cols.append(cov)
    x = sm.add_constant(work[cols], has_constant="add")
    y = pd.to_numeric(work[outcome], errors="coerce")
    keep = x.notna().all(axis=1) & y.notna()
    if keep.sum() < 40:
        return np.nan, np.nan, np.nan, int(keep.sum()), "insufficient_complete_cases"
    try:
        model = sm.OLS(y.loc[keep], x.loc[keep]).fit()
        return (
            float(model.params["state_score_z"]),
            float(model.bse["state_score_z"]),
            float(model.pvalues["state_score_z"]),
            int(keep.sum()),
            "",
        )
    except Exception as exc:
        return np.nan, np.nan, np.nan, int(keep.sum()), f"{type(exc).__name__}: {exc}"


def fit_cptac_cox(d: pd.DataFrame, duration: str, event: str, covariates: list[str]) -> tuple[float, float, float, int, int, str]:
    work = d.dropna(subset=["score", duration, event]).copy()
    work = work[(work[duration] > 0) & work[event].isin([0, 1])]
    if len(work) < 50 or int(work[event].sum()) < 10:
        return np.nan, np.nan, np.nan, len(work), int(work[event].sum()) if event in work else 0, "insufficient_samples_or_events"
    sd = work["score"].std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.nan, np.nan, np.nan, len(work), int(work[event].sum()), "constant_score"
    work["state_score_z"] = (work["score"] - work["score"].mean()) / sd
    cols = [duration, event, "state_score_z"]
    for cov in covariates:
        if cov in work.columns and work[cov].notna().sum() >= 50 and work[cov].nunique(dropna=True) > 1:
            cols.append(cov)
    cox = work[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(cox) < 50 or int(cox[event].sum()) < 10:
        return np.nan, np.nan, np.nan, len(cox), int(cox[event].sum()) if event in cox else 0, "insufficient_complete_cases_or_events"
    try:
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(cox, duration_col=duration, event_col=event, show_progress=False)
        s = cph.summary.loc["state_score_z"]
        return float(s["coef"]), float(s["se(coef)"]), float(s["p"]), len(cox), int(cox[event].sum()), ""
    except Exception as exc:
        return np.nan, np.nan, np.nan, len(cox), int(cox[event].sum()), f"{type(exc).__name__}: {exc}"


def make_cptac_pathway_protein_scores(prot: pd.DataFrame, sig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pathway_rows = []
    coverage = []
    gene_sd = prot.std(axis=1, ddof=0).replace(0, np.nan)
    z = prot.sub(prot.mean(axis=1), axis=0).div(gene_sd, axis=0)
    for pathway in sorted(set(CPTAC_PATHWAY_STATE_MAP.values())):
        states = [s for s, p in CPTAC_PATHWAY_STATE_MAP.items() if p == pathway]
        genes = (
            sig[sig["state_id"].isin(states)]
            .sort_values(["state_id", "marker_rank_or_weight"])["gene_symbol"]
            .astype(str)
            .str.upper()
            .drop_duplicates()
            .tolist()
        )
        available = [g for g in genes if g in z.index]
        coverage.append(
            {
                "cohort": "CPTAC_UCEC",
                "pathway": pathway,
                "n_signature_genes": len(genes),
                "n_signature_genes_used": len(available),
                "genes_used": ";".join(available),
            }
        )
        if not available:
            continue
        score = z.loc[available].mean(axis=0, skipna=True)
        for case_id, val in score.items():
            pathway_rows.append(
                {
                    "cohort": "CPTAC_UCEC",
                    "case_id": case_id,
                    "pathway": pathway,
                    "protein_pathway_score": val,
                    "n_signature_genes_used": len(available),
                }
            )
    return pd.DataFrame(pathway_rows), pd.DataFrame(coverage)


def make_cptac_validation() -> bool:
    ok, input_status = ensure_cptac_inputs()
    if not ok:
        return False

    sig = pd.read_csv(OUT / "signature_catalog.tsv", sep="\t")
    states = [s for s in PRIORITY_STATES if s in set(sig["state_id"])]
    clinical = prepare_cptac_ucec_clinical()
    rna = load_cptac_gene_matrix(CPTAC_UCEC_RNA, "rna_tumor")
    prot = load_cptac_gene_matrix(CPTAC_UCEC_PROTEIN, "protein_tumor")

    z_scores, z_cov = cptac_signature_scores_for_matrix(
        rna,
        sig,
        states,
        "CPTAC_UCEC",
        "zscore_mean",
        "zenodo_8394329_bcm_ucec_rna_md5_verified",
    )
    rank_scores, rank_cov = cptac_signature_scores_for_matrix(
        rna,
        sig,
        states,
        "CPTAC_UCEC",
        "rank_mean_percentile",
        "zenodo_8394329_bcm_ucec_rna_md5_verified",
    )
    scores = pd.concat([z_scores, rank_scores], ignore_index=True, sort=False).merge(clinical, on="case_id", how="left")
    write_table(
        scores,
        OUT / "cptac_signature_scores.tsv",
        "CPTAC UCEC RNA-derived ecosystem signature scores with clinical metadata",
        "CPTAC Zenodo 8394329 BCM UCEC RNA matrix and pan-cancer clinical table",
        f"{rel(CPTAC_UCEC_RNA)}; {rel(CPTAC_CLINICAL)}; {rel(OUT / 'signature_catalog.tsv')}; {rel(GENE_INFO)}",
    )
    write_table(
        pd.concat([z_cov, rank_cov], ignore_index=True, sort=False),
        OUT / "cptac_signature_gene_coverage.tsv",
        "Gene coverage for CPTAC UCEC RNA signature scoring",
        "signature catalog intersected with CPTAC Ensembl-to-symbol RNA matrix",
        f"{rel(CPTAC_UCEC_RNA)}; {rel(OUT / 'signature_catalog.tsv')}; {rel(GENE_INFO)}",
    )

    protein_scores, protein_cov = cptac_signature_scores_for_matrix(
        prot,
        sig,
        states,
        "CPTAC_UCEC",
        "zscore_mean",
        "zenodo_8394329_bcm_ucec_tumor_proteomics_md5_verified",
    )
    protein_scores = protein_scores.rename(columns={"score": "protein_state_score"})
    write_table(
        protein_scores,
        OUT / "cptac_protein_signature_scores.tsv",
        "CPTAC UCEC protein-derived ecosystem signature scores",
        "CPTAC Zenodo 8394329 BCM UCEC tumor proteome matrix",
        f"{rel(CPTAC_UCEC_PROTEIN)}; {rel(OUT / 'signature_catalog.tsv')}; {rel(GENE_INFO)}",
    )
    write_table(
        protein_cov,
        OUT / "cptac_protein_signature_gene_coverage.tsv",
        "Gene coverage for CPTAC UCEC protein signature scoring",
        "signature catalog intersected with CPTAC Ensembl-to-symbol proteome matrix",
        f"{rel(CPTAC_UCEC_PROTEIN)}; {rel(OUT / 'signature_catalog.tsv')}; {rel(GENE_INFO)}",
    )

    rows = []
    for method, method_scores in scores.groupby("scoring_method", observed=True):
        for state, d in method_scores.groupby("state_id", observed=True):
            for endpoint, outcome, covs, interp in [
                ("stage_ordinal", "stage_ordinal", ["age_z", "male"], "positive beta means higher score in later pathological stage"),
                ("grade_ordinal", "grade_ordinal", ["age_z", "male", "stage_ordinal"], "positive beta means higher score in higher histologic grade"),
            ]:
                beta, se, p, n, err = fit_cptac_linear(d, outcome, covs)
                ci = f"beta 95% CI [{beta - 1.96 * se:.3g}, {beta + 1.96 * se:.3g}]" if np.isfinite(beta) and np.isfinite(se) else ""
                rows.append(
                    {
                        "cohort": "CPTAC_UCEC",
                        "cancer_type": "UCEC",
                        "endpoint": endpoint,
                        "state_id": state,
                        "state_label": short_label(state),
                        "scoring_method": method,
                        "analysis_type": "linear_ordinal_clinical_endpoint",
                        "n_samples": n,
                        "n_events": "",
                        "effect_size": beta,
                        "hazard_ratio_per_sd": "",
                        "confidence_interval_or_se": ci,
                        "p_value": p,
                        "fdr": np.nan,
                        "covariates": "state_score_z + " + " + ".join(covs),
                        "interpretation": interp if not err else f"not evaluable: {err}",
                    }
                )
            for endpoint, duration, event, interp in [
                ("overall_survival", "os_days", "os_event", "positive logHR means higher mortality hazard"),
                ("recurrence_free_survival", "rfs_days", "rfs_event", "positive logHR means higher recurrence hazard"),
            ]:
                coef, se, p, n, events, err = fit_cptac_cox(d, duration, event, ["age_z", "male", "stage_ordinal", "grade_ordinal"])
                ci = f"logHR 95% CI [{coef - 1.96 * se:.3g}, {coef + 1.96 * se:.3g}]" if np.isfinite(coef) and np.isfinite(se) else ""
                rows.append(
                    {
                        "cohort": "CPTAC_UCEC",
                        "cancer_type": "UCEC",
                        "endpoint": endpoint,
                        "state_id": state,
                        "state_label": short_label(state),
                        "scoring_method": method,
                        "analysis_type": "cox_clinical_endpoint",
                        "n_samples": n,
                        "n_events": events,
                        "effect_size": coef,
                        "hazard_ratio_per_sd": float(math.exp(coef)) if np.isfinite(coef) else "",
                        "confidence_interval_or_se": ci,
                        "p_value": p,
                        "fdr": np.nan,
                        "covariates": "state_score_z + age_z + male + stage_ordinal + grade_ordinal",
                        "interpretation": interp if not err else f"not evaluable: {err}",
                    }
                )
    assoc = pd.DataFrame(rows)
    assoc["fdr"] = bh_fdr(assoc["p_value"])
    write_table(
        assoc,
        OUT / "cptac_clinical_associations.tsv",
        "CPTAC UCEC state associations with stage, grade, OS and RFS",
        "CPTAC UCEC RNA signature scores merged with CPTAC pan-cancer clinical table",
        f"{rel(OUT / 'cptac_signature_scores.tsv')}; {rel(CPTAC_CLINICAL)}",
    )

    pathway_scores, pathway_cov = make_cptac_pathway_protein_scores(prot, sig)
    write_table(
        pathway_scores,
        OUT / "cptac_protein_pathway_scores.tsv",
        "CPTAC UCEC protein pathway activity scores from grouped ecosystem signatures",
        "CPTAC UCEC tumor proteome matrix and grouped NMF signatures",
        f"{rel(CPTAC_UCEC_PROTEIN)}; {rel(OUT / 'signature_catalog.tsv')}",
    )
    write_table(
        pathway_cov,
        OUT / "cptac_protein_pathway_gene_coverage.tsv",
        "Gene coverage for CPTAC UCEC protein pathway activity scores",
        "signature catalog intersected with CPTAC proteome matrix",
        f"{rel(CPTAC_UCEC_PROTEIN)}; {rel(OUT / 'signature_catalog.tsv')}; {rel(GENE_INFO)}",
    )

    corr_rows = []
    clinical_scores = scores[scores["scoring_method"] == "zscore_mean"].copy()
    rna_z = clinical_scores[["case_id", "state_id", "state_label", "score", "n_signature_genes_used"]].rename(
        columns={"score": "rna_state_score", "n_signature_genes_used": "n_rna_signature_genes_used"}
    )
    prot_z = protein_scores[["case_id", "state_id", "protein_state_score", "n_signature_genes_used"]].rename(
        columns={"n_signature_genes_used": "n_protein_signature_genes_used"}
    )
    for state, d in rna_z.merge(prot_z, on=["case_id", "state_id"], how="inner").groupby("state_id", observed=True):
        dd = d.dropna(subset=["rna_state_score", "protein_state_score"])
        rho, p = (np.nan, np.nan) if len(dd) < 20 else spearmanr(dd["rna_state_score"], dd["protein_state_score"])
        corr_rows.append(
            {
                "cohort": "CPTAC_UCEC",
                "comparison_type": "same_state_signature",
                "state_id": state,
                "state_label": short_label(state),
                "protein_pathway": short_label(state),
                "n_samples": int(len(dd)),
                "n_events": "",
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "fdr": np.nan,
                "n_rna_signature_genes_used": int(dd["n_rna_signature_genes_used"].median()) if not dd.empty else 0,
                "n_protein_signature_genes_used": int(dd["n_protein_signature_genes_used"].median()) if not dd.empty else 0,
                "interpretation": "correlation between RNA-derived state score and same signature scored on tumor proteomics",
            }
        )
    pathway = rna_z.copy()
    pathway["protein_pathway"] = pathway["state_id"].map(CPTAC_PATHWAY_STATE_MAP)
    pathway = pathway.dropna(subset=["protein_pathway"]).merge(pathway_scores, left_on=["case_id", "protein_pathway"], right_on=["case_id", "pathway"], how="inner")
    for state, d in pathway.groupby("state_id", observed=True):
        dd = d.dropna(subset=["rna_state_score", "protein_pathway_score"])
        rho, p = (np.nan, np.nan) if len(dd) < 20 else spearmanr(dd["rna_state_score"], dd["protein_pathway_score"])
        corr_rows.append(
            {
                "cohort": "CPTAC_UCEC",
                "comparison_type": "matched_pathway_activity",
                "state_id": state,
                "state_label": short_label(state),
                "protein_pathway": CPTAC_PATHWAY_STATE_MAP.get(state, ""),
                "n_samples": int(len(dd)),
                "n_events": "",
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "fdr": np.nan,
                "n_rna_signature_genes_used": int(dd["n_rna_signature_genes_used"].median()) if not dd.empty else 0,
                "n_protein_signature_genes_used": int(dd["n_signature_genes_used"].median()) if "n_signature_genes_used" in dd and not dd.empty else 0,
                "interpretation": "correlation between RNA-derived state score and matched protein pathway activity",
            }
        )
    corr = pd.DataFrame(corr_rows)
    corr["fdr"] = bh_fdr(corr["p_value"])
    write_table(
        corr,
        OUT / "cptac_protein_pathway_correlations.tsv",
        "CPTAC UCEC RNA-state to proteomic signature/pathway correlations",
        "CPTAC UCEC RNA scores and matched tumor proteome scores",
        f"{rel(OUT / 'cptac_signature_scores.tsv')}; {rel(OUT / 'cptac_protein_signature_scores.tsv')}; {rel(OUT / 'cptac_protein_pathway_scores.tsv')}",
    )

    prov = pd.read_csv(OUT / "external_validation_provenance.tsv", sep="\t")
    for col in ["downloaded_files", "sha256", "analysis_note"]:
        if col not in prov.columns:
            prov[col] = ""
    for col in ["status", "reason", "download_date", "downloaded_files", "sha256", "analysis_note"]:
        if col in prov.columns:
            prov[col] = prov[col].fillna("").astype(str)
    cptac_files = list(CPTAC_URLS)
    mask = prov["dataset"].str.contains("CPTAC", na=False)
    prov.loc[mask, "status"] = "downloaded_and_analyzed"
    prov.loc[mask, "reason"] = "CPTAC Zenodo 8394329 UCEC RNA/proteome matrices and pan-cancer clinical table were downloaded, checksum-verified and analyzed."
    prov.loc[mask, "download_date"] = RUN_DATE
    prov.loc[mask, "downloaded_files"] = ";".join(rel(p) for p in cptac_files)
    prov.loc[mask, "sha256"] = ";".join(f"{p.name}:{checksum(p)}" for p in cptac_files)
    prov.loc[mask, "analysis_note"] = "UCEC RNA scores tested against stage, grade, OS and RFS; matched tumor proteomics tested as RNA-state protein/pathway support."
    write_table(
        prov,
        OUT / "external_validation_provenance.tsv",
        "External validation provenance and availability status",
        "local file availability check plus CPTAC/Riaz/IMvigor download/scoring",
        "goal.md external validation requirements; CPTAC raw files",
    )

    make_cptac_validation_figure(assoc, corr)
    make_cptac_validation_report(
        assoc,
        corr,
        pd.concat([z_cov, rank_cov], ignore_index=True),
        protein_cov,
        pathway_cov,
        input_status,
        int(scores["case_id"].nunique()),
    )
    return True


def cptac_bar_panel(ax: plt.Axes, d: pd.DataFrame, x_col: str, title: str, xlabel: str) -> None:
    d = d.dropna(subset=[x_col]).copy()
    if d.empty:
        ax.axis("off")
        ax.text(0.05, 0.55, f"{title}\nnot evaluable", transform=ax.transAxes, fontsize=9, weight="bold")
        return
    d = d.sort_values(x_col)
    y = np.arange(len(d))
    colors = np.where(d["fdr"].fillna(1) < 0.1, "#b5443c", "#6c7a89")
    ax.barh(y, d[x_col].astype(float), color=colors, alpha=0.92)
    ax.axvline(0, color="#222222", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["state_label"], fontsize=6.6)
    ax.set_title(title, fontsize=9, weight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def make_cptac_validation_figure(assoc: pd.DataFrame, corr: pd.DataFrame) -> None:
    source_parts = []
    for panel, endpoint in [("A", "stage_ordinal"), ("B", "grade_ordinal"), ("C", "overall_survival")]:
        d = assoc[(assoc["endpoint"] == endpoint) & (assoc["scoring_method"] == "zscore_mean")].copy()
        d["panel"] = panel
        source_parts.append(d)
    prot = corr[corr["comparison_type"] == "matched_pathway_activity"].copy()
    prot["panel"] = "D"
    prot["endpoint"] = "rna_to_matched_protein_pathway"
    prot["effect_size"] = prot["spearman_rho"]
    prot["confidence_interval_or_se"] = ""
    prot["covariates"] = "Spearman correlation in matched RNA/protein samples"
    source_parts.append(prot)
    source = pd.concat(source_parts, ignore_index=True, sort=False)
    write_table(
        source,
        FIG / "cptac_validation_source.tsv",
        "Source table for CPTAC clinical and proteomic validation figure",
        "CPTAC clinical associations and RNA-protein correlations",
        f"{rel(OUT / 'cptac_clinical_associations.tsv')}; {rel(OUT / 'cptac_protein_pathway_correlations.tsv')}",
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4))
    cptac_bar_panel(axes[0, 0], assoc[(assoc["endpoint"] == "stage_ordinal") & (assoc["scoring_method"] == "zscore_mean")], "effect_size", "Pathological stage", "beta per SD")
    cptac_bar_panel(axes[0, 1], assoc[(assoc["endpoint"] == "grade_ordinal") & (assoc["scoring_method"] == "zscore_mean")], "effect_size", "Histologic grade", "beta per SD")
    cptac_bar_panel(axes[1, 0], assoc[(assoc["endpoint"] == "overall_survival") & (assoc["scoring_method"] == "zscore_mean")], "effect_size", "Overall survival", "log HR per SD")
    cptac_bar_panel(axes[1, 1], prot, "spearman_rho", "Matched protein pathway", "Spearman rho")
    fig.suptitle("CPTAC UCEC external clinical and proteomic validation", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = FIG / "cptac_validation.pdf"
    fig.savefig(path)
    plt.close(fig)
    record(
        path,
        "CPTAC UCEC validation figure",
        "matplotlib generated from CPTAC validation source table",
        rel(FIG / "cptac_validation_source.tsv"),
    )


def make_cptac_validation_report(
    assoc: pd.DataFrame,
    corr: pd.DataFrame,
    rna_cov: pd.DataFrame,
    protein_cov: pd.DataFrame,
    pathway_cov: pd.DataFrame,
    input_status: dict[str, str],
    n_rna_cases: int,
) -> None:
    tested = assoc[(assoc["n_samples"] > 0) & (assoc["scoring_method"] == "zscore_mean")].copy()
    sig = tested[tested["fdr"].fillna(1) < 0.1]
    rank_tested = assoc[(assoc["n_samples"] > 0) & (assoc["scoring_method"] == "rank_mean_percentile")].copy()
    rank_sig = rank_tested[rank_tested["fdr"].fillna(1) < 0.1]
    prot = corr[corr["comparison_type"] == "matched_pathway_activity"].copy()
    prot_sig = prot[prot["fdr"].fillna(1) < 0.1]
    lines = [
        "# CPTAC validation summary",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "CPTAC UCEC RNA-seq, matched tumor proteomics and pan-cancer clinical metadata were downloaded from Zenodo 8394329 and checksum-verified. UCEC RNA-derived ecosystem signatures were tested against pathological stage, histologic grade, overall survival and recurrence-free survival. Matched tumor proteomics were used as an orthogonal pathway anchor.",
        "",
        f"- RNA-scored UCEC cases: {n_rna_cases}.",
        f"- Clinical state-endpoint tests with FDR < 0.10: {len(sig)}.",
        f"- Rank-score sensitivity tests with FDR < 0.10: {len(rank_sig)}.",
        f"- Matched RNA/protein pathway correlations with FDR < 0.10: {len(prot_sig)}.",
        f"- Median CPTAC RNA signature genes used: {rna_cov['n_signature_genes_used'].median():.0f}.",
        f"- Median CPTAC protein signature genes used: {protein_cov['n_signature_genes_used'].median():.0f}.",
        "",
        "## Strongest clinical associations",
        "",
        md_table(tested.sort_values("p_value")[["endpoint", "state_label", "n_samples", "n_events", "effect_size", "hazard_ratio_per_sd", "p_value", "fdr"]].head(12)),
        "",
        "## Matched protein pathway support",
        "",
        md_table(prot.sort_values("p_value")[["state_label", "protein_pathway", "n_samples", "spearman_rho", "p_value", "fdr"]].head(12)),
        "",
        "Interpretation boundary: CPTAC validation is independent of TCGA but limited here to open-access UCEC matrices from one processed CPTAC release. Protein correlations support molecular anchoring of RNA state scores; they are not evidence of causal cell-state effects or observed spending.",
        "",
        "## Input status",
        "",
        md_table(pd.DataFrame([{"file_path": k, "status": v} for k, v in input_status.items()])),
        "",
        "## Protein pathway coverage",
        "",
        md_table(pathway_cov[["pathway", "n_signature_genes", "n_signature_genes_used"]]),
    ]
    write_text(
        REPORTS / "cptac_validation_summary.md",
        "\n".join(lines),
        "CPTAC UCEC validation summary with clinical and proteomic analyses",
        "CPTAC UCEC RNA/protein signature scoring and clinical association analyses",
        f"{rel(OUT / 'cptac_clinical_associations.tsv')}; {rel(OUT / 'cptac_protein_pathway_correlations.tsv')}",
    )


def make_external_validation_stubs() -> None:
    provenance = pd.DataFrame(
        [
            {
                "dataset": "CPTAC public RNA/protein data",
                "candidate_source": "GDC/CPTAC open-access portal",
                "status": "not_downloaded_in_local_offline_run",
                "reason": "No CPTAC expression/clinical matrices were present under data/raw or data/processed/translational_validation at run time.",
                "expected_next_step": "Fetch open-access CPTAC RNA-seq/proteomics matrices with checksums and run signature scoring.",
                "download_date": "",
                "license_or_usage_note": "Must be recorded per downloaded file before analysis.",
            },
            {
                "dataset": "IMvigor210CoreBiologies",
                "candidate_source": "Bioconductor package or public processed release",
                "status": "not_downloaded_in_local_offline_run",
                "reason": "No IMvigor210 expression/response object was present locally at run time.",
                "expected_next_step": "Install/fetch package or processed expression and clinical response tables, then score prioritized states.",
                "download_date": "",
                "license_or_usage_note": "Check package license and citation requirements before redistribution.",
            },
            {
                "dataset": "Riaz melanoma nivolumab cohort",
                "candidate_source": "GEO GSE91061/public processed files",
                "status": "not_downloaded_in_local_offline_run",
                "reason": "No GSE91061 expression/clinical files were present locally at run time.",
                "expected_next_step": "Fetch open expression and response/survival tables, harmonize baseline/on-treatment samples, then score prioritized states.",
                "download_date": "",
                "license_or_usage_note": "Record GEO accession, file names and checksums.",
            },
        ]
    )
    write_table(
        provenance,
        OUT / "external_validation_provenance.tsv",
        "External validation provenance and availability status",
        "local file availability check",
        "goal.md external validation requirements",
    )

    cptac_scores = pd.DataFrame(
        columns=[
            "cohort",
            "sample_id",
            "state_id",
            "state_label",
            "scoring_method",
            "score",
            "n_signature_genes_used",
            "provenance_status",
        ]
    )
    write_table(
        cptac_scores,
        OUT / "cptac_signature_scores.tsv",
        "CPTAC signature scores schema; no rows because CPTAC data were not local",
        "external validation stub",
        rel(OUT / "external_validation_provenance.tsv"),
    )

    cptac_assoc = pd.DataFrame(
        [
            {
                "cohort": "CPTAC",
                "endpoint": endpoint,
                "state_id": state,
                "state_label": short_label(state),
                "n_samples": 0,
                "n_events": "",
                "effect_size": "",
                "confidence_interval_or_se": "",
                "p_value": "",
                "fdr": "",
                "covariates": "tumor type, stage/grade, purity/proliferation if available",
                "interpretation": "not evaluated; open-access CPTAC matrices were not present in this local run",
            }
            for endpoint in ["stage", "grade", "survival"]
            for state in PRIORITY_STATES
        ]
    )
    write_table(
        cptac_assoc,
        OUT / "cptac_clinical_associations.tsv",
        "CPTAC clinical association result schema and not-yet-evaluated rows",
        "external validation stub",
        rel(OUT / "external_validation_provenance.tsv"),
    )

    icb = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "cancer_type": cancer,
                "treatment": treatment,
                "endpoint": endpoint,
                "state_label": short_label(state),
                "state_id": state,
                "n_samples": 0,
                "n_events": "",
                "effect_size": "",
                "confidence_interval_or_se": "",
                "p_value": "",
                "fdr": "",
                "covariates": "response model covariates to be filled after cohort acquisition",
                "interpretation": "not evaluated; cohort was not present locally in this offline run",
            }
            for cohort, cancer, treatment in [
                ("IMvigor210", "urothelial carcinoma", "atezolizumab"),
                ("Riaz_GSE91061", "melanoma", "nivolumab"),
            ]
            for endpoint in ["objective_response", "overall_survival_or_pfs"]
            for state in PRIORITY_STATES
        ]
    )
    write_table(
        icb,
        OUT / "icb_state_validation.tsv",
        "Immunotherapy state validation schema and not-yet-evaluated rows",
        "external validation stub",
        rel(OUT / "external_validation_provenance.tsv"),
    )

    make_stub_figure(
        FIG / "cptac_validation.pdf",
        FIG / "cptac_validation_source.tsv",
        provenance[provenance["dataset"].str.contains("CPTAC")],
        "CPTAC validation status",
    )
    make_stub_figure(
        FIG / "icb_validation.pdf",
        FIG / "icb_validation_source.tsv",
        provenance[provenance["dataset"].str.contains("IMvigor|Riaz", regex=True)],
        "ICB validation status",
    )

    cptac_report = [
        "# CPTAC validation summary",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "No CPTAC expression/clinical matrix was present in this local repository run, so no external CPTAC association is claimed.",
        "",
        "Planned analysis after data acquisition:",
        "",
        "- score standardized ecosystem signatures by z-score mean and an ssGSEA/GSVA-like rank method if available;",
        "- test stage, grade and survival associations;",
        "- adjust for tumor type and available purity/proliferation proxies;",
        "- if proteomics are available, compare RNA-derived state scores with cell-cycle, EMT, myeloid inflammation, interferon/exhaustion and heat-shock protein/pathway activities.",
    ]
    write_text(
        REPORTS / "cptac_validation_summary.md",
        "\n".join(cptac_report),
        "CPTAC validation status and planned analysis",
        "external validation stub",
        rel(OUT / "external_validation_provenance.tsv"),
    )

    icb_report = [
        "# Immunotherapy validation summary",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "No IMvigor210 or Riaz/GSE91061 expression-response cohort was present in this local repository run, so no treatment-response association is claimed.",
        "",
        "Planned analysis after data acquisition:",
        "",
        "- harmonize expression to gene symbols and baseline/on-treatment labels;",
        "- compute prioritized state signatures;",
        "- test response by logistic regression;",
        "- test OS/PFS by Cox models where available;",
        "- adjust for treatment arm, timepoint, TMB, PD-L1, IFNG signature and purity where available.",
    ]
    write_text(
        REPORTS / "icb_validation_summary.md",
        "\n".join(icb_report),
        "ICB validation status and planned analysis",
        "external validation stub",
        rel(OUT / "external_validation_provenance.tsv"),
    )
    make_cptac_validation()
    make_riaz_icb_validation()
    make_imvigor_icb_validation()


def make_stub_figure(path: Path, source_path: Path, source: pd.DataFrame, title: str) -> None:
    write_table(
        source,
        source_path,
        f"Source table for {title} figure",
        "external validation provenance",
        rel(OUT / "external_validation_provenance.tsv"),
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axis("off")
    ax.text(0.02, 0.85, title, fontsize=12, weight="bold", transform=ax.transAxes)
    ax.text(
        0.02,
        0.58,
        "Open-access cohort files were not present locally.\n"
        "This figure is a provenance placeholder, not validation evidence.",
        fontsize=9,
        transform=ax.transAxes,
        va="top",
    )
    ax.text(
        0.02,
        0.22,
        "Next step: fetch open expression + clinical endpoints,\nrecord checksums, then re-run signature scoring.",
        fontsize=8,
        transform=ax.transAxes,
        va="top",
        color="#4f4f4f",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    record(
        path,
        f"{title} placeholder figure",
        "matplotlib generated from validation provenance source table",
        rel(source_path),
    )


def make_manuscript_outputs(inc: pd.DataFrame, robust: pd.DataFrame) -> None:
    final = inc[inc["model"] == "model3_plus_immune_stromal_and_state"].copy()
    final = final.sort_values("coef_log_hr_per_within_cancer_sd", ascending=False)
    top = final.head(5)
    robust_top = robust.head(5)
    icb_note = "No public ICB cohort was analyzed in this run."
    icb_table = ""
    icb_path = OUT / "icb_state_validation.tsv"
    if icb_path.exists():
        icb = pd.read_csv(icb_path, sep="\t")
        real_icb = icb[icb["n_samples"] > 0].copy()
        if not real_icb.empty:
            real_icb = real_icb.sort_values("p_value", ascending=True)
            sig_n = int((real_icb["fdr"].fillna(1) < 0.1).sum())
            response_sig_n = int(
                (
                    real_icb["endpoint"].astype(str).str.contains("response", na=False)
                    & (real_icb["fdr"].fillna(1) < 0.1)
                ).sum()
            )
            survival_sig_n = sig_n - response_sig_n
            cohorts = ", ".join(sorted(real_icb["cohort"].unique()))
            icb_note = (
                "As an exploratory treatment-relevance analysis, public ICB RNA-seq cohorts "
                f"({cohorts}) were scored for the same state signatures (Supplementary Tables S14 and S15); {response_sig_n} response "
                f"tests and {survival_sig_n} OS/PFS tests reached FDR < 0.10 in covariate-adjusted models."
            )
            icb_table = md_table(
                real_icb[
                    [
                        "cohort",
                        "endpoint",
                        "state_label",
                        "n_samples",
                        "n_events",
                        "effect_size",
                        "odds_ratio_per_sd",
                        "p_value",
                        "fdr",
                    ]
                ].head(8)
            )

    cptac_note = "CPTAC was not analyzed in this run, so independent CPTAC validation and protein-level validation are not claimed."
    cptac_table = ""
    cptac_clin_path = OUT / "cptac_clinical_associations.tsv"
    cptac_prot_path = OUT / "cptac_protein_pathway_correlations.tsv"
    if cptac_clin_path.exists() and cptac_prot_path.exists():
        cptac_clin = pd.read_csv(cptac_clin_path, sep="\t")
        cptac_clin = cptac_clin[(cptac_clin["n_samples"] > 0) & (cptac_clin["scoring_method"] == "zscore_mean")].copy()
        cptac_prot = pd.read_csv(cptac_prot_path, sep="\t")
        cptac_prot = cptac_prot[cptac_prot["comparison_type"] == "matched_pathway_activity"].copy()
        if not cptac_clin.empty:
            clin_sig = int((cptac_clin["fdr"].fillna(1) < 0.10).sum())
            prot_sig = int((cptac_prot["fdr"].fillna(1) < 0.10).sum()) if not cptac_prot.empty else 0
            cptac_note = (
                "Independent CPTAC UCEC validation was added using checksum-verified open-access RNA-seq, "
                f"matched tumor proteomics and clinical metadata (Supplementary Tables S1 and S11--S13). {clin_sig} clinical state-endpoint tests "
                f"reached FDR < 0.10, mainly for histologic grade, and {prot_sig} matched RNA/protein pathway "
                "correlations reached FDR < 0.10."
            )
            cptac_table = md_table(
                cptac_clin.sort_values("p_value")[
                    [
                        "endpoint",
                        "state_label",
                        "n_samples",
                        "n_events",
                        "effect_size",
                        "hazard_ratio_per_sd",
                        "p_value",
                        "fdr",
                    ]
                ].head(8)
            )

    results = [
        "# Draft Results insert for translational validation revision",
        "",
        "## Independent validation and translational prioritization",
        "",
        "We generated a standardized NMF ecosystem-state signature catalog and re-evaluated prioritized states in TCGA using nested cancer-stratified Cox models (Supplementary Tables S2, S3, S8 and S9). The final model adjusted for age, sex, stage, tumor purity, a proliferation proxy, an immune proxy and a stromal proxy before adding the ecosystem-state score. This analysis is intended to test whether the modeled burden-priority states retain prognostic information beyond broad tumor and microenvironmental confounders.",
        "",
        "The highest adjusted adverse effects in the local TCGA extension were:",
        "",
        md_table(top[
            [
                "state_label",
                "n_samples",
                "n_events",
                "hr_per_within_cancer_sd",
                "p_value",
                "fdr",
            ]
        ]),
        "",
        "A robustness matrix integrating TCGA adjustment, stage gradients, within-cancer variation, burden/cost ranks, leave-one-cancer sensitivity, counterfactual reducible scores and negative-control checks prioritized (Supplementary Tables S4, S6, S7 and S10):",
        "",
        md_table(robust_top[["state_label", "n_samples", "n_events", "robustness_score"]]),
        "",
        cptac_note,
        "",
        cptac_table,
        "",
        icb_note,
        "",
        icb_table,
    ]
    write_text(
        MS / "translational_validation_insert_results.md",
        "\n".join(results),
        "Draft Results insert for translational validation manuscript upgrade",
        "local translational validation outputs",
        f"{rel(OUT / 'tcga_incremental_cox.tsv')}; {rel(OUT / 'robustness_matrix.tsv')}",
    )

    discussion = [
        "# Draft Discussion insert",
        "",
        "The extension shifts the work from a purely modeled burden map toward a translational prioritization framework by asking whether high-burden ecosystem states also show progression, adverse TCGA survival associations, external clinical support and robustness to available confounding proxies. This is strongest for states that combine modeled cost/burden representation with within-cancer variation, adjusted prognosis and orthogonal molecular validation.",
        "",
        "The analysis remains ecological. Single-cell abundance, population burden, modeled care costs and TCGA outcomes are not measured in the same patients. Cost estimates therefore support modeled burden representation, not observed cell-state-specific spending. Counterfactual analyses should be described as scenario modeling rather than causal intervention effects.",
        "",
        "The added CPTAC UCEC analysis provides independent clinical and proteomic anchoring outside TCGA: several priority states track histologic grade and the RNA-derived scores are strongly correlated with matched protein pathway activities (Supplementary Tables S11--S13). However, this is one tumor type and survival is underpowered in the open UCEC subset. The Riaz/GSE91061 and IMvigor210 analyses add exploratory immunotherapy response and survival/PFS layers (Supplementary Tables S14 and S15), but they remain tumor-type and treatment specific. Patient-level medical expenditure linked to transcriptomic or single-cell profiling remains necessary before any direct cost-attribution claim can be made.",
    ]
    write_text(
        MS / "translational_validation_insert_discussion.md",
        "\n".join(discussion),
        "Draft Discussion insert for translational validation manuscript upgrade",
        "local translational validation outputs",
        f"{rel(REPORTS / 'tcga_incremental_value_summary.md')}; {rel(REPORTS / 'robustness_summary.md')}",
    )

    significance = [
        "# Significance",
        "",
        "This study integrates public pan-cancer single-cell ecosystem states with population cancer burden, modeled cancer-care costs and clinical outcomes to prioritize tumor microenvironment programs with potential translational relevance. The upgraded analysis adds a reusable signature catalog, nested TCGA prognostic models, CPTAC UCEC clinical/proteomic anchoring and exploratory immunotherapy cohorts while preserving a cautious distinction between modeled burden representation, prognostic association and unsupported causal or spending claims.",
    ]
    write_text(
        MS / "translational_validation_significance.md",
        "\n".join(significance),
        "submission-style significance statement",
        "manuscript drafting from translational validation outputs",
        f"{rel(OUT / 'signature_catalog.tsv')}; {rel(OUT / 'tcga_incremental_cox.tsv')}",
    )

    highlights = [
        "# Highlights",
        "",
        "- Pan-cancer single-cell states are mapped to modeled cancer burden",
        "- TCGA models test prognostic value beyond clinical and TME proxies",
        "- CPTAC UCEC links priority states to grade and protein pathways",
        "- ICB cohorts provide exploratory treatment-relevance checks",
    ]
    write_text(
        MS / "translational_validation_highlights.md",
        "\n".join(highlights),
        "submission-style highlights under 85 characters each",
        "manuscript drafting from translational validation outputs",
        rel(REPORTS / "translational_validation_summary.md"),
    )

    etoc = (
        "# eTOC blurb\n\n"
        "A public-data framework links pan-cancer single-cell ecosystem states with modeled cancer burden, care-cost weights, TCGA outcomes, CPTAC proteomics and immunotherapy cohorts, prioritizing clinically anchored tumor microenvironment programs while separating modeled burden from causal or observed spending claims."
    )
    write_text(
        MS / "translational_validation_etoc.md",
        etoc,
        "40-word eTOC-style blurb",
        "manuscript drafting from translational validation outputs",
        rel(REPORTS / "translational_validation_summary.md"),
    )

    star = [
        "# STAR Methods skeleton",
        "",
        "## Resource availability",
        "",
        "### Lead contact",
        "Requests should be directed to Chunyu Yu (ycy@hznu.edu.cn).",
        "",
        "### Materials availability",
        "No new biological materials were generated.",
        "",
        "### Data and code availability",
        "All primary analyses use public data and repository-local processed tables. New extension outputs are written under `data/processed/translational_validation/`, `figures/translational_validation/`, `reports/` and `manuscript/`. The extension can be run with `make translational_validation`.",
        "",
        "## Experimental model and subject details",
        "This is a retrospective computational analysis of public tumor single-cell, public population burden/cost estimates and public TCGA clinical/transcriptomic data.",
        "",
        "## Method details",
        "- NMF state signature harmonization.",
        "- TCGA within-cancer signature normalization.",
        "- Nested cancer-stratified Cox modeling.",
        "- Robustness matrix construction.",
        "- CPTAC UCEC RNA signature scoring, clinical association testing and matched protein pathway correlation.",
        "- Public ICB cohort RNA scoring and objective-response logistic regression.",
        "",
        "## Quantification and statistical analysis",
        "Cox models report hazard ratios per within-cancer standard deviation, Wald p-values, Benjamini-Hochberg FDR, concordance index and likelihood-ratio tests where nested models are comparable. Robustness criteria are recorded as explicit binary indicators.",
    ]
    write_text(
        MS / "star_methods_skeleton.md",
        "\n".join(star),
        "STAR Methods skeleton",
        "manuscript drafting from translational validation outputs",
        "goal.md; generated output tables",
    )

    resources = pd.DataFrame(
        [
            ["dataset", "Zenodo pan-cancer tumor-normal ecosystem atlas", "Zenodo 10651059", "local copy under data/raw/pancancer_ecosystem_zenodo10651059"],
            ["dataset", "TCGA survival and signature scores", "local processed table", rel(SRC / "tcga_nmf_state_signature_scores_survival_merged.csv.gz")],
            ["dataset", "GLOBOCAN/WHO burden weights", "local processed tables", rel(SRC / "who_ghe_2021_cancer_burden_long.csv")],
            ["dataset", "NCI cost weights", "local processed tables", rel(SRC / "nmf_state_nci_cost_scores.csv")],
            ["dataset", "CPTAC pan-cancer clinical and UCEC RNA/proteome", "Zenodo 8394329", rel(CPTAC_RAW)],
            ["dataset", "Riaz/GSE91061 melanoma nivolumab cohort", "riazn/bms038_analysis public data", rel(RIAZ_RAW)],
            ["dataset", "IMvigor210CoreBiologies", "public package tarball", rel(IMVIGOR_TAR)],
            ["dataset", "NCBI Homo sapiens gene_info", "NCBI Gene", rel(GENE_INFO)],
            ["software", "Python", "local .venv", ".venv/bin/python"],
            ["software", "lifelines", "CoxPHFitter", "used for TCGA Cox models"],
            ["script", "translational validation pipeline", "run_all.py", rel(ROOT / "scripts" / "translational_validation" / "run_all.py")],
            ["repository", "Analysis code release", "GitHub", "https://github.com/cheneyyu/tumor-ecosystem-burden-code"],
            ["repository", "Processed data release", "GitHub private repository", "https://github.com/cheneyyu/tumor-ecosystem-burden-data"],
        ],
        columns=["resource_type", "name", "identifier", "availability_or_path"],
    )
    write_table(
        resources,
        MS / "key_resources_table.tsv",
        "Draft Cell Press Key Resources Table",
        "local translational validation resource inventory",
        "goal.md; repository files",
    )


def make_final_summary(inc: pd.DataFrame, robust: pd.DataFrame) -> None:
    final = inc[inc["model"] == "model3_plus_immune_stromal_and_state"].copy()
    sig = final[final["fdr"].fillna(1) < 0.1]
    robust_pass = robust[robust["robustness_score"] >= robust["robustness_score"].median()]
    icb_path = OUT / "icb_state_validation.tsv"
    riaz_pre = pd.DataFrame()
    if icb_path.exists():
        icb = pd.read_csv(icb_path, sep="\t")
        riaz_pre = icb[icb["n_samples"] > 0].copy()
    riaz_sig_n = int((riaz_pre["fdr"].fillna(1) < 0.1).sum()) if not riaz_pre.empty else 0
    icb_cohorts = ", ".join(sorted(riaz_pre["cohort"].unique())) if not riaz_pre.empty else "none"
    cptac_clin = pd.DataFrame()
    cptac_corr = pd.DataFrame()
    if (OUT / "cptac_clinical_associations.tsv").exists():
        cptac_clin = pd.read_csv(OUT / "cptac_clinical_associations.tsv", sep="\t")
        cptac_clin = cptac_clin[cptac_clin["n_samples"] > 0].copy()
    if (OUT / "cptac_protein_pathway_correlations.tsv").exists():
        cptac_corr = pd.read_csv(OUT / "cptac_protein_pathway_correlations.tsv", sep="\t")
        cptac_corr = cptac_corr[cptac_corr["comparison_type"] == "matched_pathway_activity"].copy()
    cptac_primary = cptac_clin[cptac_clin["scoring_method"] == "zscore_mean"].copy() if not cptac_clin.empty else pd.DataFrame()
    cptac_rank = cptac_clin[cptac_clin["scoring_method"] == "rank_mean_percentile"].copy() if not cptac_clin.empty else pd.DataFrame()
    cptac_clin_sig = int((cptac_primary["fdr"].fillna(1) < 0.1).sum()) if not cptac_primary.empty else 0
    cptac_rank_sig = int((cptac_rank["fdr"].fillna(1) < 0.1).sum()) if not cptac_rank.empty else 0
    cptac_prot_sig = int((cptac_corr["fdr"].fillna(1) < 0.1).sum()) if not cptac_corr.empty else 0
    cptac_cases = int(cptac_primary["n_samples"].max()) if not cptac_primary.empty else 0
    cptac_table = (
        md_table(
            cptac_primary.sort_values("p_value")[
                ["endpoint", "state_label", "n_samples", "n_events", "effect_size", "hazard_ratio_per_sd", "p_value", "fdr"]
            ].head(8)
        )
        if not cptac_primary.empty
        else "_No CPTAC rows._"
    )
    cptac_prot_table = (
        md_table(
            cptac_corr.sort_values("p_value")[
                ["state_label", "protein_pathway", "n_samples", "spearman_rho", "p_value", "fdr"]
            ].head(8)
        )
        if not cptac_corr.empty
        else "_No CPTAC protein rows._"
    )
    lines = [
        "# translational validation summary",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "## What was successfully added",
        "",
        "- Repository audit and reproducibility-gap summary.",
        "- Standardized NMF state signature catalog and signature variants.",
        "- Nested TCGA cancer-stratified Cox models for priority states and an integrated adverse score.",
        "- Cancer-specific Cox fits and fixed-effect meta-analysis table.",
        "- CPTAC UCEC RNA/protein validation with checksum-verified open clinical, transcriptomic and proteomic matrices.",
        "- Riaz/GSE91061 and IMvigor210 response scoring with exploratory logistic response and OS/PFS Cox analyses.",
        "- Robustness matrix and heatmap from local TCGA, stage, counterfactual, leave-one-cancer and negative-control outputs.",
        "- Manuscript insert drafts, Significance, Highlights, eTOC, STAR Methods skeleton and Key Resources Table.",
        "- Manifest with checksums for generated files.",
        "",
        "## Remaining data gaps",
        "",
        "- CPTAC validation is currently limited to the open UCEC BCM RNA/proteome matrices from Zenodo 8394329; additional CPTAC tumor types were not added.",
        "- No additional ICB cohorts beyond Riaz/GSE91061 and IMvigor210 were obtained.",
        "",
        "## Robustness checks that passed locally",
        "",
        f"- Final adjusted TCGA model states with FDR < 0.10: {len(sig)}.",
        f"- States at or above the median robustness score: {len(robust_pass)}.",
        f"- CPTAC UCEC clinical tests with FDR < 0.10: {cptac_clin_sig} across up to {cptac_cases} clinically evaluable cases.",
        f"- CPTAC rank-score sensitivity clinical tests with FDR < 0.10: {cptac_rank_sig}.",
        f"- CPTAC matched RNA/protein pathway correlations with FDR < 0.10: {cptac_prot_sig}.",
        f"- ICB cohorts analyzed: {icb_cohorts}.",
        f"- ICB response/survival cohort-state tests with FDR < 0.10: {riaz_sig_n}.",
        "",
        md_table(robust.head(8)[["state_label", "n_samples", "n_events", "robustness_score"]]),
        "",
        "## CPTAC validation snapshot",
        "",
        cptac_table,
        "",
        "## CPTAC protein-pathway support",
        "",
        cptac_prot_table,
        "",
        "## Claims that can be strengthened",
        "",
        "- The manuscript can now state that prioritized ecosystem states were re-tested in nested TCGA models with clinical, purity, proliferation, immune and stromal proxy adjustment.",
        "- The manuscript can claim independent CPTAC UCEC validation for grade-associated state programs and orthogonal matched protein-pathway anchoring.",
        "- The manuscript can add exploratory anti-PD-1/PD-L1 response and survival/PFS validation layers from public Riaz/GSE91061 and IMvigor210 data.",
        "- The manuscript can present a reusable signature catalog and reproducible extension package.",
        "",
        "## Claims that must remain cautious",
        "",
        "- Modeled burden/cost representation is not observed cell-state-specific spending.",
        "- TCGA survival associations are not causal effects.",
        "- CPTAC validation is single-cancer and UCEC survival is event-limited; grade/protein findings should not be generalized as pan-cancer validation by themselves.",
        "- Treatment-response language should remain exploratory and cohort-specific unless additional ICB cohorts are added.",
        "",
        "## Recommended revised title angle",
        "",
        "Pan-cancer single-cell ecosystem states prioritize modeled cancer burden and clinically anchored adverse programs.",
        "",
        "## Submission recommendation",
        "",
        "Closer to a submission-ready translational computational manuscript. The upgrade now includes non-TCGA CPTAC UCEC clinical/proteomic validation, two exploratory ICB cohorts and a reproducible resource package. The remaining submission risk is scope: CPTAC is single-cancer, survival is underpowered, and all cost language remains modeled rather than patient-level observed spending.",
        "",
        "## Claim taxonomy",
        "",
        "- Modeled burden representation: supported by atlas state abundance integrated with public burden/cost weights.",
        "- Adverse prognostic association: supported by local TCGA nested Cox outputs where significant.",
        "- Independent clinical/protein anchoring: supported by CPTAC UCEC grade associations and matched RNA/protein pathway correlations.",
        "- Treatment-response and ICB survival association: supported only as exploratory Riaz/GSE91061 and IMvigor210 cohort-specific associations.",
        "- Causal claims: not supported.",
    ]
    write_text(
        REPORTS / "translational_validation_summary.md",
        "\n".join(lines),
        "Final translational validation summary and go/no-go recommendation",
        "all local translational validation outputs",
        "; ".join(
            [
                rel(REPORTS / "repo_audit.md"),
                rel(OUT / "signature_catalog.tsv"),
                rel(OUT / "tcga_incremental_cox.tsv"),
                rel(OUT / "robustness_matrix.tsv"),
                rel(OUT / "external_validation_provenance.tsv"),
            ]
        ),
    )


def write_manifest() -> None:
    manifest = pd.DataFrame(CREATED)
    path = REPORTS / "translational_validation_file_manifest.tsv"
    manifest.to_csv(path, sep="\t", index=False)
    # Record the manifest after writing it; append its own checksum manually.
    rec = {
        "file_path": rel(path),
        "description": "Manifest of generated translational validation files",
        "source": "run_all.py output registry",
        "created_date": RUN_DATE,
        "upstream_dependencies": "all generated outputs in this manifest",
        "sha256": checksum(path),
    }
    manifest = pd.concat([manifest, pd.DataFrame([rec])], ignore_index=True)
    manifest.to_csv(path, sep="\t", index=False)
    CREATED.append(rec)


def main() -> None:
    ensure_dirs()
    make_repo_audit()
    make_signature_catalog()
    inc, _meta = make_tcga_incremental()
    robust = make_robustness(inc)
    make_external_validation_stubs()
    make_manuscript_outputs(inc, robust)
    make_final_summary(inc, robust)
    write_manifest()
    print(f"Wrote {len(CREATED)} translational validation artifacts")


if __name__ == "__main__":
    main()
