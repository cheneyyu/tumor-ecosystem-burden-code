#!/usr/bin/env python3
"""Direct TCGA prognosis tests for selected Zenodo NMF state signatures."""

from __future__ import annotations

import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xenaPython as xena
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TCGA = RAW / "tcga_xena"
NMF = RAW / "pancancer_ecosystem_zenodo10651059" / "NMF_h5ad"
OUT = ROOT / "data" / "processed" / "pancancer_ecosystem"

HUB = "https://pancanatlas.xenahubs.net"
EXPR_DATASET = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"
EXPR_OUT = TCGA / "tcga_pancan_nmf_state_signature_gene_expression.tsv.gz"

NMF_FILES = {
    "myeloid": "myl_NMF.h5ad",
    "T_NK": "tnk_NMF.h5ad",
    "B_plasma": "b_NMF.h5ad",
    "mesenchymal": "mesenchymal_NMF.h5ad",
    "epithelial": "epi_NMF.h5ad",
}

SELECTED_STATES = [
    ("myeloid", "C1QC+ macrophage"),
    ("myeloid", "Cell cycling"),
    ("myeloid", "Heat shock"),
    ("myeloid", "Mast"),
    ("myeloid", "SPP1+ macrophage"),
    ("myeloid", "CXCL9+ macrophage"),
    ("myeloid", "FCN1+ monocyte derived macrophage"),
    ("myeloid", "NLRP3+ monocyte derived macrophage"),
    ("T_NK", "Proliferating T-cell (cell cycling)"),
    ("T_NK", "Treg"),
    ("T_NK", "CD16+ NK-cell"),
    ("T_NK", "Exhausted CD8+ T-cell"),
    ("B_plasma", "Plasma cell"),
    ("mesenchymal", "Myofibroblast"),
    ("mesenchymal", "Desmoplastic fibroblast"),
    ("mesenchymal", "PI16+ fibroblast"),
    ("epithelial", "Cell cycling"),
    ("epithelial", "Complete mesenchymal"),
    ("epithelial", "Stress"),
]

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


def clean_gene(g: str) -> bool:
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", str(g)):
        return False
    bad_prefix = ("AC", "AL", "AP", "LINC", "RP11", "RP13")
    return not str(g).startswith(bad_prefix)


def state_key(compartment: str, state: str) -> str:
    return f"{compartment} | {state}"


def stage_to_ordinal(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    s = str(value).upper()
    if "STAGE IV" in s:
        return 4.0
    if "STAGE III" in s:
        return 3.0
    if "STAGE II" in s:
        return 2.0
    if "STAGE I" in s:
        return 1.0
    return np.nan


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return s * np.nan
    return (s - s.mean()) / sd


def extract_signatures(top_n: int = 12) -> pd.DataFrame:
    out_path = OUT / "zenodo_nmf_state_signature_genes.csv"
    if out_path.exists():
        cached = pd.read_csv(out_path)
        expected = {state_key(c, s) for c, s in SELECTED_STATES}
        if expected.issubset(set(cached["state_key"])):
            return cached
    rows = []
    by_comp: dict[str, list[str]] = {}
    for comp, st in SELECTED_STATES:
        by_comp.setdefault(comp, []).append(st)
    for comp, states in by_comp.items():
        a = ad.read_h5ad(NMF / NMF_FILES[comp])
        x = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        genes = np.array([str(g) for g in a.var_names])
        obs_state = a.obs["Cell_state"].astype(str).to_numpy()
        for st in states:
            mask = obs_state == st
            if mask.sum() < 5:
                continue
            score = x[mask].mean(axis=0) - x[~mask].mean(axis=0)
            order = np.argsort(score)[::-1]
            kept = []
            for idx in order:
                gene = genes[idx]
                if not clean_gene(gene):
                    continue
                kept.append((gene, float(score[idx]), float(x[mask, idx].mean()), float(x[~mask, idx].mean())))
                if len(kept) >= top_n:
                    break
            for rank, (gene, marker_delta, state_mean, rest_mean) in enumerate(kept, 1):
                rows.append(
                    {
                        "state_key": state_key(comp, st),
                        "nmf_compartment": comp,
                        "Cell_state": st,
                        "rank": rank,
                        "gene": gene,
                        "marker_delta": marker_delta,
                        "state_mean": state_mean,
                        "rest_mean": rest_mean,
                        "n_state_modules": int(mask.sum()),
                    }
                )
        del a, x
    sig = pd.DataFrame(rows)
    sig.to_csv(out_path, index=False)
    return sig


def fetch_expression(genes: list[str]) -> pd.DataFrame:
    if EXPR_OUT.exists():
        cached = pd.read_csv(EXPR_OUT, sep="\t")
        missing = sorted(set(genes) - set(cached.columns))
        if not missing:
            return cached
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:1086")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:1086")
    TCGA.mkdir(parents=True, exist_ok=True)
    samples = xena.dataset_samples(HUB, EXPR_DATASET, None)
    genes = sorted(set(genes))
    _positions, values = xena.dataset_probe_values(HUB, EXPR_DATASET, samples, genes)
    expr = pd.DataFrame(values, index=genes, columns=samples).T.reset_index()
    expr = expr.rename(columns={"index": "sample"})
    for gene in genes:
        expr[gene] = pd.to_numeric(expr[gene].replace("NaN", np.nan), errors="coerce")
    expr.to_csv(EXPR_OUT, sep="\t", index=False, compression="gzip")
    (TCGA / "NMF_STATE_SIGNATURE_EXPRESSION_MANIFEST.md").write_text(
        "\n".join(
            [
                "# TCGA NMF state signature expression manifest",
                "",
                f"- hub: {HUB}",
                f"- dataset: {EXPR_DATASET}",
                f"- genes_requested: {len(genes)}",
                f"- samples: {len(samples)}",
                f"- local_file: {EXPR_OUT}",
                f"- proxy_env_HTTP_PROXY: {os.environ.get('HTTP_PROXY', '')}",
                f"- proxy_env_HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', '')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return expr


def build_scores(expr: pd.DataFrame, sig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_path = OUT / "tcga_nmf_state_signature_scores.csv.gz"
    cov_path = OUT / "tcga_nmf_state_signature_gene_coverage.csv"
    if out_path.exists() and cov_path.exists():
        cached_scores = pd.read_csv(out_path)
        cached_cov = pd.read_csv(cov_path)
        expected = set(sig["state_key"].dropna().astype(str))
        if expected.issubset(set(cached_scores.columns)) and expected.issubset(set(cached_cov["state_key"])):
            return cached_scores, cached_cov
    scores = expr[["sample"]].copy()
    coverage_rows = []
    for key, sub in sig.groupby("state_key", observed=True):
        genes = [g for g in sub.sort_values("rank")["gene"] if g in expr.columns and expr[g].notna().sum() > 100]
        coverage_rows.append(
            {
                "state_key": key,
                "genes_requested": int(sub["gene"].nunique()),
                "genes_present": len(genes),
                "genes_used": ",".join(genes),
            }
        )
        if not genes:
            scores[key] = np.nan
            continue
        z = expr[genes].apply(zscore, axis=0)
        scores[key] = z.mean(axis=1)
    scores.to_csv(out_path, index=False, compression="gzip")
    cov = pd.DataFrame(coverage_rows)
    cov.to_csv(cov_path, index=False)
    return scores, cov


def prepare_survival(scores: pd.DataFrame) -> pd.DataFrame:
    out_path = OUT / "tcga_nmf_state_signature_scores_survival_merged.csv.gz"
    if out_path.exists():
        cached = pd.read_csv(out_path)
        expected = set(c for c in scores.columns if c not in {"sample", "sample15"})
        if expected.issubset(set(cached.columns)):
            return cached
    surv = pd.read_csv(TCGA / "Survival_SupplementalTable_S1_20171025_xena_sp", sep="\t")
    purity = pd.read_csv(TCGA / "TCGA_mastercalls.abs_tables_JSedit.fixed.txt", sep="\t")
    surv = surv.rename(columns={"cancer type abbreviation": "CancerType"})
    surv["sample15"] = surv["sample"].str[:15]
    scores = scores.copy()
    scores["sample15"] = scores["sample"].str[:15]
    purity["sample15"] = purity["array"].astype(str).str[:15]
    purity["absolute_purity"] = pd.to_numeric(purity["purity"], errors="coerce")
    df = scores.merge(surv, on="sample15", suffixes=("", "_surv")).merge(
        purity[["sample15", "absolute_purity"]], on="sample15", how="left"
    )
    df["duration"] = pd.to_numeric(df["OS.time"], errors="coerce")
    df["event"] = pd.to_numeric(df["OS"], errors="coerce")
    df["age"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce")
    df["age_z"] = zscore(df["age"])
    df["male"] = (df["gender"].astype(str).str.upper() == "MALE").astype(float)
    df["stage_ordinal"] = df["ajcc_pathologic_tumor_stage"].map(stage_to_ordinal)
    df["CancerType"] = df["CancerType"].astype(str)
    df["globocan_label"] = df["CancerType"].map(TCGA_TO_GLOBOCAN)
    df.to_csv(out_path, index=False, compression="gzip")
    return df


def fit_stratified_models(df: pd.DataFrame, state_cols: list[str]) -> pd.DataFrame:
    out_path = OUT / "tcga_nmf_state_signature_stratified_cox.csv"
    if out_path.exists():
        cached = pd.read_csv(out_path)
        if set(state_cols).issubset(set(cached["state_key"].dropna().astype(str))):
            return cached
    rows = []
    for state in state_cols:
        dat = df[["duration", "event", "age_z", "male", "stage_ordinal", "absolute_purity", "CancerType", state]].replace(
            [np.inf, -np.inf], np.nan
        )
        dat[state] = pd.to_numeric(dat[state], errors="coerce")
        dat = dat.dropna(subset=["duration", "event", state, "CancerType"])
        dat = dat[dat["duration"] > 0].copy()
        dat["score_z"] = dat.groupby("CancerType", observed=True)[state].transform(zscore)
        dat = dat.dropna(subset=["score_z"])
        model_covars = {
            "age_sex": ["score_z", "age_z", "male"],
            "age_sex_stage_available": ["score_z", "age_z", "male", "stage_ordinal"],
            "age_sex_purity_available": ["score_z", "age_z", "male", "absolute_purity"],
            "age_sex_stage_purity_available": ["score_z", "age_z", "male", "stage_ordinal", "absolute_purity"],
        }
        for model, covars in model_covars.items():
            sub = dat[["duration", "event", "CancerType"] + covars].dropna().copy()
            if len(sub) < 500 or sub["event"].sum() < 100 or sub["CancerType"].nunique() < 5:
                continue
            try:
                cph = CoxPHFitter(penalizer=0.02)
                cph.fit(sub, duration_col="duration", event_col="event", strata=["CancerType"])
                s = cph.summary.loc["score_z"]
                rows.append(
                    {
                        "model": f"stratified_by_tcga_cancer__{model}",
                        "state_key": state,
                        "n_samples": len(sub),
                        "n_events": int(sub["event"].sum()),
                        "n_cancer_types": int(sub["CancerType"].nunique()),
                        "coef_log_hr_per_within_cancer_sd": float(s["coef"]),
                        "hr_per_within_cancer_sd": float(s["exp(coef)"]),
                        "ci95_low": float(s["exp(coef) lower 95%"]),
                        "ci95_high": float(s["exp(coef) upper 95%"]),
                        "p": float(s["p"]),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append({"model": model, "state_key": state, "error": repr(exc)})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sig = extract_signatures()
    expr = fetch_expression(sig["gene"].dropna().astype(str).unique().tolist())
    scores, cov = build_scores(expr, sig)
    df = prepare_survival(scores)
    state_cols = [c for c in scores.columns if c not in {"sample", "sample15"}]
    cox = fit_stratified_models(df, state_cols)
    print(f"wrote {OUT / 'zenodo_nmf_state_signature_genes.csv'}")
    print(f"wrote {OUT / 'tcga_nmf_state_signature_gene_coverage.csv'}")
    print(f"wrote {OUT / 'tcga_nmf_state_signature_stratified_cox.csv'}")


if __name__ == "__main__":
    main()
