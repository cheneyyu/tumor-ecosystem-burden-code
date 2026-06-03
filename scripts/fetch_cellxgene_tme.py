#!/usr/bin/env python3
"""Fetch the selected CELLxGENE multi-tissue TME H5AD and audit metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests


DATASET_VERSION_ID = "921d46a3-69b4-44a8-b2d6-9ef5c7803bc3"
COLLECTION_ID = "3f7c572c-cd73-4b51-a313-207c7f20f188"
COLLECTION_API = f"https://api.cellxgene.cziscience.com/curation/v1/collections/{COLLECTION_ID}"
DEFAULT_PROXY = "http://127.0.0.1:1086"
OUT_DIR = Path("data/raw/cellxgene_tme")
PROCESSED_DIR = Path("data/processed")
H5AD_NAME = f"{DATASET_VERSION_ID}.h5ad"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def proxies(proxy: str | None) -> dict[str, str] | None:
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def fetch_json(url: str, proxy: str | None) -> dict:
    resp = requests.get(url, timeout=60, proxies=proxies(proxy))
    resp.raise_for_status()
    return resp.json()


def find_dataset(collection: dict) -> dict:
    for dataset in collection.get("datasets", []):
        if dataset.get("dataset_version_id") == DATASET_VERSION_ID:
            return dataset
    raise RuntimeError(f"Dataset version {DATASET_VERSION_ID} not found in collection metadata")


def h5ad_asset(dataset: dict) -> dict:
    assets = dataset.get("assets") or dataset.get("dataset_assets") or []
    for asset in assets:
        if asset.get("filetype") == "H5AD" and asset.get("url"):
            return asset
    raise RuntimeError("No downloadable H5AD asset found in dataset metadata")


def download_with_resume(url: str, dest: Path, expected_size: int | None, proxy: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"already complete: {dest} ({expected_size} bytes)")
        return

    headers = {}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"resuming at byte {existing}")

    with requests.get(url, stream=True, timeout=(30, 180), headers=headers, proxies=proxies(proxy)) as resp:
        if existing and resp.status_code == 200:
            print("server ignored Range header; restarting partial download")
            existing = 0
            mode = "wb"
        resp.raise_for_status()
        total = expected_size or int(resp.headers.get("content-length", 0)) + existing
        downloaded = existing
        last_report = downloaded
        with partial.open(mode + "") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 128 * 1024 * 1024:
                    pct = (downloaded / total * 100) if total else 0
                    print(f"downloaded {downloaded / 1e9:.2f} GB / {total / 1e9:.2f} GB ({pct:.1f}%)", flush=True)
                    last_report = downloaded

    final_size = partial.stat().st_size
    if expected_size and final_size != expected_size:
        raise RuntimeError(f"Downloaded size mismatch: got {final_size}, expected {expected_size}")
    partial.replace(dest)
    print(f"download complete: {dest} ({final_size} bytes)")


def write_manifest(collection: dict, dataset: dict, asset: dict, out_file: Path, proxy: str | None, status: str) -> None:
    disease_labels = ", ".join(item["label"] for item in dataset.get("disease", []))
    cell_type_labels = ", ".join(item["label"] for item in dataset.get("cell_type", []))
    tissue_labels = ", ".join(item["label"] for item in dataset.get("tissue", []))
    lines = [
        "# CELLxGENE multi-tissue TME dataset",
        "",
        f"- fetched_at_utc: {now_utc()}",
        f"- status: {status}",
        f"- collection_id: {COLLECTION_ID}",
        f"- collection_url: {collection.get('collection_url', '')}",
        f"- collection_api: {COLLECTION_API}",
        f"- dataset_version_id: {DATASET_VERSION_ID}",
        f"- dataset_id: {dataset.get('dataset_id', '')}",
        f"- citation: {dataset.get('citation', '')}",
        f"- h5ad_url: {asset.get('url', '')}",
        f"- h5ad_filesize: {asset.get('filesize', '')}",
        f"- proxy: {proxy or ''}",
        f"- local_h5ad: {out_file}",
        f"- local_h5ad_size: {out_file.stat().st_size if out_file.exists() else ''}",
        f"- cell_count: {dataset.get('cell_count', '')}",
        f"- primary_cell_count: {dataset.get('primary_cell_count', '')}",
        f"- disease: {disease_labels}",
        f"- tissue: {tissue_labels}",
        f"- cell_type: {cell_type_labels}",
        "",
        "## Use in this project",
        "",
        "This h5ad is the planned single-cell evidence layer for true TAM-state analysis,",
        "complementing the existing TCGA CIBERSORT macrophage M2 burden signal.",
        "",
    ]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    (out_file.parent / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def audit_h5ad(h5ad_path: Path, out_dir: Path) -> None:
    import anndata as ad

    out_dir.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs
    var = a.var

    summary = {
        "h5ad_path": str(h5ad_path),
        "n_obs": int(a.n_obs),
        "n_vars": int(a.n_vars),
        "obs_columns": list(obs.columns),
        "var_columns": list(var.columns),
        "has_raw": a.raw is not None,
        "layers": list(a.layers.keys()),
        "obsm": list(a.obsm.keys()),
        "uns_keys": list(a.uns.keys()),
    }
    (out_dir / "cellxgene_tme_metadata_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    column_rows = []
    for col in obs.columns:
        ser = obs[col]
        row = {
            "column": col,
            "dtype": str(ser.dtype),
            "n_missing": int(ser.isna().sum()),
            "n_unique": int(ser.nunique(dropna=True)),
        }
        values = ser.dropna().astype(str).value_counts().head(20)
        row["top_values"] = "; ".join(f"{idx}={val}" for idx, val in values.items())
        column_rows.append(row)
    pd.DataFrame(column_rows).to_csv(out_dir / "cellxgene_tme_obs_column_audit.csv", index=False)

    key_cols = [
        col
        for col in [
            "disease",
            "tissue",
            "cell_type",
            "assay",
            "suspension_type",
            "donor_id",
            "sample_id",
            "dataset_id",
            "self_reported_ethnicity",
            "sex",
        ]
        if col in obs.columns
    ]
    for col in key_cols:
        obs[col].astype(str).value_counts().rename_axis(col).reset_index(name="n_cells").to_csv(
            out_dir / f"cellxgene_tme_{col}_counts.csv",
            index=False,
        )

    if {"disease", "cell_type"}.issubset(obs.columns):
        pd.crosstab(obs["disease"].astype(str), obs["cell_type"].astype(str)).to_csv(
            out_dir / "cellxgene_tme_disease_by_cell_type_counts.csv"
        )
    if {"disease", "cell_type", "donor_id"}.issubset(obs.columns):
        (
            obs.groupby(["disease", "cell_type"], observed=False)["donor_id"]
            .nunique()
            .rename("n_donors")
            .reset_index()
            .to_csv(out_dir / "cellxgene_tme_disease_by_cell_type_donor_counts.csv", index=False)
        )
    audit_myeloid(obs, var, out_dir)
    a.file.close()


def audit_myeloid(obs: pd.DataFrame, var: pd.DataFrame, out_dir: Path) -> None:
    myeloid = obs[obs["cell_type"].astype(str).eq("mononuclear phagocyte")].copy()
    if myeloid.empty:
        return

    count_cols = [
        "author_cell_type",
        "disease",
        "harm_tumor.type",
        "harm_tumor.site",
        "harm_sample.type",
        "harm_condition",
        "donor_id",
    ]
    for col in count_cols:
        if col in myeloid.columns:
            myeloid[col].astype(str).value_counts().rename_axis(col).reset_index(name="n_cells").to_csv(
                out_dir / f"cellxgene_tme_myeloid_{col.replace('.', '_')}_counts.csv",
                index=False,
            )

    if {"author_cell_type", "harm_tumor.type", "donor_id"}.issubset(myeloid.columns):
        (
            myeloid.groupby(["harm_tumor.type", "author_cell_type"], observed=True)
            .agg(n_cells=("author_cell_type", "size"), n_donors=("donor_id", "nunique"))
            .reset_index()
            .sort_values(["harm_tumor.type", "n_cells"], ascending=[True, False])
            .to_csv(out_dir / "cellxgene_tme_myeloid_state_by_tumor_type.csv", index=False)
        )

    if {"author_cell_type", "harm_sample.type", "donor_id"}.issubset(myeloid.columns):
        (
            myeloid.groupby(["harm_sample.type", "author_cell_type"], observed=True)
            .agg(n_cells=("author_cell_type", "size"), n_donors=("donor_id", "nunique"))
            .reset_index()
            .sort_values(["harm_sample.type", "n_cells"], ascending=[True, False])
            .to_csv(out_dir / "cellxgene_tme_myeloid_state_by_sample_type.csv", index=False)
        )

    state_markers = {
        "SPP1_TAM": ["SPP1", "APOE", "TREM2", "GPNMB", "LGALS3"],
        "C1QC_resident_like": ["C1QA", "C1QB", "C1QC", "FOLR2", "SELENOP"],
        "FCN1_inflammatory_mono": ["FCN1", "S100A8", "S100A9", "IL1B", "VCAN"],
        "IFN_myeloid": ["CXCL10", "ISG15", "IFIT1", "IFIT3", "STAT1"],
        "angiogenic_TAM": ["VEGFA", "ANGPT2", "EREG", "MMP9", "SPP1"],
        "cycling_TAM": ["MKI67", "TOP2A", "STMN1", "UBE2C", "HMGB2"],
    }
    feature_names = set(var["feature_name"].astype(str)) if "feature_name" in var.columns else set(map(str, var.index))
    marker_rows = []
    for state, genes in state_markers.items():
        present = [gene for gene in genes if gene in feature_names]
        marker_rows.append(
            {
                "tam_state": state,
                "marker_genes": ",".join(genes),
                "present_marker_genes": ",".join(present),
                "n_present": len(present),
                "n_markers": len(genes),
            }
        )
    pd.DataFrame(marker_rows).to_csv(out_dir / "cellxgene_tme_tam_state_marker_availability.csv", index=False)

    top_states = (
        myeloid["author_cell_type"].astype(str).value_counts().head(15)
        if "author_cell_type" in myeloid.columns
        else pd.Series(dtype=int)
    )
    tumor_type_counts = (
        myeloid["harm_tumor.type"].astype(str).value_counts()
        if "harm_tumor.type" in myeloid.columns
        else pd.Series(dtype=int)
    )
    donor_n = myeloid["donor_id"].nunique() if "donor_id" in myeloid.columns else ""
    def series_markdown(series: pd.Series, index_name: str, value_name: str) -> str:
        rows = [f"| {index_name} | {value_name} |", "|---|---:|"]
        rows.extend(f"| {idx} | {int(val)} |" for idx, val in series.items())
        return "\n".join(rows)

    def dataframe_markdown(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join(rows)

    md_lines = [
        "# CELLxGENE TME myeloid/TAM audit",
        "",
        f"- h5ad_cells: {len(obs)}",
        f"- h5ad_genes: {len(var)}",
        f"- mononuclear_phagocyte_cells: {len(myeloid)}",
        f"- mononuclear_phagocyte_donors: {donor_n}",
        "",
        "## Main author myeloid labels",
        "",
        series_markdown(top_states, "author_cell_type", "n_cells"),
        "",
        "## Myeloid tumor-type coverage",
        "",
        series_markdown(tumor_type_counts, "harm_tumor.type", "n_cells"),
        "",
        "## Marker availability",
        "",
        dataframe_markdown(pd.DataFrame(marker_rows)),
        "",
        "## Interpretation",
        "",
        "This dataset is suitable for the next true TAM-state layer: the curated",
        "`author_cell_type` labels already separate mononuclear phagocytes into",
        "macrophage, resident-like, angiogenic, hypoxic, IFN, monocyte-like and DC",
        "subsets across multiple tumor types and donors. The first robust analysis",
        "should use author labels for abundance, then use marker scores as a",
        "secondary validation rather than reclustering the full object immediately.",
        "",
    ]
    (out_dir / "cellxgene_tme_myeloid_audit.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or DEFAULT_PROXY)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    h5ad_path = OUT_DIR / H5AD_NAME

    if args.audit_only:
        if not h5ad_path.exists():
            raise SystemExit(f"missing h5ad for audit: {h5ad_path}")
        audit_h5ad(h5ad_path, PROCESSED_DIR)
        return

    collection = fetch_json(COLLECTION_API, args.proxy)
    dataset = find_dataset(collection)
    asset = h5ad_asset(dataset)
    (OUT_DIR / "collection_metadata.json").write_text(
        json.dumps(collection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "selected_dataset_metadata.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_manifest(collection, dataset, asset, h5ad_path, args.proxy, "metadata fetched")
    print(f"selected H5AD: {asset['url']}")
    print(f"expected bytes: {asset.get('filesize')}")

    if args.metadata_only:
        return

    download_with_resume(asset["url"], h5ad_path, asset.get("filesize"), args.proxy)
    write_manifest(collection, dataset, asset, h5ad_path, args.proxy, "h5ad downloaded")
    if not args.skip_audit:
        audit_h5ad(h5ad_path, PROCESSED_DIR)
        write_manifest(collection, dataset, asset, h5ad_path, args.proxy, "h5ad downloaded and audited")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
