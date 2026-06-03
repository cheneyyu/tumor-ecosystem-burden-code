#!/usr/bin/env python3
"""Fetch public data for modeled cell-type attributable cancer burden.

The script intentionally separates directly usable downloads from sources that
need manual export or source-specific interpretation.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "metadata"

PROXY = "http://127.0.0.1:1086"
PROXIES = {"http": PROXY, "https": PROXY}
HEADERS = {"User-Agent": "proj_eco_public_burden_fetch/0.1"}


def mkdirs() -> None:
    for sub in [
        RAW / "gbd_globocan",
        RAW / "tisch_cancersea",
        RAW / "tcga_xena",
        RAW / "costs",
        META,
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def get(url: str, *, timeout: int = 90) -> requests.Response:
    response = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=timeout)
    response.raise_for_status()
    return response


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def download_file(url: str, path: Path, *, timeout: int = 180) -> tuple[bool, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(
            url, headers=HEADERS, proxies=PROXIES, timeout=timeout, stream=True
        ) as response:
            response.raise_for_status()
            with path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        return True, f"downloaded {path.stat().st_size} bytes"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def clean_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "nan", "None"}:
        return None
    text = text.replace("$", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def fetch_globocan() -> None:
    out_dir = RAW / "gbd_globocan"
    manifest: list[str] = ["# GBD / GLOBOCAN manifest", ""]
    base = "https://gco.iarc.fr/gateway_prod/api/globocan/v3/2022/"

    cancer_url = base + "meta/cancers/all/"
    pop_url = base + "meta/populations/all/"
    cancers = get(cancer_url).json()
    populations = get(pop_url).json()
    (out_dir / "globocan_cancers_2022.json").write_text(json.dumps(cancers, indent=2))
    (out_dir / "globocan_populations_2022.json").write_text(
        json.dumps(populations, indent=2)
    )
    manifest += [
        f"- OK `globocan_cancers_2022.json` from {cancer_url}",
        f"- OK `globocan_populations_2022.json` from {pop_url}",
    ]

    # GLOBOCAN uses its own `country`/`country_code` field. China is 160
    # here, while ISO numeric 156 returns an empty dataset from this endpoint.
    locations = {"Global": 900, "China": 160, "United States": 840}
    measures = {0: "incidence", 1: "mortality", 2: "prevalence"}
    rows: list[dict[str, Any]] = []

    cancer_lookup = {int(c["cancer"]): c for c in cancers}
    pop_lookup = {int(p["country"]): p for p in populations}
    for location_label, location_code in locations.items():
        for measure_code, measure_label in measures.items():
            url = (
                f"{base}data/rate/{measure_code}/0/{location_code}/all/"
                "?ages_group=0_17&group_CRC=1&include_nmsc=0&include_nmsc_other=1"
            )
            payload = get(url).json()
            fn = f"globocan_2022_{location_label.lower().replace(' ', '_')}_{measure_label}.json"
            (out_dir / fn).write_text(json.dumps(payload, indent=2))
            manifest.append(f"- OK `{fn}` from {url}")
            for item in payload.get("dataset", []):
                cancer_code = int(item["cancer_code"])
                cancer = cancer_lookup.get(cancer_code, {})
                pop = pop_lookup.get(int(item["country_code"]), {})
                rows.append(
                    {
                        "source": "GLOBOCAN 2022 v1.1 API",
                        "location": location_label,
                        "location_code": item.get("country_code"),
                        "location_iso3": pop.get("country_iso3"),
                        "sex": item.get("sex"),
                        "measure": measure_label,
                        "cancer_code": cancer_code,
                        "cancer_label": cancer.get("label"),
                        "cancer_short_label": cancer.get("short_label"),
                        "icd": cancer.get("ICD"),
                        "total": item.get("total"),
                        "asr_world": item.get("asr"),
                        "crude_rate": item.get("crude_rate"),
                        "cum_risk_74": item.get("cum_risk_74"),
                        "rank": item.get("rank"),
                        "ui_low": item.get("ui", {}).get("low")
                        if isinstance(item.get("ui"), dict)
                        else None,
                        "ui_high": item.get("ui", {}).get("high")
                        if isinstance(item.get("ui"), dict)
                        else None,
                    }
                )
            time.sleep(0.1)

    pd.DataFrame(rows).to_csv(out_dir / "globocan_2022_burden_long.csv", index=False)
    manifest += [
        "- OK `globocan_2022_burden_long.csv`: normalized incidence, mortality, prevalence for Global, China, United States.",
        "- NOTE GBD/IHME DALYs were not directly downloaded here because the browser tool uses an interactive export workflow; GLOBOCAN mortality is used for the first reproducible burden-weighted pass.",
    ]
    write_text(out_dir / "MANIFEST.md", "\n".join(manifest) + "\n")


def parse_tisch_rows(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict[str, Any]] = []
    for tr in soup.select("table#dataset-table tbody tr"):
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(tds) < 10:
            continue
        dataset = tds[2]
        cancer_prefix = dataset.split("_", 1)[0]
        rows.append(
            {
                "tisch_id": tds[1],
                "dataset": dataset,
                "tisch_cancer": cancer_prefix,
                "species": tds[3],
                "treatment": tds[4],
                "sample_count": clean_number(tds[5]),
                "cell_count": clean_number(tds[6]),
                "platform": tds[7],
                "primary_metastatic": tds[8],
                "pmid": tds[9],
            }
        )
    return pd.DataFrame(rows)


def fetch_tisch_and_cancersea() -> None:
    out_dir = RAW / "tisch_cancersea"
    manifest: list[str] = ["# TISCH2 / CancerSEA manifest", ""]

    gallery_url = "https://tisch.compbio.cn/gallery/"
    html = get(gallery_url).text
    write_text(out_dir / "tisch2_gallery.html", html)
    tisch = parse_tisch_rows(html)
    tisch.to_csv(out_dir / "tisch2_datasets.csv", index=False)
    manifest += [
        f"- OK `tisch2_gallery.html` from {gallery_url}",
        f"- OK `tisch2_datasets.csv`: {len(tisch)} dataset rows parsed.",
    ]

    soup = BeautifulSoup(html, "lxml")
    celltype_options = [
        (opt.get("value"), opt.get_text(" ", strip=True))
        for opt in soup.select("select#celltype option")
        if opt.get("value")
    ]
    pd.DataFrame(celltype_options, columns=["celltype", "label"]).to_csv(
        out_dir / "tisch2_celltype_options.csv", index=False
    )

    presence_rows: list[dict[str, Any]] = []
    for celltype, celltype_label in celltype_options:
        url = gallery_url + f"?celltype={requests.utils.quote(celltype)}&species=Human&treatment=None"
        try:
            page = get(url).text
            subset = parse_tisch_rows(page)
            write_text(out_dir / f"tisch2_filter_{re.sub(r'[^A-Za-z0-9]+', '_', celltype)}.html", page)
            for _, row in subset.iterrows():
                rec = row.to_dict()
                rec["celltype"] = celltype
                rec["celltype_label"] = celltype_label
                rec["filter_url"] = url
                presence_rows.append(rec)
            manifest.append(
                f"- OK TISCH2 celltype filter `{celltype}`: {len(subset)} human no-treatment datasets."
            )
        except Exception as exc:  # noqa: BLE001
            manifest.append(f"- FAIL TISCH2 celltype filter `{celltype}`: {exc!r}")
        time.sleep(0.1)

    presence = pd.DataFrame(presence_rows)
    presence.to_csv(out_dir / "tisch2_celltype_dataset_presence.csv", index=False)
    manifest.append(
        "- OK `tisch2_celltype_dataset_presence.csv`: dataset-level cell-type presence proxy, not cell-level abundance."
    )
    manifest.append(
        "- NOTE TISCH2 `*_Data.zip` links are commented in the page and tested as HTTP 404; this fetch does not claim cell-level abundance from TISCH2."
    )

    sea_url = "http://biocc.hrbmu.edu.cn/CancerSEA/goDownload"
    sea_html = get(sea_url).text
    write_text(out_dir / "cancersea_download.html", sea_html)
    sea_soup = BeautifulSoup(sea_html, "lxml")
    link_rows: list[dict[str, Any]] = []
    for a in sea_soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("download/"):
            link_rows.append(
                {
                    "label": a.get_text(" ", strip=True),
                    "href": href,
                    "url": urljoin(sea_url, href),
                    "kind": "state_score"
                    if "/StateScore/" in href
                    else ("expression" if "/Expression/" in href else "other"),
                }
            )
    links = pd.DataFrame(link_rows)
    links.to_csv(out_dir / "cancersea_download_links.csv", index=False)
    manifest.append(
        f"- OK `cancersea_download_links.csv`: {len(links)} downloadable CancerSEA files found."
    )

    score_dir = out_dir / "cancersea_state_scores"
    score_dir.mkdir(exist_ok=True)
    for _, row in links[links["kind"] == "state_score"].iterrows():
        filename = Path(row["href"]).name
        ok, msg = download_file(row["url"], score_dir / filename, timeout=120)
        manifest.append(f"- {'OK' if ok else 'FAIL'} CancerSEA `{filename}`: {msg}")
        time.sleep(0.05)

    write_text(out_dir / "MANIFEST.md", "\n".join(manifest) + "\n")


def fetch_tcga_panimmune() -> None:
    out_dir = RAW / "tcga_xena"
    manifest: list[str] = ["# TCGA / Xena / GDC PanImmune manifest", ""]
    files = [
        (
            "Survival_SupplementalTable_S1_20171025_xena_sp",
            "https://pancanatlas.xenahubs.net/download/Survival_SupplementalTable_S1_20171025_xena_sp",
        ),
        (
            "TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
            "https://pancanatlas.xenahubs.net/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
        ),
        (
            "TCGA.Kallisto.fullIDs.cibersort.relative.tsv",
            "https://api.gdc.cancer.gov/data/b3df502e-3594-46ef-9f94-d041a20a0b9a",
        ),
        (
            "PanImmune_GeneSet_Definitions.xlsx",
            "https://api.gdc.cancer.gov/data/9b174979-fe97-48bc-9e97-9384b0519f03",
        ),
    ]
    for filename, url in files:
        path = out_dir / filename
        ok, msg = download_file(url, path, timeout=240)
        manifest.append(f"- {'OK' if ok else 'FAIL'} `{filename}` from {url}: {msg}")
    manifest.append(
        "- NOTE RNA expression matrix is intentionally not downloaded by default because it is large; PanImmune CIBERSORT fractions are enough for the first immune-cell prognosis model."
    )
    write_text(out_dir / "MANIFEST.md", "\n".join(manifest) + "\n")


def fetch_costs() -> None:
    out_dir = RAW / "costs"
    manifest: list[str] = ["# Cancer cost manifest", ""]
    nci_url = "https://progressreport.cancer.gov/after/economic_burden"
    html = get(nci_url).text
    write_text(out_dir / "nci_progress_economic_burden.html", html)
    manifest.append(f"- OK `nci_progress_economic_burden.html` from {nci_url}")

    tables = pd.read_html(html)
    all_rows: list[pd.DataFrame] = []
    for idx, table in enumerate(tables, start=1):
        if "Cancer Site" not in table.columns:
            continue
        table = table.copy()
        table["table_index"] = idx
        table["source_url"] = nci_url
        all_rows.append(table)
    if all_rows:
        cost_tables = pd.concat(all_rows, ignore_index=True)
        cost_tables.to_csv(out_dir / "nci_cost_tables_raw.csv", index=False)
        long_rows: list[dict[str, Any]] = []
        for _, row in cost_tables.iterrows():
            table_index = int(row["table_index"])
            site = row["Cancer Site"]
            for col, val in row.items():
                if col in {"Cancer Site", "table_index", "source_url"}:
                    continue
                long_rows.append(
                    {
                        "source": "NCI Cancer Trends Progress Report",
                        "source_url": nci_url,
                        "table_index": table_index,
                        "cancer_site": site,
                        "metric": str(col),
                        "value": clean_number(val),
                    }
                )
        pd.DataFrame(long_rows).to_csv(out_dir / "nci_cost_tables_long.csv", index=False)
        manifest.append(
            f"- OK `nci_cost_tables_raw.csv` and `nci_cost_tables_long.csv`: {len(all_rows)} HTML tables parsed."
        )
    else:
        manifest.append("- FAIL no NCI cost tables with `Cancer Site` column were parsed.")

    manifest.append(
        "- NOTE JAMA Oncology global economic-cost article was blocked by Cloudflare from direct curl in this environment; NCI US costs are used for reproducible exploratory cost weighting."
    )
    write_text(out_dir / "MANIFEST.md", "\n".join(manifest) + "\n")


def main() -> None:
    mkdirs()
    fetch_globocan()
    fetch_tisch_and_cancersea()
    fetch_tcga_panimmune()
    fetch_costs()


if __name__ == "__main__":
    main()
