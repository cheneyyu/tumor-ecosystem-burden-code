# Pan-cancer tumor ecosystem burden code

This repository contains analysis code for a public-data pan-cancer tumor
ecosystem burden and validation study. The code integrates single-cell tumor
ecosystem state representations, public cancer burden and modeled cost weights,
TCGA survival and stage analyses, CPTAC UCEC validation, and exploratory public
immunotherapy cohort analyses.

## What is included

- `scripts/`: reproducible Python data-fetch, analysis and figure/table generation scripts.
- `scripts/fetch_public_data.py`: fetches directly downloadable public burden,
  cost, TISCH/CancerSEA metadata and TCGA/PanImmune inputs used by the first
  reproducible pass.
- `scripts/fetch_cellxgene_tme.py`: fetches and audits the selected CELLxGENE
  multi-tissue tumor microenvironment H5AD.
- `scripts/translational_validation/run_all.py`: extension pipeline for
  signature harmonization, TCGA nested Cox models, CPTAC UCEC validation,
  immunotherapy response/survival analyses, robustness outputs, and manuscript
  insert drafts.
- `metadata/cancer_site_crosswalk.tsv`: local cancer-label harmonization table.
- `Makefile`: workflow targets for data fetch, translational validation and
  supplementary table generation.
- `requirements.txt`: Python package requirements used by the analysis scripts.

## What is not included

This code repository intentionally does not include manuscript drafts, compiled
manuscript PDFs, raw data caches, generated figures, large processed data, or
submission files. Those files are excluded by `.gitignore` and should be stored
or deposited separately as appropriate.

## Expected directory layout

The scripts assume the following local layout after public data have been
downloaded or restored from an external data deposit:

```text
data/
  raw/
  processed/
    pancancer_ecosystem/
    translational_validation/
figures/
reports/
submission/
metadata/
scripts/
```

Large raw and processed data are not tracked in this repository. The generated
file manifest from the analysis records file paths, upstream dependencies, and
checksums for reproducibility.

## Environment

The original analysis used Python 3.10 and `uv`.

```bash
uv venv .venv --python 3.10
uv pip install -r requirements.txt
```

System-level tools used by optional figure/manuscript workflows include
`pdflatex`, `pdftoppm`, and `zip`.

## Data download commands

Fetch the smaller public inputs:

```bash
make fetch_public_data
```

Fetch the selected CELLxGENE H5AD and run the metadata audit:

```bash
make fetch_cellxgene_tme
```

If a local proxy is needed, run the script directly:

```bash
.venv/bin/python scripts/fetch_cellxgene_tme.py --proxy http://127.0.0.1:1086
```

For CELLxGENE metadata only, add `--metadata-only`; for an already downloaded
H5AD audit, add `--audit-only`.

## Main analysis commands

Run the translational validation extension:

```bash
make translational_validation
```

Build the submission-ready supplementary tables workbook and per-table TSVs:

```bash
make submission_supplementary_tables
```

## Interpretation boundary

The cost and burden outputs are modeled, ecological public-data estimates. They
should not be interpreted as observed patient-level cell-state-specific medical
spending or causal treatment effects.
