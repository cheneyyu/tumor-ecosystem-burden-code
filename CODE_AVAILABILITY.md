# Code availability

Analysis code for the public-data pan-cancer tumor ecosystem burden study is
provided in this repository. Manuscript drafts, generated manuscript PDFs, raw
data caches, generated figures, large processed outputs, and submission files
are intentionally excluded.

The main reproducible data-fetch entry points are:

- `make fetch_public_data`
- `make fetch_cellxgene_tme`

The main reproducible analysis entry points are:

- `make translational_validation`
- `make submission_supplementary_tables`

The scripts expect the public raw data and processed intermediate tables to be
available under `data/raw/` and `data/processed/`. See `README.md` for the
expected directory structure and interpretation boundaries.
