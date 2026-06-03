.PHONY: fetch_public_data fetch_cellxgene_tme translational_validation submission_supplementary_tables

fetch_public_data:
	.venv/bin/python scripts/fetch_public_data.py

fetch_cellxgene_tme:
	.venv/bin/python scripts/fetch_cellxgene_tme.py

translational_validation:
	.venv/bin/python scripts/translational_validation/run_all.py

submission_supplementary_tables:
	.venv/bin/python scripts/prepare_submission_supplementary_tables.py
