#!/usr/bin/env python3
"""Run pan-TCGA stratified Cox models for all available CIBERSORT features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "tcga_patient_level_cibersort_survival.csv"
OUT = ROOT / "data" / "processed" / "pancancer_ecosystem" / "tcga_cibersort_all_stratified_cox.csv"

META = {
    "SampleID",
    "sample15",
    "CancerType",
    "globocan_label",
    "duration",
    "event",
    "age",
    "age_z",
    "male",
}


def zscore_within_cancer(df: pd.DataFrame, col: str) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce")
    out = vals.copy()
    for _, idx in df.groupby("CancerType", observed=True).groups.items():
        sub = vals.loc[idx]
        sd = sub.std(ddof=0)
        if np.isfinite(sd) and sd > 0:
            out.loc[idx] = (sub - sub.mean()) / sd
        else:
            out.loc[idx] = np.nan
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(IN)
    feature_cols = [c for c in d.columns if c not in META]
    rows = []
    for col in feature_cols:
        fit = d[["duration", "event", "CancerType", "age_z", "male", col]].copy()
        fit["score_z"] = zscore_within_cancer(fit.rename(columns={col: "score"}), "score")
        fit = fit.drop(columns=[col]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(fit) < 200 or fit["event"].sum() < 50 or fit["score_z"].nunique() < 5:
            rows.append({"cell_type": col, "error": "insufficient_data"})
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(
                fit,
                duration_col="duration",
                event_col="event",
                strata=["CancerType"],
                formula="score_z + age_z + male",
            )
            s = cph.summary.loc["score_z"]
            rows.append(
                {
                    "cell_type": col,
                    "n_samples": int(len(fit)),
                    "n_events": int(fit["event"].sum()),
                    "n_cancer_types": int(fit["CancerType"].nunique()),
                    "coef_log_hr_per_within_cancer_sd": float(s["coef"]),
                    "hr_per_within_cancer_sd": float(np.exp(s["coef"])),
                    "ci95_low": float(np.exp(s["coef lower 95%"])),
                    "ci95_high": float(np.exp(s["coef upper 95%"])),
                    "p": float(s["p"]),
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - records model failures for audit
            rows.append({"cell_type": col, "error": str(exc)})
    out = pd.DataFrame(rows).sort_values(["error", "coef_log_hr_per_within_cancer_sd"], ascending=[True, False])
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
