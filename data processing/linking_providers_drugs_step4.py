#!/usr/bin/env python3
"""
Step 4 — Merge high-risk drug exposures with providers (source specialty only).

Assumptions:
- This script's folder is a SIBLING of the 'data/' folder:
    repo_root/
      data/
      scripts_or_your_folder/
        step4_merge_providers.py  <-- this file

Inputs (in data/):
  - high_risk_drug_exposure_filtered.csv
  - providers.csv

Output (in data/):
  - high_risk_drug_exposure_with_providers.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# ---------- Config: paths relative to this file ----------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

EXPOSURES_PATH = DATA_DIR / "drug_exposure_high_risk.csv"
PROVIDERS_PATH = DATA_DIR / "providers.csv"
OUT_PATH = DATA_DIR / "high_risk_drug_exposure_with_providers.csv"


# ---------- Helpers ----------
def _coerce_int(s: pd.Series) -> pd.Series:
    # Safe integer coercion; keeps NaN if non-numeric
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _normalize_specialty_source_value(s: pd.Series) -> pd.Series:
    # Strip whitespace; treat empty strings as NaN; KEEP literal "unknown"
    s = s.astype("string").str.strip()
    s = s.mask(s == "", np.nan)
    return s

def load_inputs(exposures_path: Path, providers_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # --- Drug exposures ---
    de = pd.read_csv(exposures_path, low_memory=False)
    needed_de = {"person_id", "drug_concept_id", "provider_id",
                 "drug_exposure_start_DATE", "drug_exposure_end_DATE"}
    missing_de = needed_de - set(de.columns)
    if missing_de:
        raise ValueError(f"Missing columns in exposures: {missing_de}")

    de["person_id"] = _coerce_int(de["person_id"])
    de["drug_concept_id"] = _coerce_int(de["drug_concept_id"])
    de["provider_id"] = _coerce_int(de["provider_id"])
    de = _parse_dates(de, ["drug_exposure_start_DATE", "drug_exposure_end_DATE"])

    # --- Providers ---
    pr = pd.read_csv(providers_path, low_memory=False)
    needed_pr = {"provider_id", "specialty_source_value"}
    missing_pr = needed_pr - set(pr.columns)
    if missing_pr:
        raise ValueError(f"Missing columns in providers: {missing_pr}")

    pr["provider_id"] = _coerce_int(pr["provider_id"])
    pr["specialty_source_value"] = _normalize_specialty_source_value(pr["specialty_source_value"])

    return de, pr

def merge_and_clean(de: pd.DataFrame, pr: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    # Inner join to keep only exposures tied to known providers
    m = de.merge(pr[["provider_id", "specialty_source_value"]],
                 on="provider_id", how="inner", validate="m:1")

    # Drop rows with blank/NA specialty_source_value; keep "unknown"
    before = len(m)
    m = m[~m["specialty_source_value"].isna()].copy()
    after = len(m)

    # Acceptance checks
    if m["provider_id"].isna().any():
        raise AssertionError("Found NA provider_id after merge.")
    if m.empty:
        raise AssertionError("Merged dataset is empty after cleaning.")

    return m, before, after

def compute_metrics(m: pd.DataFrame) -> dict:
    metrics = {}
    metrics["rows"] = len(m)
    metrics["unique_providers"] = int(m["provider_id"].nunique())
    metrics["unique_drugs"] = int(m["drug_concept_id"].nunique())
    metrics["share_unknown"] = float((m["specialty_source_value"].str.lower() == "unknown").mean())

    # exposures per provider / per drug
    per_provider = m.groupby("provider_id", observed=True).size()
    per_drug = m.groupby("drug_concept_id", observed=True).size()

    def _summary(s: pd.Series) -> dict:
        if len(s) == 0:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0, "max": 0}
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p90": float(s.quantile(0.90)),
            "min": int(s.min()),
            "max": int(s.max()),
        }

    metrics["exposures_per_provider"] = _summary(per_provider)
    metrics["exposures_per_drug"] = _summary(per_drug)

    # Top 10 specialties by exposure count
    top_specs = (
        m.assign(_one=1)
         .groupby("specialty_source_value", observed=True)["_one"]
         .sum()
         .sort_values(ascending=False)
         .head(10)
    )
    metrics["top_specialties_head"] = top_specs.to_dict()
    return metrics

def save_output(m: pd.DataFrame, out_path: Path) -> None:
    keep_cols = [
        "person_id",
        "provider_id",
        "drug_concept_id",
        "specialty_source_value",
        "drug_exposure_start_DATE",
        "drug_exposure_end_DATE",
    ]
    missing = [c for c in keep_cols if c not in m.columns]
    if missing:
        raise ValueError(f"Missing expected columns before write: {missing}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out_path, index=False)

def main():
    print(f">>> Using data directory: {DATA_DIR}")
    if not EXPOSURES_PATH.exists():
        raise FileNotFoundError(f"Missing exposures file: {EXPOSURES_PATH}")
    if not PROVIDERS_PATH.exists():
        raise FileNotFoundError(f"Missing providers file: {PROVIDERS_PATH}")

    print(">>> Loading inputs...")
    de, pr = load_inputs(EXPOSURES_PATH, PROVIDERS_PATH)
    print(f"Exposures rows: {len(de):,} | Providers rows: {len(pr):,}")

    print(">>> Merging and cleaning...")
    merged, before, after = merge_and_clean(de, pr)
    print(f"Joined rows: {before:,} | After dropping blank specialties: {after:,}")

    print(">>> Computing metrics...")
    metrics = compute_metrics(merged)

    print(">>> Saving output...")
    save_output(merged, OUT_PATH)

    # Console summary
    print("\n=== STEP 4 SUMMARY ===")
    print(f"Output file: {OUT_PATH}")
    print(f"Rows: {metrics['rows']:,}")
    print(f"Unique providers: {metrics['unique_providers']:,}")
    print(f"Unique drugs: {metrics['unique_drugs']:,}")
    print(f"Share 'unknown' specialty: {metrics['share_unknown']:.3f}")
    epp = metrics["exposures_per_provider"]
    epd = metrics["exposures_per_drug"]
    print(f"Exposures per provider — mean {epp['mean']:.2f}, median {epp['median']:.0f}, p90 {epp['p90']:.0f}, min {epp['min']}, max {epp['max']}")
    print(f"Exposures per drug     — mean {epd['mean']:.2f}, median {epd['median']:.0f}, p90 {epd['p90']:.0f}, min {epd['min']}, max {epd['max']}")
    print("Top specialties (head):")
    for k, v in metrics["top_specialties_head"].items():
        print(f"  {k}: {v:,}")

if __name__ == "__main__":
    try:
        pd.options.mode.copy_on_write = True  # pandas 2.x
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
