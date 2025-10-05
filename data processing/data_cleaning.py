# clean_provider_and_drug_exposure_keep_refills.py
import pandas as pd
from pathlib import Path

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "raw_data"        # sibling 'data' folder
OUT_DIR = BASE_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROVIDER_CSV = DATA_DIR / "provider.csv"
DRUG_EXPOSURE_CSV = DATA_DIR / "sampled_drug_exposure.csv"

# ---------- LOAD ----------
prov = pd.read_csv(PROVIDER_CSV, low_memory=False)
dex = pd.read_csv(DRUG_EXPOSURE_CSV, low_memory=False)

# ---------- KEEP ONLY NEEDED COLUMNS ----------
prov_cols = ["provider_id", "specialty_concept_id", "specialty_source_value"]
for c in prov_cols:
    if c not in prov.columns:
        prov[c] = pd.NA
prov = prov[prov_cols].copy()

drug_cols = ["person_id", "drug_concept_id", "drug_exposure_start_date",
             "drug_exposure_end_date", "provider_id"]
for c in drug_cols:
    if c not in dex.columns:
        dex[c] = pd.NA
dex = dex[drug_cols].copy()

# ---------- TYPE COERCION ----------
# providers
prov["provider_id"] = pd.to_numeric(prov["provider_id"], errors="coerce").astype("Int64")
prov["specialty_concept_id"] = pd.to_numeric(prov["specialty_concept_id"], errors="coerce").astype("Int64")
prov["specialty_source_value"] = prov["specialty_source_value"].astype("string")

# drug exposures
for col in ["person_id", "drug_concept_id", "provider_id"]:
    dex[col] = pd.to_numeric(dex[col], errors="coerce").astype("Int64")
# Parse dates (keep as datetime; OK if NA)
for col in ["drug_exposure_start_date", "drug_exposure_end_date"]:
    dex[col] = pd.to_datetime(dex[col], errors="coerce")

# ---------- BASIC CLEAN ----------
# drop rows without critical IDs
dex = dex.dropna(subset=["person_id", "drug_concept_id", "provider_id"]).copy()

# prefer specialty_source_value as the best label; if blank, fall back to concept_id string
prov["specialty_best"] = prov["specialty_source_value"]
blank = prov["specialty_best"].isna() | (prov["specialty_best"].str.strip() == "")
prov.loc[blank, "specialty_best"] = prov["specialty_concept_id"].astype("string")

# keep only providers that appear in the drug exposure data
active_provider_ids = dex["provider_id"].dropna().unique()
prov_active = prov[prov["provider_id"].isin(active_provider_ids)].reset_index(drop=True)

# ---------- SAVE ----------
prov_active.to_csv(OUT_DIR / "providers.csv", index=False)
dex.to_csv(DATA_DIR / "drug_semiprocessed.csv", index=False)

# ---------- SUMMARIES ----------
print("[SUMMARY] Providers (clean, active):", len(prov_active))
print("[SUMMARY] Drug exposures (all, refills kept):", len(dex))
print("Unique providers in drug data:", dex["provider_id"].nunique())
print("Unique drugs:", dex["drug_concept_id"].nunique())
print("Unique patients:", dex["person_id"].nunique())
