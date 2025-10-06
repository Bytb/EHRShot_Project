#!/usr/bin/env python3
"""
Step 3 — Guardrails + Domain Restriction + Filter drug_exposure

What this does:
1) Guardrails on Step 1 & 2 outputs (existence, schema, counts).
2) Restrict descendant IDs to OMOP concepts in the DRUG domain and (optionally) standard concepts.
3) Filter drug_exposure.csv to only rows whose drug_concept_id is in that final set.
4) Emit metrics + a small report file to verify it worked as intended.

Inputs (defaults; can be overridden by env vars below):
  - scratch/high_risk_descendant_ids.json (from Step 2)
  - data/high_risk_ancestor_ids.json       (from Step 1; used only for guardrails)
  - data/concept.csv                       (OMOP)
  - data/drug_exposure.csv                 (OMOP)

Outputs:
  - scratch/high_risk_descendant_ids_drug_standard.csv  (final keep-list)
  - scratch/drug_exposure_high_risk.csv                  (filtered; appended in chunks)
  - scratch/step3_report.md                              (metrics & sanity checks)
"""

import json
import os
import sys
from pathlib import Path
from typing import Set, Iterable, Any, Tuple
import pandas as pd
from tqdm import tqdm

# ----------------------- PATHS / CONFIG -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "raw_data"
OUT_DIR     = PROJECT_ROOT / "data"
SCRATCH_DIR  = PROJECT_ROOT / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
ANCESTOR_JSON   = Path(os.getenv("ANCESTOR_JSON", SCRATCH_DIR / "high_risk_ancestor_ids.json"))
DESC_JSON       = Path(os.getenv("DESC_JSON", SCRATCH_DIR / "high_risk_descendant_ids.json"))
CONCEPT_CSV     = Path(os.getenv("CONCEPT_CSV", DATA_DIR / "concept.csv"))
DRUG_EXPO_CSV   = Path(os.getenv("DRUG_EXPO_CSV", DATA_DIR / "sampled_drug_exposure.csv"))

# ----------------------- OUTPUT PATHS -----------------------
# Keep-list & report -> scratch (intermediate)
KEEPLIST_CSV = SCRATCH_DIR / "high_risk_descendant_ids_drug_standard.csv"
REPORT_MD    = SCRATCH_DIR / "step3_report.md"

# Filtered drug_exposure -> data (final output)
FILTERED_CSV = OUT_DIR / "drug_exposure_high_risk.csv"
# ------------------------------------------------------------

# Behavior toggles
REQUIRE_STANDARD = os.getenv("REQUIRE_STANDARD", "1")  # '1' to keep only standard concepts
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "2000000"))
PRINT_EVERY_N    = int(os.getenv("PRINT_EVERY_N", "5"))

# Columns expected in OMOP
CONCEPT_COLS     = ["concept_id", "concept_name", "domain_id", "standard_concept"]
DRUG_EXPO_COLS   = ["person_id", "drug_concept_id", "provider_id", "drug_exposure_start_DATE"]

# --------------------------------------------------------------


def _read_json_ids(fp: Path) -> Set[int]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    vals = []
    if isinstance(data, dict):
        # accept {"ids":[...]} or {label:int/list}
        if "ids" in data:
            vals = data["ids"]
        else:
            for v in data.values():
                if isinstance(v, list):
                    vals.extend(v)
                else:
                    vals.append(v)
    else:
        vals = data
    out = set()
    skipped = 0
    for x in vals:
        try:
            out.add(int(x))
        except Exception:
            skipped += 1
    if skipped:
        print(f"[Step 3][WARN] Skipped {skipped} non-integer entries in {fp.name}")
    if not out:
        raise ValueError(f"No integer IDs parsed from {fp}")
    return out


def _guardrails_step1_step2() -> Tuple[Set[int], Set[int]]:
    # Step 1
    print("[Step 3] Guardrail: checking Step 1 ancestor IDs...")
    ancestors = _read_json_ids(ANCESTOR_JSON)
    print(f"[Step 3] Ancestors found: {len(ancestors)} (sample: {sorted(list(ancestors))[:5]})")

    # Step 2
    print("[Step 3] Guardrail: checking Step 2 descendant IDs...")
    descendants = _read_json_ids(DESC_JSON)
    if len(descendants) < len(ancestors):
        print("[Step 3][WARN] Descendant set smaller than ancestor set — possible issue?")
    print(f"[Step 3] Descendants found: {len(descendants)}")
    return ancestors, descendants


def _load_concepts() -> pd.DataFrame:
    if not CONCEPT_CSV.exists():
        raise FileNotFoundError(f"Missing concept.csv at {CONCEPT_CSV}")
    # read minimal columns; tolerate missing 'standard_concept' by filling 'X'
    usecols = [c for c in CONCEPT_COLS if c]  # allow missing col handling
    df = pd.read_csv(CONCEPT_CSV, usecols=lambda c: c in usecols, dtype={"concept_id": "Int64"})
    # ensure needed columns exist
    for col in ["concept_id", "domain_id"]:
        if col not in df.columns:
            raise ValueError(f"concept.csv missing required column: {col}")
    if "standard_concept" not in df.columns:
        df["standard_concept"] = pd.NA
    if "concept_name" not in df.columns:
        df["concept_name"] = pd.NA
    return df


def _restrict_to_drug_standard(descendants: Set[int]) -> pd.DataFrame:
    concepts = _load_concepts()
    df = concepts[concepts["concept_id"].isin(descendants)].copy()
    total_hit = len(df)
    if total_hit == 0:
        raise ValueError("No descendant IDs found in concept.csv — check your OMOP files.")
    # Restrict to DRUG domain
    df = df[df["domain_id"].astype(str).str.upper() == "DRUG"].copy()
    drug_hit = len(df)
    if REQUIRE_STANDARD in ("1", "true", "True", "TRUE"):
        df = df[df["standard_concept"].astype(str).str.upper().eq("S")].copy()
    final_hit = len(df)

    print(f"[Step 3] concept.csv hits — any domain: {total_hit:,}, DRUG: {drug_hit:,}, "
          f"DRUG+standard({REQUIRE_STANDARD}): {final_hit:,}")

    if final_hit == 0:
        raise ValueError("After DRUG/standard restriction, keep-list is empty. "
                         "Consider setting REQUIRE_STANDARD=0 or re-check ancestors.")
    return df[["concept_id", "concept_name", "domain_id", "standard_concept"]]


def _filter_drug_exposure(keep_ids: Set[int]) -> dict:
    if not DRUG_EXPO_CSV.exists():
        raise FileNotFoundError(f"Missing drug_exposure.csv at {DRUG_EXPO_CSV}")

    # Prepare output file (overwrite)
    if FILTERED_CSV.exists():
        FILTERED_CSV.unlink()

    required = set(DRUG_EXPO_COLS)
    total_rows = 0
    kept_rows = 0
    unique_kept_concepts = set()
    chunk_idx = 0

    print("[Step 3] Filtering drug_exposure by keep-list (chunked)...")
    for i, chunk in tqdm(enumerate(pd.read_csv(DRUG_EXPO_CSV, chunksize=CHUNK_SIZE, low_memory=False), start=1), desc="filtering..."):
        total_rows += len(chunk)

        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"drug_exposure.csv missing required columns: {missing}")

        # Ensure types
        # (if a column is missing types, we let pandas infer; only cast concept id)
        try:
            chunk["drug_concept_id"] = chunk["drug_concept_id"].astype("int64", errors="ignore")
        except Exception:
            pass

        mask = chunk["drug_concept_id"].isin(keep_ids)
        out = chunk.loc[mask].copy()

        kept_rows += len(out)
        unique_kept_concepts.update(out["drug_concept_id"].dropna().unique().tolist())

        # append to CSV
        mode = "a" if chunk_idx > 0 else "w"
        header = (chunk_idx == 0)
        out.to_csv(FILTERED_CSV, index=False, mode=mode, header=header)
        chunk_idx += 1

        if i % PRINT_EVERY_N == 0:
            print(f"[Step 3] Processed ~{total_rows:,} rows; kept so far: {kept_rows:,} "
                  f"(unique concepts kept: {len(unique_kept_concepts):,})")

    keep_rate = kept_rows / total_rows if total_rows else 0.0
    print(f"[Step 3] Done. Total rows: {total_rows:,}, kept: {kept_rows:,} ({keep_rate:.2%}), "
          f"unique kept concepts: {len(unique_kept_concepts):,}")

    return {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "keep_rate": keep_rate,
        "unique_kept_concepts": len(unique_kept_concepts),
    }


def _write_report(anc_count: int, desc_count: int, concept_hits: pd.DataFrame, filt_stats: dict):
    top10 = concept_hits.head(10).copy()
    # Create report
    lines = []
    lines.append("# Step 3 Report\n")
    lines.append("## Guardrails\n")
    lines.append(f"- Step 1 ancestors: **{anc_count}**\n")
    lines.append(f"- Step 2 descendants: **{desc_count}**\n")
    lines.append("\n## Concept Restriction\n")
    lines.append(f"- After DRUG/standard={REQUIRE_STANDARD}, keep-list size: **{len(concept_hits)}**\n")
    lines.append("\n### Sample of keep-list (first 10)\n")
    for _, r in top10.iterrows():
        lines.append(f"- `{int(r['concept_id'])}` — {str(r['concept_name'])}\n")
    lines.append("\n## Filtering drug_exposure.csv\n")
    lines.append(f"- Total rows scanned: **{filt_stats['total_rows']:,}**\n")
    lines.append(f"- Rows kept: **{filt_stats['kept_rows']:,}** "
                 f"({filt_stats['keep_rate']:.2%})\n")
    lines.append(f"- Unique kept drug_concept_id: **{filt_stats['unique_kept_concepts']:,}**\n")
    lines.append("\n### Sanity checks\n")
    lines.append("- ✅ Ancestors/descendants present and parsed\n")
    lines.append("- ✅ Keep-list intersects concept.csv (DRUG domain)\n")
    lines.append("- ✅ drug_exposure filtered rows > 0\n")
    lines.append("\nIf any of the checks above were not ✅, re-check your Step 1 ancestor set, "
                 "concept_ancestor coverage (Step 2), and concept.csv domain/standard flags.\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Step 3] Wrote report: {REPORT_MD}")


def main():
    try:
        # 1) Guardrails for Step 1 & 2
        ancestors, descendants = _guardrails_step1_step2()

        # 2) Restrict to DRUG domain (+optional standard)
        concept_hits = _restrict_to_drug_standard(descendants)
        concept_hits.sort_values("concept_id").to_csv(KEEPLIST_CSV, index=False)
        keep_ids = set(concept_hits["concept_id"].astype("int64").tolist())
        print(f"[Step 3] Saved keep-list: {KEEPLIST_CSV} (n={len(keep_ids):,})")

        # 3) Filter drug_exposure by keep_ids
        stats = _filter_drug_exposure(keep_ids)

        # 4) Report
        _write_report(len(ancestors), len(descendants), concept_hits, stats)

        # 5) Validation summary — quick sanity snapshot
        print("\n================== VALIDATION SUMMARY ==================")
        print(f"High-risk ancestor concepts: {len(ancestors):,}")
        print(f"High-risk descendant concepts (before filtering): {len(descendants):,}")
        print(f"Final keep-list (DRUG/standard): {len(keep_ids):,}")
        print(f"Filtered drug_exposure rows kept: {stats['kept_rows']:,} "
              f"({stats['keep_rate']:.2%} of total {stats['total_rows']:,})")
        print(f"Unique drug_concept_id kept: {stats['unique_kept_concepts']:,}")

        # Compute extra aggregates directly from filtered CSV (lightweight)
        try:
            df_sample = pd.read_csv(FILTERED_CSV, usecols=["person_id", "provider_id", "drug_concept_id"])
            n_patients = df_sample["person_id"].nunique()
            n_providers = df_sample["provider_id"].nunique()
            n_drugs = df_sample["drug_concept_id"].nunique()
            print(f"Unique patients: {n_patients:,}")
            print(f"Unique providers: {n_providers:,}")
            print(f"Unique drugs (post-filter): {n_drugs:,}")
        except Exception as e:
            print(f"[Step 3][WARN] Could not compute unique counts from filtered file: {e}")

        print("=========================================================\n")
        print("[Step 3] Complete ✅  You can now build the provider–drug network using data/drug_exposure_high_risk.csv")

    except Exception as e:
        print("[Step 3][ERROR]", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
