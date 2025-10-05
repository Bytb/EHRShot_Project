#!/usr/bin/env python3
"""
Step 2 — Expand high-risk descendants from concept_ancestor

Inputs:
  - data/high_risk_ancestor_ids.json               (from Step 1; flexible schema)
  - data/concept_ancestor.csv                      (columns: ancestor_concept_id, descendant_concept_id, ...)

Outputs:
  - scratch/high_risk_descendant_ids.json          (unique descendant IDs)
  - scratch/high_risk_descendant_ids.csv           (one ID per line)
"""

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Set, Any
import pandas as pd

# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "raw_data"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Inputs (note: use DATA_DIR for OMOP artifacts)
ANCESTOR_JSON = SCRATCH_DIR / "high_risk_ancestor_ids.json"  # adjust if you actually wrote Step 1 into data/
if not ANCESTOR_JSON.exists():
    # fall back to data/ if the file lives there
    fallback = DATA_DIR / "high_risk_ancestor_ids.json"
    if fallback.exists():
        ANCESTOR_JSON = fallback

CONCEPT_ANCESTOR_CSV = Path(os.getenv("CONCEPT_ANCESTOR_CSV", DATA_DIR / "concept_ancestor.csv"))

# Outputs (to SCRATCH_DIR)
DESC_JSON = SCRATCH_DIR / "high_risk_descendant_ids.json"
DESC_CSV  = SCRATCH_DIR / "high_risk_descendant_ids.csv"

# Chunk size for reading large CSVs (no underscores in string literal)
def _parse_chunksize(val: str) -> int:
    try:
        return int(val.replace("_", ""))
    except Exception:
        return 2_000_000

CHUNK_SIZE = _parse_chunksize(os.getenv("CHUNK_SIZE", "2000000"))
# ---------------------------


def _flatten(items: Iterable[Any]) -> Iterable[Any]:
    for x in items:
        if isinstance(x, (list, tuple, set)):
            for y in x:
                yield y
        else:
            yield x

def _as_int_set(it: Iterable[Any]) -> Set[int]:
    out, skipped = set(), []
    for x in it:
        if x is None:
            continue
        if isinstance(x, (int,)):
            out.add(int(x))
            continue
        # strings that might be numeric
        if isinstance(x, str):
            xs = x.strip()
            if xs.isdigit():
                out.add(int(xs))
                continue
        # dict record with a likely id field
        if isinstance(x, dict):
            for k in ("ancestor_concept_id", "concept_id", "id"):
                if k in x:
                    try:
                        out.add(int(x[k]))
                        break
                    except Exception:
                        pass
            else:
                skipped.append(x)
            continue
        # anything else
        skipped.append(x)

    if skipped:
        print(f"[Step 2][WARN] Skipped {len(skipped)} non-integer entries (showing up to 5): {skipped[:5]}")
    return out

def _load_ancestors(fp: Path) -> Set[int]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing ancestor JSON: {fp}")
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept flexible shapes:
    # 1) [21604254, 21604414, ...]
    # 2) {"ids": [ ... ]}
    # 3) {"opioid_analgesics": 21604254, "benzodiazepines": 21604414, ...}
    # 4) {"opioid_analgesics": [21604254, ...], ...}
    # 5) [{"ancestor_concept_id": 21604254}, {"ancestor_concept_id": 21604414}, ...]
    if isinstance(data, dict):
        if "ids" in data:
            raw = data["ids"]
        else:
            # gather dict values (could be ints or lists)
            raw = list(_flatten(data.values()))
    else:
        raw = data

    ids = _as_int_set(raw)
    if not ids:
        raise ValueError("No valid integer ancestor IDs parsed from JSON. "
                         "Check the file contents and schema.")
    return ids

def expand_descendants(concept_ancestor_csv: Path, ancestor_ids: Set[int]) -> Set[int]:
    if not concept_ancestor_csv.exists():
        raise FileNotFoundError(f"Missing concept_ancestor CSV: {concept_ancestor_csv}")

    usecols = ["ancestor_concept_id", "descendant_concept_id"]
    dtypes = {"ancestor_concept_id": "Int64", "descendant_concept_id": "Int64"}

    found_desc = set()
    scanned_rows = 0
    matched_rows = 0

    print("[Step 2] Expanding descendants via concept_ancestor...")
    print(f"[Step 2] Ancestor IDs (n={len(ancestor_ids)}): sample={sorted(list(ancestor_ids))[:5]}")

    for i, chunk in enumerate(pd.read_csv(
        concept_ancestor_csv,
        usecols=usecols,
        dtype=dtypes,
        chunksize=CHUNK_SIZE
    ), start=1):
        scanned_rows += len(chunk)

        mask = chunk["ancestor_concept_id"].isin(ancestor_ids)
        hits = chunk.loc[mask, "descendant_concept_id"].dropna().astype("int64")

        matched_rows += int(mask.sum())
        found_desc.update(hits.tolist())

        if i % 5 == 0:
            print(f"[Step 2] Processed ~{scanned_rows:,} rows ... "
                  f"cumulative matches={matched_rows:,}, unique descendants={len(found_desc):,}")

    print(f"[Step 2] Done scanning concept_ancestor: rows_scanned={scanned_rows:,}, rows_matched={matched_rows:,}")
    print(f"[Step 2] Unique descendant IDs found: {len(found_desc):,}")

    return found_desc

def _save_outputs(desc_ids: Set[int]) -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    with open(DESC_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(desc_ids), f, indent=2)

    pd.Series(sorted(desc_ids), name="drug_concept_id").to_csv(DESC_CSV, index=False)

    print(f"[Step 2] Saved descendant IDs (JSON): {DESC_JSON}")
    print(f"[Step 2] Saved descendant IDs (CSV):  {DESC_CSV}")

def main():
    try:
        print("[Step 2] Loading ancestor IDs...")
        ancestor_ids = _load_ancestors(ANCESTOR_JSON)

        descendants = expand_descendants(CONCEPT_ANCESTOR_CSV, ancestor_ids)

        if not descendants:
            print("[Step 2][WARN] No descendants found. "
                  "Verify your ancestor IDs exist in concept_ancestor and are DRUG-related.")
        _save_outputs(descendants)

        print("[Step 2] Complete. Proceed to Step 3 (filter drug_exposure by these IDs; "
              "optionally restrict to DRUG domain/standard concepts via concept.csv first).")
    except Exception as e:
        print("[Step 2][ERROR]", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
