# clean_drugs.py — Step 1: pick & persist high-risk ancestor concept IDs
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CONCEPT_ANCESTOR_CSV = HERE / "raw_data" / "concept_ancestor.csv"
OUTPUT_JSON = HERE / "scratch" / "high_risk_ancestor_ids.json"

# ATC ancestors (from Athena)
HIGH_RISK_ANCESTORS = {
    "opioid_analgesics": [21604254],                       # N02A (Opioids)
    "benzodiazepine_receptor_agonists": [21604656, 21604635, 21604414],  # N05BA, N05CD, N03AE
    "cns_stimulants": [21604678]                           # N06BA (Centrally acting sympathomimetics)
}

def _flatten(values):
    """Yield a flat stream of scalars from arbitrarily nested lists/tuples/sets."""
    for v in values:
        if isinstance(v, (list, tuple, set)):
            yield from _flatten(v)
        else:
            yield v

def _assert_ids_set(ids_dict: dict):
    """Each entry must be an int or a non-empty list/tuple/set of ints/floats."""
    missing_keys, bad_types = [], []
    for k, v in ids_dict.items():
        if v is None:
            missing_keys.append(k)
        elif isinstance(v, (list, tuple, set)):
            if len(v) == 0:
                missing_keys.append(k)
            elif not all(isinstance(x, (int, float)) for x in v):
                bad_types.append(k)
        elif not isinstance(v, (int, float)):
            bad_types.append(k)
    if missing_keys or bad_types:
        msgs = []
        if missing_keys: msgs.append("missing or empty: " + ", ".join(missing_keys))
        if bad_types:    msgs.append("non-numeric: " + ", ".join(bad_types))
        raise ValueError("Invalid HIGH_RISK_ANCESTORS entries -> " + " | ".join(msgs))

def _check_concept_ancestor_contains_ids(csv_path: Path, ancestor_ids):
    """
    Verify each proposed ancestor_id exists in concept_ancestor.csv.
    Returns {ancestor_id: True/False}.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["ancestor_concept_id"],
                     dtype={"ancestor_concept_id": "Int64"})
    present = set(df["ancestor_concept_id"].dropna().astype("int64").tolist())
    return {aid: (aid in present) for aid in ancestor_ids}

def _persist_ids(json_path: Path, ids_dict: dict):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ids_dict, f, indent=2)
    print(f"[Step 1] Saved ancestor IDs to: {json_path}")

def main_step_1():
    print("[Step 1] Selecting high-risk ancestor concept IDs...")
    _assert_ids_set(HIGH_RISK_ANCESTORS)

    # FLATTEN -> ints
    ancestor_list = [int(a) for a in _flatten(HIGH_RISK_ANCESTORS.values())]

    # sanity print
    sample = sorted(set(ancestor_list))
    print(f"[Step 1] Checking {len(ancestor_list)} ancestor IDs: {sample[:10]}..."
          + ("" if len(sample) <= 10 else " (showing first 10)"))

    presence = _check_concept_ancestor_contains_ids(CONCEPT_ANCESTOR_CSV, ancestor_list)

    not_found = [aid for aid, ok in presence.items() if not ok]
    if not_found:
        raise ValueError(
            "The following ancestor_concept_id(s) were not found as ancestors in "
            f"{CONCEPT_ANCESTOR_CSV}:\n  {sorted(set(not_found))}\n"
            "Double-check these are ATC class concept_ids present in your vocabulary "
            "(e.g., N02A, N05BA, N05CD, N03AE, N06BA) and that ATC is included in your OMOP vocabs."
        )

    _persist_ids(OUTPUT_JSON, HIGH_RISK_ANCESTORS)
    print("[Step 1] Done. Proceed to Step 2 (expand descendants via concept_ancestor).")

if __name__ == "__main__":
    main_step_1()
