#!/usr/bin/env python3
"""
Toy bipartite network generator (visualization-first).

Assumptions:
- This script lives in a folder that is a SIBLING of the 'data/' folder:
    repo_root/
      data/
      scripts_or_your_folder/
        make_toy_bipartite.py  <-- this file

What it creates in data/:
- toy_exposures.csv                          (person_id, provider_id, drug_concept_id)
- toy_edges_provider_drug.csv                (provider_id, drug_concept_id, weight_exposures, weight_patients)
- toy_nodes_providers.csv                    (provider_id, specialty_source_value, degree_patients, label)
- toy_nodes_drugs.csv                        (drug_concept_id, drug_name, drug_class, degree_patients, label)
- toy_layout_coords.csv                      (node_type, node_id, x, y, label)  # fixed coords for reproducible plots
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = DATA_DIR / "toy_graph"

EXPOSURES_OUT = OUT_DIR / "toy_exposures.csv"
EDGES_OUT = OUT_DIR / "toy_edges_provider_drug.csv"
PROV_OUT = OUT_DIR / "toy_nodes_providers.csv"
DRUG_OUT = OUT_DIR / "toy_nodes_drugs.csv"
COORDS_OUT = OUT_DIR / "toy_layout_coords.csv"

# ---------- Define tiny, readable universe ----------
# Providers (6): use numeric IDs but keep readable labels for plotting
providers = [
    {"provider_id": 1001, "specialty_source_value": "Emergency Medicine", "label": "P1 (EM)"},
    {"provider_id": 1002, "specialty_source_value": "Emergency Medicine", "label": "P2 (EM)"},
    {"provider_id": 1003, "specialty_source_value": "Anesthesia",         "label": "P3 (AN)"},
    {"provider_id": 1004, "specialty_source_value": "Anesthesia",         "label": "P4 (AN)"},
    {"provider_id": 1005, "specialty_source_value": "Internal Medicine",  "label": "P5 (IM)"},
    {"provider_id": 1006, "specialty_source_value": "Nurse Practitioner", "label": "P6 (NP)"},
]
prov_df = pd.DataFrame(providers)

# Drugs (5): numeric concept IDs + names + broad classes
drugs = [
    {"drug_concept_id": 2001, "drug_name": "Opioid A",   "drug_class": "Opioid",      "label": "D1 (OpA)"},
    {"drug_concept_id": 2002, "drug_name": "Opioid B",   "drug_class": "Opioid",      "label": "D2 (OpB)"},
    {"drug_concept_id": 2003, "drug_name": "Benzo A",    "drug_class": "Benzodiazepine","label": "D3 (BnA)"},
    {"drug_concept_id": 2004, "drug_name": "Benzo B",    "drug_class": "Benzodiazepine","label": "D4 (BnB)"},
    {"drug_concept_id": 2005, "drug_name": "MuscleRelax","drug_class": "Muscle Relaxant","label": "D5 (MR)"},
]
drug_df = pd.DataFrame(drugs)

# ---------- Hand-crafted motifs (counts = unique patients) ----------
# Each entry = one provider–drug edge with a specified number of unique patients.
# Keep small integers (1–8) for visible edge width variation; degrees <= 4-ish.
motifs = [
    # Concentrated prescriber: P3 (AN) -> D1 (Opioid A) heavy
    (1003, 2001, 8),
    # Hub drug: D2 (Opioid B) touched by EM, IM, NP
    (1001, 2002, 4),
    (1002, 2002, 3),
    (1005, 2002, 5),
    (1006, 2002, 2),
    # Balanced triad on D3 (Benzo A): EM providers share evenly
    (1001, 2003, 3),
    (1002, 2003, 3),
    # Cross-specialty overlap on D4 (Benzo B): AN + EM small counts
    (1003, 2004, 2),
    (1004, 2004, 2),
    (1001, 2004, 1),
    # Niche provider: P6 (NP) only prescribes Muscle Relaxant
    (1006, 2005, 4),
    # Give P5 (IM) one niche on D3 to avoid isolates on that side
    (1005, 2003, 2),
]

# ---------- Expand to synthetic exposures (one row per unique patient) ----------
# Person IDs: deterministic, compact, no PHI. We just increment a counter.
rows = []
person_counter = 10001
for provider_id, drug_id, n_patients in motifs:
    for _ in range(n_patients):
        rows.append(
            {"person_id": person_counter, "provider_id": provider_id, "drug_concept_id": drug_id}
        )
        person_counter += 1

exposures_df = pd.DataFrame(rows, columns=["person_id", "provider_id", "drug_concept_id"])
exposures_df.to_csv(EXPOSURES_OUT, index=False)

# ---------- Aggregate to edges (both exposure and unique-patient weights are identical here) ----------
edges = (
    exposures_df
    .groupby(["provider_id", "drug_concept_id"], as_index=False)
    .agg(weight_exposures=("person_id", "size"),
         weight_patients =("person_id", "nunique"))
)
edges.to_csv(EDGES_OUT, index=False)

# ---------- Add degrees to node tables (degree by patients) ----------
deg_prov = edges.groupby("provider_id")["weight_patients"].sum().rename("degree_patients")
deg_drug = edges.groupby("drug_concept_id")["weight_patients"].sum().rename("degree_patients")

prov_final = prov_df.merge(deg_prov, on="provider_id", how="left").fillna({"degree_patients": 0}).astype({"degree_patients":"int64"})
drug_final = drug_df.merge(deg_drug, on="drug_concept_id", how="left").fillna({"degree_patients": 0}).astype({"degree_patients":"int64"})

prov_final.to_csv(PROV_OUT, index=False)
drug_final.to_csv(DRUG_OUT, index=False)

# ---------- Fixed coordinates for reproducible plotting ----------
# Providers on the left (x=0), drugs on the right (x=1). Order within column aids readability.
coords = []
# Sort providers by specialty then label for tidy stacking
prov_plot = prov_final.sort_values(by=["specialty_source_value", "label"]).reset_index(drop=True)
for i, r in prov_plot.iterrows():
    coords.append({"node_type":"provider", "node_id": r["provider_id"], "x": 0.0, "y": float(i), "label": r["label"]})

# Sort drugs by class then label
drug_plot = drug_final.sort_values(by=["drug_class", "label"]).reset_index(drop=True)
for i, r in drug_plot.iterrows():
    coords.append({"node_type":"drug", "node_id": r["drug_concept_id"], "x": 1.0, "y": float(i), "label": r["label"]})

coords_df = pd.DataFrame(coords, columns=["node_type", "node_id", "x", "y", "label"])
coords_df.to_csv(COORDS_OUT, index=False)

print("=== TOY NETWORK WRITTEN ===")
print(f"Exposures: {EXPOSURES_OUT}")
print(f"Edges:     {EDGES_OUT}")
print(f"Providers: {PROV_OUT}")
print(f"Drugs:     {DRUG_OUT}")
print(f"Coords:    {COORDS_OUT}")
print("\nQuick edge head:")
print(edges.head().to_string(index=False))
