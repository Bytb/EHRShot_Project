import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import pickle
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GRAPH_PATH = HERE / "data" / "provider_drug_graph.pkl"
OUTPUT_DIR = HERE / "Results" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

print(f"[INFO] Loaded graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

# -----------------------------
# 1. Compute degree metrics
# -----------------------------
# --- Corrected version ---
providers = [n for n, d in G.nodes(data=True) if d.get("ntype") == "provider"]

# Degree and weighted degree
degree_dict = dict(G.degree(providers))
weighted_degree_dict = dict(G.degree(providers, weight="weight"))

# Specialty
specialty_dict = {n: G.nodes[n].get("specialty", "Unknown") for n in providers}


# Combine into DataFrame
df = pd.DataFrame({
    "provider_id": providers,
    "drugs_prescribed": [degree_dict[p] for p in providers],
    "exposures": [weighted_degree_dict[p] for p in providers],
    "specialty": [specialty_dict[p] for p in providers],
})
print(f"[DEBUG] Providers in df: {len(df)} (expect hundreds or more)")
print(df.head())


# -----------------------------
# 2. Identify top 5% providers
# -----------------------------
top_p = 0.05
cutoff_drugs = np.percentile(df["drugs_prescribed"], 100 * (1 - top_p))
cutoff_exposures = np.percentile(df["exposures"], 100 * (1 - top_p))

top_drugs = df[df["drugs_prescribed"] >= cutoff_drugs].copy()
top_exposures = df[df["exposures"] >= cutoff_exposures].copy()

# -----------------------------
# 3. Scatter Plot
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df["drugs_prescribed"], df["exposures"], alpha=0.3, label="All providers")
plt.scatter(top_drugs["drugs_prescribed"], top_drugs["exposures"], color="orange", label="Top 5% (drugs)")
plt.scatter(top_exposures["drugs_prescribed"], top_exposures["exposures"], color="red", label="Top 5% (exposures)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Drugs Prescribed (Degree)")
plt.ylabel("Total Exposures (Weighted Degree)")
plt.title("Provider Degree vs. Exposures (Top 5% Highlighted)")
plt.legend()
plt.tight_layout()
scatter_path = OUTPUT_DIR / "scatter_top_providers.png"
plt.savefig(scatter_path, dpi=300)
print(f"[INFO] Saved scatter plot → {scatter_path}")
plt.show()

# -----------------------------
# 5. Pie Charts for Specialty Composition
# -----------------------------
def plot_pie(df_subset, title, filename):
    specialty_counts = df_subset["specialty"].value_counts()
    total = specialty_counts.sum()

    # Build labels with percentages for legend
    legend_labels = [
        f"{spec} ({count / total * 100:.1f}%)"
        for spec, count in specialty_counts.items()
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts = ax.pie(
        specialty_counts,
        labels=None,  # keep pie clean
        autopct=None,  # no text on slices
        startangle=90
    )

    # Legend with percentages
    ax.legend(
        wedges,
        legend_labels,
        title="Specialties",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=False
    )

    ax.set_title(title)
    plt.tight_layout()
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved pie chart → {save_path}")
    plt.show()

# Call:
plot_pie(top_drugs, "Specialty Composition (Top 5% by Drugs Prescribed)", "pie_top_drugs.png")
plot_pie(top_exposures, "Specialty Composition (Top 5% by Exposures)", "pie_top_exposures.png")

