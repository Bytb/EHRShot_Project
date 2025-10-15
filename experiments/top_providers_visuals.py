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
# 4. Identify Outlier Provider
# -----------------------------
# threshold can be adjusted slightly if there’s more than one near that range
outlier_df = df[(df["exposures"] > 100) & (df["drugs_prescribed"] >= 10) & (df["drugs_prescribed"] <= 12)]
print(outlier_df.sort_values("exposures", ascending=False))

# -----------------------------
# 3. Scatter Plot
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df["drugs_prescribed"], df["exposures"], alpha=0.3, label="All providers")
plt.scatter(top_drugs["drugs_prescribed"], top_drugs["exposures"], color="orange", label="Top 5% (drugs)")
plt.scatter(top_exposures["drugs_prescribed"], top_exposures["exposures"], color="red", label="Top 5% (exposures)")
# highlight outlier
plt.scatter(outlier_df["drugs_prescribed"], outlier_df["exposures"], color="lime", edgecolor="black", label="Outlier")

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

# -------------------------------------------------
# Task 3 — Heatmap: Top prescriber vs specialty mean
# -------------------------------------------------
print("\n[INFO] Building heatmap: Top vs. Mean (Top 7 specialties) ...")

TABLES_DIR = HERE / "Results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# 1) Select top 7 specialties by provider count
# Consistent with PPR: rank specialties by TOTAL EXPOSURES (sum of weighted degree)
df_clean = df[~df["specialty"].str.strip().isin(["", "0", "Unknown"])].copy()
spec_exposure_sum = (
    df_clean.groupby("specialty")["exposures"].sum().sort_values(ascending=False)
)
top7_specs = spec_exposure_sum.head(7).index.tolist()

df_top7 = df[df["specialty"].isin(top7_specs)].copy()

# 2) Per-specialty means (weighted degree = exposures; #drugs = unweighted degree)
spec_means = (
    df_top7.groupby("specialty")
           .agg(mean_exposures=("exposures", "mean"),
                mean_drugs=("drugs_prescribed", "mean"),
                n_providers=("provider_id", "count"))
)

# 3) Top provider within each specialty by exposures (tie-breaker: #drugs)
top_rows = (
    df_top7.sort_values(["specialty", "exposures", "drugs_prescribed"], ascending=[True, False, False])
           .groupby("specialty", as_index=False)
           .head(1)
           .set_index("specialty")
)

# 4) Compose ratios
stats = pd.DataFrame({
    "top_provider_id": top_rows["provider_id"],
    "top_exposures": top_rows["exposures"],
    "top_drugs": top_rows["drugs_prescribed"],
    "mean_exposures": spec_means["mean_exposures"],
    "mean_drugs": spec_means["mean_drugs"],
    "n_providers": spec_means["n_providers"],
})
stats["ratio_exposures"] = stats["top_exposures"] / stats["mean_exposures"]
stats["ratio_drugs"] = stats["top_drugs"] / stats["mean_drugs"]

# 5) Order specialties by exposures ratio (desc); build 2-row matrix
ordered_specs = stats.sort_values("ratio_exposures", ascending=False).index.tolist()
heat_df = pd.DataFrame(
    {s: [stats.loc[s, "ratio_exposures"], stats.loc[s, "ratio_drugs"]] for s in ordered_specs},
    index=["Weighted degree ratio", "Number of drugs ratio"]
)

# 6) Save table
csv_path = TABLES_DIR / "specialty_top_vs_mean.csv"
stats.loc[ordered_specs].to_csv(csv_path)
print(f"[INFO] Saved ratios table → {csv_path}")

# 7) Plot heatmap (Reds: darker = farther above mean)
plt.figure(figsize=(max(8, 1.2*len(ordered_specs)), 3.8))
ax = sns.heatmap(
    heat_df,
    annot=True, fmt=".2f",
    cmap=sns.color_palette("Reds", as_cmap=True),
    cbar=True, vmin=None, vmax=None,
    linewidths=0.5, linecolor="white"
)
ax.set_title("Top Prescriber vs. Specialty Mean (Top 7 Specialties)")
ax.set_xlabel("Specialty")
ax.set_ylabel("")  # row labels already descriptive

# 8) Highlight: (a) max exposures-ratio specialty (black), (b) Family Medicine (blue)
import matplotlib.patches as patches

def _outline_column(ax, col_idx, n_rows, color, lw=2.5, pad=0.02):
    ax.add_patch(
        patches.Rectangle(
            (col_idx + pad, 0 + pad),            # (x, y) bottom-left in data coords
            1 - 2*pad,                           # width = one column
            n_rows - 2*pad,                      # height = all rows
            fill=False, edgecolor=color, linewidth=lw
        )
    )

spec_max = ordered_specs[0] if ordered_specs else None
if spec_max is not None:
    _outline_column(ax, ordered_specs.index(spec_max), heat_df.shape[0], color="black", lw=3.0)

fam_idx = next((i for i, s in enumerate(ordered_specs) if s.lower() == "family medicine"), None)
if fam_idx is not None:
    _outline_column(ax, fam_idx, heat_df.shape[0], color="dodgerblue", lw=3.0)

plt.tight_layout()
hm_path = OUTPUT_DIR / "heatmap_top_vs_specialty.png"
plt.savefig(hm_path, dpi=300, bbox_inches="tight")
print(f"[INFO] Saved heatmap → {hm_path}")
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

