"""
build_real_provider_drug_graph.py
---------------------------------
Constructs the real provider–drug bipartite graph from the joined EHRShot dataset.

Inputs:
    data/high_risk_drug_exposure_with_providers.csv

Outputs (mini sample):
    data/toy/degree_distribution.png
    data/toy/weight_distribution.png
    data/toy/mini_provider_drug_edgelist.csv
    data/toy/mini_provider_nodes.csv
    data/toy/mini_provider_drug.pkl
"""

from pathlib import Path
import os
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# -------------------------------------------------
# Config
# -------------------------------------------------
HERE = Path(__file__).resolve().parents[1]
DATA_PATH = HERE / "data" / "high_risk_drug_exposure_with_providers.csv"
OUTPUT_DIR = HERE / "Results" / "toy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MINI_N_PROVIDERS = 100
MINI_N_DRUGS = 30

# -------------------------------------------------
# 1) Load and clean
# -------------------------------------------------
print(f"[INFO] Reading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

required = {"person_id", "provider_id", "drug_concept_id", "specialty_source_value"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in input CSV: {missing}")

df = df.dropna(subset=["provider_id", "drug_concept_id"])
df["provider_id"] = df["provider_id"].astype(str)
df["drug_concept_id"] = df["drug_concept_id"].astype(str)
df["specialty_source_value"] = df["specialty_source_value"].fillna("Unknown")

print(f"[INFO] Rows after cleaning: {len(df):,}")

# -------------------------------------------------
# 2) Aggregate edges
# -------------------------------------------------
print("[INFO] Aggregating by (provider_id, drug_concept_id) ...")
agg = (
    df.groupby(["provider_id", "drug_concept_id"])
      .agg(
          n_exposures=("person_id", "count"),   # includes refills
          n_patients=("person_id", "nunique")   # unique individuals
      )
      .reset_index()
)
print(f"[INFO] Unique provider–drug pairs: {len(agg):,}")

# -------------------------------------------------
# 3) Build bipartite graph
# -------------------------------------------------
print("[INFO] Building bipartite graph ...")
G = nx.Graph()

providers = (
    df[["provider_id", "specialty_source_value"]]
    .drop_duplicates()
    .rename(columns={"specialty_source_value": "specialty"})
)

# Add provider nodes
for _, row in providers.iterrows():
    G.add_node(row.provider_id, ntype="provider", specialty=row.specialty)

# Add drug nodes
for d in agg["drug_concept_id"].unique():
    G.add_node(d, ntype="drug")

# Add edges with exposure weights
for _, row in agg.iterrows():
    G.add_edge(
        row.provider_id,
        row.drug_concept_id,
        weight=row.n_exposures,
        n_patients=row.n_patients
    )

print(f"[INFO] Graph built: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

# -------------------------------------------------
# 4) Summary
# -------------------------------------------------
n_prov = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "provider")
n_drug = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "drug")
print(f"  Providers: {n_prov:,} | Drugs: {n_drug:,}")

print("\n[INFO] Top specialties:")
print(providers["specialty"].value_counts().head(15))

print("\n[INFO] Top 10 edges by exposures:")
top_edges = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)[:10]
for u, v, data in top_edges:
    print(f"  {u} — {v}: {data['weight']} exposures | {data['n_patients']} patients")

# -------------------------------------------------
# 5) Distributions (log-scale) and save PNGs
# -------------------------------------------------
print("\n[INFO] Plotting and saving distributions (log scale)...")

# Degree distributions
prov_deg = [G.degree(n, weight="weight") for n, d in G.nodes(data=True) if d["ntype"] == "provider"]
drug_deg = [G.degree(n, weight="weight") for n, d in G.nodes(data=True) if d["ntype"] == "drug"]

plt.figure(figsize=(7.5, 5.5))
plt.hist(prov_deg, bins=30, alpha=0.7, label="Providers")
plt.hist(drug_deg, bins=30, alpha=0.7, label="Drugs")
plt.yscale("log")
plt.xlabel("Weighted degree (sum of exposures)")
plt.ylabel("Count (log scale)")
plt.title("Degree Distribution (Providers vs Drugs)")
plt.legend()
plt.tight_layout()
deg_path = OUTPUT_DIR / "degree_distribution.png"
plt.savefig(deg_path, dpi=300)
print(f"[INFO] Saved degree distribution → {deg_path}")
plt.close()

# Edge-weight distribution
weights = [d["weight"] for _, _, d in G.edges(data=True)]
plt.figure(figsize=(7.5, 5.5))
plt.hist(weights, bins=40, alpha=0.85, color="tomato")
plt.yscale("log")
plt.xlabel("Edge weight (n_exposures)")
plt.ylabel("Count (log scale)")
plt.title("Edge-Weight Distribution")
plt.tight_layout()
wt_path = OUTPUT_DIR / "weight_distribution.png"
plt.savefig(wt_path, dpi=300)
print(f"[INFO] Saved weight distribution → {wt_path}")
plt.close()

# -------------------------------------------------
# 6) Random mini sample and save
# -------------------------------------------------
print("\n[INFO] Creating random mini-sample ...")

# --- Better mini-sample: edge-aware and connected ---
from collections import defaultdict
import numpy as np

providers_all = [n for n, d in G.nodes(data=True) if d["ntype"] == "provider"]

# Degree-weighted random sample of providers (still random, but favors active ones)
prov_strength = {p: G.degree(p, weight="weight") for p in providers_all}
prov_pool = list(prov_strength.keys())
weights_arr = np.array([prov_strength[p] for p in prov_pool], dtype=float)
weights_arr = weights_arr / weights_arr.sum() if weights_arr.sum() > 0 else None
k_prov = min(30, len(prov_pool))
sample_prov = list(np.random.choice(prov_pool, size=k_prov, replace=False, p=weights_arr))

# Pick drugs those providers actually use (top by total exposures across sampled providers)
drug_totals = defaultdict(int)
for p in sample_prov:
    for d, data in G[p].items():
        drug_totals[d] += data.get("weight", 1)

k_drug = min(10, len(drug_totals))
sample_drug = [d for d, _ in sorted(drug_totals.items(), key=lambda kv: kv[1], reverse=True)[:k_drug]]

sub_nodes = set(sample_prov) | set(sample_drug)
G_sub = G.subgraph(sub_nodes).copy()

# Drop any isolates, just in case
isolates = [n for n in list(G_sub.nodes()) if G_sub.degree(n) == 0]
G_sub.remove_nodes_from(isolates)
print(f"[INFO] Mini graph (edge-aware): {G_sub.number_of_nodes():,} nodes | {G_sub.number_of_edges():,} edges")


# -------------------------------------------------
# Function: visualize_mini_graph
# -------------------------------------------------
def visualize_mini_graph(
    G_sub,
    save_path=None,
    seed=42,
    layout="bipartite",     # "bipartite" or "spring"
    label_providers=False,  # set True if you want provider labels
    label_drugs=False       # stays False as requested
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import networkx as nx

    print("[INFO] Visualizing mini graph...")

    # Split sets
    providers = [n for n, d in G_sub.nodes(data=True) if d.get("ntype") == "provider"]
    drugs     = [n for n, d in G_sub.nodes(data=True) if d.get("ntype") == "drug"]

    # ---- Colors: reserve green for drugs; exclude greens for specialties ----
    # A small qualitative palette that avoids green hues
    # (tableau + a few extras, but skipping greens)
    palette = [
        "#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ffbb78",
        "#c5b0d5", "#ff9896", "#c49c94", "#f7b6d2", "#c7c7c7",
        "#dbdb8d", "#9edae5", "#aec7e8"  # none of these are the same green as drugs
    ]
    drug_color = "#2ca02c"  # reserved for drugs

    specialties = sorted({G_sub.nodes[p].get("specialty", "Unknown") for p in providers})
    # Map specialties to colors, cycling if necessary
    spec_colors = {s: palette[i % len(palette)] for i, s in enumerate(specialties)}

    # ---- Sizes: provider node size by exposures-weighted degree; drugs fixed ----
    prov_strength = {p: G_sub.degree(p, weight="weight") for p in providers}
    pvals = np.array(list(prov_strength.values())) if providers else np.array([1])
    pmin, pmax = pvals.min(), pvals.max()
    def scale_size(val, min_px=300, max_px=1200):
        if pmax == pmin:
            return (min_px + max_px) / 2
        return min_px + (max_px - min_px) * ((val - pmin) / (pmax - pmin))
    prov_sizes = {p: scale_size(prov_strength[p]) for p in providers}
    drug_size = 900

    # ---- Edge colors/widths with percentile normalization (avoid washed-out whites) ----
    weights = np.array([d["weight"] for _, _, d in G_sub.edges(data=True)])
    if len(weights):
        vmin, vmax = np.percentile(weights, [5, 95])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        w_min, w_max = weights.min(), weights.max()
    else:
        norm = plt.Normalize(vmin=0, vmax=1)
        w_min, w_max = 0, 1
    edge_colors = [plt.cm.Reds(norm(G_sub.edges[e]["weight"])) for e in G_sub.edges]
    edge_widths = list(0.5 + 5.5 * ((weights - w_min) / (w_max - w_min + 1e-9))) if len(weights) else []

    # ---- Layout: compact bipartite by default; optional spring ----
    if layout == "bipartite":
        # Position drugs in the center column; providers split left/right to reduce overlap
        # Use nx.bipartite_layout as a base, then nudge columns for compactness.
        top = providers  # "top" set is arbitrary; we’ll override x positions
        pos = nx.bipartite_layout(G_sub, nodes=top, align="vertical", scale=1)
        # Force x for providers/drugs to three columns (-1, 0, +1) for visual balance
        for n in providers:
            pos[n] = (-1.0, pos[n][1])
        for n in drugs:
            pos[n] = (0.0, pos[n][1])
        # Optionally mirror half of providers to +1 to reduce crowding when many providers
        if len(providers) > 20:
            right_half = set(sorted(providers)[::2])
            for n in right_half:
                pos[n] = (1.0, pos[n][1] * 0.9)  # slight vertical squeeze
    else:
        pos = nx.spring_layout(G_sub, seed=seed, k=None, iterations=300)

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(
        "Mini Provider–Drug Bipartite Graph (Sampled Real Data)\n"
        "Providers by specialty | Drugs in green | Edges weighted by exposures",
        fontsize=12, pad=12
    )

    # Edges
    nx.draw_networkx_edges(
        G_sub, pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.9,
        ax=ax
    )

    # Provider nodes by specialty
    for s in specialties:
        nodes_s = [p for p in providers if G_sub.nodes[p].get("specialty", "Unknown") == s]
        nx.draw_networkx_nodes(
            G_sub, pos,
            nodelist=nodes_s,
            node_color=spec_colors[s],
            node_size=[prov_sizes[p] for p in nodes_s],
            edgecolors="white",
            linewidths=1.2,
            label=s,
            ax=ax
        )

    # Drug nodes (no labels)
    nx.draw_networkx_nodes(
        G_sub, pos,
        nodelist=drugs,
        node_color=drug_color,
        node_size=drug_size,
        edgecolors="black",
        linewidths=1.2,
        label="Drugs",
        ax=ax
    )

    # Labels (providers optional; drugs off)
    if label_providers:
        prov_labels = {p: p for p in providers}
        nx.draw_networkx_labels(G_sub, pos, labels=prov_labels, font_size=8, font_weight="bold")
    if label_drugs:
        drug_labels = {d: d for d in drugs}
        nx.draw_networkx_labels(G_sub, pos, labels=drug_labels, font_size=8, font_weight="bold")

    # Legend + colorbar
    handles = [
        Line2D([0], [0], marker='o', color='w', label=s,
               markerfacecolor=spec_colors[s], markeredgecolor='white', markersize=9)
        for s in specialties
    ] + [
        Line2D([0], [0], marker='o', color='w', label='Drugs',
               markerfacecolor=drug_color, markeredgecolor='black', markersize=9)
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="Node Types")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
    sm.set_array([])
    cax = inset_axes(ax, width="2%", height="40%", loc="lower left",
                     bbox_to_anchor=(1.02, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Edge weight (n_exposures)", rotation=90)

    # ---- Reduce whitespace: set bounds tightly around nodes ----
    xs, ys = zip(*[pos[n] for n in G_sub.nodes()])
    pad_x = 0.15 * (max(xs) - min(xs) if max(xs) > min(xs) else 1.0)
    pad_y = 0.15 * (max(ys) - min(ys) if max(ys) > min(ys) else 1.0)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved mini graph visualization → {save_path}")
    plt.show()

# After pickle.dump(G_sub, f)
visualize_mini_graph(G_sub, save_path=OUTPUT_DIR / "mini_provider_drug_graph.png")
# -------------------------------------------------
# 7) Save full graph for later analysis
# -------------------------------------------------
full_graph_path = HERE / "data" / "provider_drug_graph.pkl"
with open(full_graph_path, "wb") as f:
    pickle.dump(G, f)
print(f"[INFO] Saved full provider–drug graph → {full_graph_path}")

