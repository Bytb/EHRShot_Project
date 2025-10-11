# toy_provider_drug_graph.py
# Synthetic bipartite graph for visualization (Project 1 Network Analysis)
# Spec:
# - 3 specialties (S1,S2,S3) × 8 providers each = 24 providers
# - 4 drugs (D1–D4)
# - D1 & D2: cross-specialty popular; D3 & D4: specialty-leaning (D3→S1, D4→S3)
# - One overprescriber in S2 with very high weights to D1, D2, and one leaning drug
# - Sparse: each provider connects to 2–3 drugs
# - Edge weights clipped at 300; edges colored by continuous red gradient; thickness ∝ weight
# - Providers colored by specialty; drugs green; provider node size ∝ weighted degree
# - Spring layout (seed=42), central drugs + three provider lobes
# - Legend inset + edge-weight colorbar; black halo around overprescriber
# - Output: only plt.show() (no saving)

import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ------------------------
# Reproducibility
# ------------------------
random.seed(42)
np.random.seed(42)

# ------------------------
# Parameters
# ------------------------
SPECIALTIES = ["S1", "S2", "S3"]
PROVIDERS_PER_SPEC = 8
DRUGS = ["D1", "D2", "D3", "D4"]
OVERPRESCRIBER = "P_S2_4"      # highlight node
PPR_ALPHA = 0.70               # for later when you run PPR on this graph

SPEC_COLORS = {
    "S1": "#4169E1",  # royal blue
    "S2": "#8A2BE2",  # blueviolet
    "S3": "#FF8C00",  # dark orange
}
DRUG_COLOR = "#2ca02c"         # green

# Specialty-leaning mapping
LEAN_MAP = {"D3": "S1", "D4": "S3"}  # D1 & D2 are cross-specialty

# ------------------------
# Helpers
# ------------------------
def sample_weight(kind="normal"):
    """Sample realistic edge weights, clipped to [5, 300]."""
    if kind == "high":
        w = int(np.random.normal(loc=230, scale=40))
    elif kind == "mid":
        w = int(np.random.normal(loc=140, scale=35))
    else:
        w = int(np.random.normal(loc=70, scale=25))
    return int(np.clip(w, 5, 300))

def build_graph():
    """Construct the synthetic bipartite provider–drug graph per spec."""
    G = nx.Graph()

    # Add providers
    providers = []
    for s in SPECIALTIES:
        for i in range(1, PROVIDERS_PER_SPEC + 1):
            pid = f"P_{s}_{i}"
            providers.append(pid)
            G.add_node(pid, ntype="provider", specialty=s)

    # Add drugs
    for d in DRUGS:
        G.add_node(d, ntype="drug")

    # Connect providers sparsely (2–3 drugs each)
    for p in providers:
        spec = G.nodes[p]["specialty"]

        if p == OVERPRESCRIBER:
            # Strong links to D1, D2 (cross-specialty)
            for d in ["D1", "D2"]:
                G.add_edge(p, d, weight=sample_weight("high"))
            # Modest link to one leaning drug (pick D3 if S2 for contrast)
            dlean = "D3" if spec == "S2" else "D4"
            G.add_edge(p, dlean, weight=sample_weight("mid"))
            continue

        # Base probabilities favoring D1/D2 globally
        probs = {"D1": 0.35, "D2": 0.35, "D3": 0.15, "D4": 0.15}
        # Bias D3 or D4 if it matches provider specialty; slightly downweight otherwise
        for d in ["D3", "D4"]:
            if LEAN_MAP[d] == spec:
                probs[d] += 0.10
            else:
                probs[d] -= 0.05

        # Normalize
        total = sum(probs.values())
        for k in probs:
            probs[k] = max(0.0, probs[k] / total)

        # Choose 2–3 drugs
        k_edges = np.random.choice([2, 3], p=[0.6, 0.4])
        chosen = np.random.choice(DRUGS, size=k_edges, replace=False,
                                  p=[probs[d] for d in DRUGS])

        for d in chosen:
            if d in ["D1", "D2"]:
                w = sample_weight("mid" if np.random.rand() < 0.5 else "normal")
            else:
                w = sample_weight("mid" if LEAN_MAP.get(d) == spec else "normal")
            G.add_edge(p, d, weight=w)

    return G

def compute_layout(G):
    """Seed positions so drugs start central and specialties form lobes; then spring layout."""
    pos_init = {}
    # Central drugs
    pos_init["D1"] = (0.00, 0.10)
    pos_init["D2"] = (0.10, -0.05)
    pos_init["D3"] = (-0.10, -0.05)
    pos_init["D4"] = (0.00, -0.15)

    # Lobe centers
    lobe_centers = {"S1": (-1.0, 1.0), "S2": (1.0, 1.0), "S3": (0.0, -1.2)}

    providers = [n for n, d in G.nodes(data=True) if d.get("ntype") == "provider"]
    for p in providers:
        s = G.nodes[p]["specialty"]
        cx, cy = lobe_centers[s]
        jitter = np.random.normal(scale=0.20, size=2)
        pos_init[p] = (cx + jitter[0], cy + jitter[1])

    pos = nx.spring_layout(G, pos=pos_init, seed=42, k=None, iterations=200)
    return pos

def draw_graph(G, pos):
    """Render graph with aesthetics (no saving; just plt.show())."""
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(
        "Toy Provider–Drug Bipartite Graph (Synthetic)\n"
        "D1/D2 cross-specialty, D3→S1, D4→S3; α(PPR)=0.70",
        fontsize=12, pad=12
    )

    # Edge styles
    weights = np.array([G.edges[e]["weight"] for e in G.edges])
    w_min, w_max = weights.min(), weights.max()
    norm = plt.Normalize(vmin=w_min, vmax=w_max)
    cmap = plt.cm.Reds
    edge_colors = [cmap(norm(G.edges[e]["weight"])) for e in G.edges]

    # Thickness ~ 0.5–6 px
    w_norm = (weights - w_min) / (w_max - w_min + 1e-9)
    edge_widths = list(0.5 + 5.5 * w_norm)

    # Node sizes
    providers = [n for n, d in G.nodes(data=True) if d.get("ntype") == "provider"]
    drugs = [n for n, d in G.nodes(data=True) if d.get("ntype") == "drug"]

    prov_strength = {p: sum(G.edges[p, d]["weight"] for d in G.neighbors(p)) for p in providers}
    pvals = np.array(list(prov_strength.values())) if providers else np.array([1])
    pmin, pmax = pvals.min(), pvals.max()

    def scale_size(val, min_px=300, max_px=1200):
        if pmax == pmin:
            return (min_px + max_px) / 2
        return min_px + (max_px - min_px) * ((val - pmin) / (pmax - pmin))

    prov_sizes = {p: scale_size(prov_strength[p]) for p in providers}
    drug_size = 900

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=list(G.edges()),
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.9,
        ax=ax
    )

    # Draw provider nodes by specialty
    for s in SPECIALTIES:
        nodes_s = [p for p in providers if G.nodes[p]["specialty"] == s]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes_s,
            node_color=SPEC_COLORS[s],
            node_size=[prov_sizes[p] for p in nodes_s],
            edgecolors="white",
            linewidths=1.2,
            ax=ax
        )

    # Draw drug nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=drugs,
        node_color=DRUG_COLOR,
        node_size=drug_size,
        edgecolors="black",
        linewidths=1.2,
        ax=ax
    )

    # Minimal labels: drugs + overprescriber
    labels = {d: d for d in drugs}
    labels[OVERPRESCRIBER] = "O.P."
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold")

    # Halo around overprescriber
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[OVERPRESCRIBER],
        node_color="none",
        node_size=prov_sizes[OVERPRESCRIBER] + 350,
        edgecolors="black",
        linewidths=2.5,
        ax=ax
    )

    # Legend (nodes)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='S1 (providers)',
               markerfacecolor=SPEC_COLORS['S1'], markeredgecolor='white', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='S2 (providers)',
               markerfacecolor=SPEC_COLORS['S2'], markeredgecolor='white', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='S3 (providers)',
               markerfacecolor=SPEC_COLORS['S3'], markeredgecolor='white', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Drugs',
               markerfacecolor="#2ca02c", markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Overprescriber halo',
               markerfacecolor='none', markeredgecolor='black', markersize=10, linewidth=2),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="Node Types")

    # Edge-weight colorbar (inset)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = inset_axes(ax, width="2%", height="40%", loc="lower left",
                     bbox_to_anchor=(1.02, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Edge weight", rotation=90)

    ax.axis("off")
    plt.tight_layout()

    from pathlib import Path
    OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Results" / "toy"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "toy_synthetic_graph.png", dpi=300)

    plt.show()

def main():
    G = build_graph()
    pos = compute_layout(G)
    draw_graph(G, pos)

if __name__ == "__main__":
    main()
