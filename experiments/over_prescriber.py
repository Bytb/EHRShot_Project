"""
ego_overpresriber.py
--------------------
Build a 2-hop ego graph centered on the specified over-prescribing provider and
render a clear bipartite visualization.

Inputs:
    data/provider_drug_graph.pkl  (built by build_real_provider_drug_graph.py)

Output:
    Results/images/overpresriber/ego_overpresriber_<provider_id>.png
"""

from pathlib import Path
import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =====================
# ---- CONFIG AREA ----
# =====================
HERE = Path(__file__).resolve().parents[1]
GRAPH_PATH = HERE / "data" / "provider_drug_graph.pkl"
OUT_DIR = HERE / "Results" / "images" / "overpresriber"  # spelling per request
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOCAL_PROVIDER = "6822175"  # over-prescriber provider_id (string to match graph IDs)
FIG_DPI = 300
SEED = 42

# Styling
FOCAL_COLOR = "#00C853"  # bright green
FOCAL_EDGE_COLOR = "#333333"
PROVIDER_EDGE_COLOR = "#999999"
DRUG_COLOR = "#000000"   # black nodes for drugs
BG_COLOR = "white"

# =====================
# ---- HELPERS     ----
# =====================
def _assert_graph(G: nx.Graph):
    # Ensure node attributes exist
    sample = next(iter(G.nodes))
    if "ntype" not in G.nodes[sample]:
        raise RuntimeError("Graph nodes must have attribute 'ntype' in {'provider','drug'}.")
    # Ensure edges carry 'weight' (exposures)
    # Not strictly required, but the focal edge widths use it.
    # We'll tolerate missing weights by treating as 1.
    return

def _neighbors_by_type(G: nx.Graph, node, ntype: str):
    return [nbr for nbr in G.neighbors(node) if G.nodes[nbr].get("ntype") == ntype]

def _calc_fade_alpha(n_coprov: int) -> float:
    """
    Adaptive fade for non-focal elements:
      - <= 50 co-providers: 0.60 (clear)
      - 51..200:            0.35
      - 201..800:           0.20
      - >800:               0.12
    """
    if n_coprov <= 50:
        return 0.60
    if n_coprov <= 200:
        return 0.35
    if n_coprov <= 800:
        return 0.20
    return 0.12

def _specialty_palette(exclude_hex={"#00C853", "#000000"}):
    """
    Return a cycle of distinct colors for provider specialties,
    avoiding focal green and black.
    """
    base = [
        "#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ffbb78",
        "#c5b0d5", "#ff9896", "#c49c94", "#f7b6d2", "#c7c7c7",
        "#dbdb8d", "#9edae5", "#aec7e8"
    ]
    return [c for c in base if c.lower() not in {e.lower() for e in exclude_hex}]

def _triple_ring_layout(focal, drugs, coproviders, r1=1.0, r2=2.1, seed=SEED):
    """
    Place focal at (0,0), drugs on radius r1, co-providers on radius r2.
    Spread nodes uniformly around each ring.
    Returns a dict of positions for all nodes in the ego graph.
    """
    rng = np.random.default_rng(seed)
    pos = {focal: (0.0, 0.0)}

    k_d = max(1, len(drugs))
    k_p = max(1, len(coproviders))

    # Jittered uniform angles to avoid perfect overlap
    angles_d = np.linspace(0, 2*np.pi, num=k_d, endpoint=False) + rng.normal(0, 0.04, size=k_d)
    angles_p = np.linspace(0, 2*np.pi, num=k_p, endpoint=False) + rng.normal(0, 0.04, size=k_p)

    for a, d in zip(angles_d, drugs):
        pos[d] = (r1 * math.cos(a), r1 * math.sin(a))

    for a, p in zip(angles_p, coproviders):
        pos[p] = (r2 * math.cos(a), r2 * math.sin(a))

    return pos

# =====================
# ---- MAIN LOGIC   ----
# =====================
def main():
    # Load graph
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    _assert_graph(G)

    # Normalize focal provider type
    focal = FOCAL_PROVIDER
    if focal not in G:
        # Attempt int fallback if the graph uses int ids
        try:
            focal_int = int(FOCAL_PROVIDER)
            if focal_int in G:
                focal = focal_int
            else:
                raise KeyError
        except Exception:
            raise RuntimeError(f"Focal provider '{FOCAL_PROVIDER}' not found in graph.")

    if G.nodes[focal].get("ntype") != "provider":
        raise RuntimeError(f"Node {focal} exists but is not labeled ntype='provider'.")

    # 2-hop ego:
    # level-1 = drugs of focal
    drugs = _neighbors_by_type(G, focal, "drug")
    # level-2 = all providers attached to any of those drugs (excluding focal)
    coproviders = set()
    for d in drugs:
        coproviders.update(_neighbors_by_type(G, d, "provider"))
    coproviders.discard(focal)
    coproviders = sorted(coproviders, key=lambda n: str(n))

    # Build induced subgraph
    ego_nodes = {focal} | set(drugs) | set(coproviders)
    H = G.subgraph(ego_nodes).copy()

    # Map drugs to simple labels 1..K for display
    drug_order = sorted(drugs, key=lambda x: str(x))
    drug_label_map = {d: i+1 for i, d in enumerate(drug_order)}

    # Compute edge widths for focal edges (by exposures weight)
    focal_edges = []
    focal_weights = []
    for d in drugs:
        if H.has_edge(focal, d):
            w = float(H[focal][d].get("weight", 1.0))
            focal_edges.append((focal, d))
            focal_weights.append(w)
    if focal_weights:
        w_min, w_max = min(focal_weights), max(focal_weights)
    else:
        w_min, w_max = 1.0, 1.0

    def scale_width(w, min_w=1.2, max_w=6.5):
        if w_max == w_min:
            return (min_w + max_w) / 2.0
        return min_w + (max_w - min_w) * ((w - w_min) / (w_max - w_min))

    focal_widths = { (focal, d): scale_width(float(H[focal][d].get("weight", 1.0))) for d in drugs if H.has_edge(focal, d) }

    # Specialty colors for other providers
    palette = _specialty_palette()
    spec_to_color = {}
    i = 0
    for p in coproviders:
        spec = H.nodes[p].get("specialty", "Unknown")
        if spec not in spec_to_color:
            spec_to_color[spec] = palette[i % len(palette)]
            i += 1

    # Layout (three rings)
    pos = _triple_ring_layout(focal, drug_order, coproviders, r1=1.0, r2=2.1, seed=SEED)

    # Fading alpha for non-focal elements
    fade_alpha = _calc_fade_alpha(len(coproviders))

    # ---- Draw ----
    plt.figure(figsize=(11, 9), facecolor=BG_COLOR)
    ax = plt.gca()
    ax.set_facecolor(BG_COLOR)

    # 1) Edges: non-focal first (thin + faded)
    non_focal_edges = []
    for u, v, d in H.edges(data=True):
        if (u == focal or v == focal):
            continue
        non_focal_edges.append((u, v))
    nx.draw_networkx_edges(
        H, pos,
        edgelist=non_focal_edges,
        width=0.6,
        edge_color=PROVIDER_EDGE_COLOR,
        alpha=fade_alpha
    )

    # 2) Edges: focal edges with weight-based width (darker)
    nx.draw_networkx_edges(
        H, pos,
        edgelist=list(focal_edges),
        width=[focal_widths.get((u, v), 2.5) for (u, v) in focal_edges],
        edge_color=FOCAL_EDGE_COLOR,
        alpha=0.9
    )

    # 3) Nodes: drugs (black, labeled 1..K)
    nx.draw_networkx_nodes(
        H, pos,
        nodelist=drug_order,
        node_color=DRUG_COLOR,
        node_size=550,
        edgecolors="white",
        linewidths=1.2,
        alpha=max(0.5, fade_alpha)  # keep drugs readable
    )
    # drug labels (numbers only)
    drug_labels = {d: str(drug_label_map[d]) for d in drug_order}
    nx.draw_networkx_labels(
        H, pos, labels=drug_labels,
        font_color="white", font_size=9, font_weight="bold"
    )

    # 4) Nodes: co-providers by specialty (faded)
    for spec, color in spec_to_color.items():
        nodes_s = [p for p in coproviders if H.nodes[p].get("specialty", "Unknown") == spec]
        if not nodes_s:
            continue
        nx.draw_networkx_nodes(
            H, pos,
            nodelist=nodes_s,
            node_color=color,
            node_size=320,
            edgecolors="white",
            linewidths=1.0,
            alpha=fade_alpha
        )

    # 5) Node: focal provider (bright green with black outline, labeled)
    nx.draw_networkx_nodes(
        H, pos,
        nodelist=[focal],
        node_color=FOCAL_COLOR,
        node_size=900,
        edgecolors="black",
        linewidths=2.0,
        alpha=1.0
    )
    nx.draw_networkx_labels(
        H, pos,
        labels={focal: str(focal)},
        font_color="black",
        font_size=10,
        font_weight="bold"
    )

    # 6) Legend (specialties) + title + tidy axes
    # Build proxy handles for specialties
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', label='Provider 0',
               markerfacecolor=FOCAL_COLOR, markeredgecolor='black', markersize=10)
    ]
    # Add a single "Drug" legend item
    handles.append(Line2D([0], [0], marker='o', color='w', label='Drug',
                          markerfacecolor=DRUG_COLOR, markeredgecolor='white', markersize=9))
    # Add specialties
    for spec, color in spec_to_color.items():
        handles.append(Line2D([0], [0], marker='o', color='w', label=spec,
                              markerfacecolor=color, markeredgecolor='white', markersize=8))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="Node types / specialties")

    ax.set_title("2-Hop Ego Graph of Over-Prescriber\n"
                 "Center: focal provider • Middle ring: drugs (numbered) • Outer ring: co-providers",
                 pad=12)
    ax.axis("off")

    # Tighten view bounds
    xs, ys = zip(*[pos[n] for n in H.nodes()])
    pad_x = 0.25 * (max(xs) - min(xs) if max(xs) > min(xs) else 1.0)
    pad_y = 0.25 * (max(ys) - min(ys) if max(ys) > min(ys) else 1.0)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    # Save
    out_path = OUT_DIR / f"ego_overpresriber_{str(FOCAL_PROVIDER)}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"[INFO] Saved ego graph → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
