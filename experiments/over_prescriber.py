"""
ego_overpresriber.py
--------------------
Build a 2-hop ego graph centered on a specified over-prescribing provider and
render a compact bipartite visualization. Also prints and annotates the top-3
drugs (by exposures) for easy Athena lookup.

Inputs:
    data/provider_drug_graph.pkl  (from build_real_provider_drug_graph.py)

Outputs:
    Results/images/overpresriber/ego_overpresriber_<provider_id>.png
"""

from pathlib import Path
import os
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

# =====================
# ---- CONFIG AREA ----
# =====================
HERE = Path(__file__).resolve().parents[1]
GRAPH_PATH = HERE / "data" / "provider_drug_graph.pkl"
OUT_DIR = HERE / "Results" / "images" / "overpresriber"  # per user path/spelling
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOCAL_PROVIDER = "6822175"  # center provider (string by default)
FIG_DPI = 300
SEED = 42

# Styling
FOCAL_COLOR = "#00C853"   # bright green
FOCAL_EDGE_COLOR = "#333333"
NONFOCAL_EDGE_COLOR = "#B0B0B0"
DRUG_COLOR = "#000000"    # black nodes for drugs
BG_COLOR = "white"

# Ring radii (tight)
R_DRUGS = 0.8
R_COPROV = 1.45

# =====================
# ---- HELPERS     ----
# =====================
def ensure_graph_ok(G: nx.Graph):
    n, data = next(iter(G.nodes(data=True)))
    if "ntype" not in data:
        raise RuntimeError("Graph nodes must have attribute 'ntype' in {'provider','drug'}.")
    return

def neighbors_by_type(G: nx.Graph, node, ntype: str):
    return [nbr for nbr in G.neighbors(node) if G.nodes[nbr].get("ntype") == ntype]

def fade_alpha(n_coprov: int) -> float:
    """Adaptive fade for non-focal nodes/edges to reduce clutter."""
    if n_coprov <= 50:   return 0.60
    if n_coprov <= 200:  return 0.35
    if n_coprov <= 800:  return 0.20
    return 0.12

def specialty_palette(exclude_hex={"#00C853", "#000000"}):
    """Palette for specialties, avoiding focal green and black."""
    base = [
        "#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ffbb78",
        "#c5b0d5", "#ff9896", "#c49c94", "#f7b6d2", "#c7c7c7",
        "#dbdb8d", "#9edae5", "#aec7e8"
    ]
    return [c for c in base if c.lower() not in {e.lower() for e in exclude_hex}]

def triple_ring_layout(focal, drugs, coproviders, r1=R_DRUGS, r2=R_COPROV, seed=SEED):
    """Place focal at (0,0), drugs on radius r1, co-providers on radius r2."""
    rng = np.random.default_rng(seed)
    pos = {focal: (0.0, 0.0)}

    k_d = max(1, len(drugs))
    k_p = max(1, len(coproviders))

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
    ensure_graph_ok(G)

    # Normalize focal provider
    focal = FOCAL_PROVIDER
    if focal not in G:
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

    # 2-hop ego neighborhood
    drugs = neighbors_by_type(G, focal, "drug")
    coproviders = set()
    for d in drugs:
        coproviders.update(neighbors_by_type(G, d, "provider"))
    coproviders.discard(focal)
    coproviders = sorted(coproviders, key=lambda n: str(n))

    # Induced subgraph
    ego_nodes = {focal} | set(drugs) | set(coproviders)
    H = G.subgraph(ego_nodes).copy()

    # Drug display labels (1..K)
    drug_order = sorted(drugs, key=lambda x: str(x))
    drug_label_map = {d: i+1 for i, d in enumerate(drug_order)}

    # Weighted widths for focal edges (by exposures)
    focal_edges = []
    focal_weights = []
    for d in drug_order:
        if H.has_edge(focal, d):
            focal_edges.append((focal, d))
            focal_weights.append(float(H[focal][d].get("weight", 1.0)))
    w_min, w_max = (min(focal_weights), max(focal_weights)) if focal_weights else (1.0, 1.0)
    def scale_width(w, min_w=1.6, max_w=7.0):
        if w_max == w_min: return (min_w + max_w) / 2.0
        return min_w + (max_w - min_w) * ((w - w_min) / (w_max - w_min))
    focal_widths = { (focal, d): scale_width(float(H[focal][d].get("weight", 1.0)))
                     for d in drug_order if H.has_edge(focal, d) }

    # ---- TOP 3 DRUGS by exposures (console + left legend data) ----
    top_drugs_info = []
    for d in drug_order:
        if H.has_edge(focal, d):
            w = float(H[focal][d].get("weight", 1.0))
            npat = H[focal][d].get("n_patients", None)
            top_drugs_info.append((d, drug_label_map[d], w, npat))
    top_drugs_info.sort(key=lambda t: t[2], reverse=True)
    top3 = top_drugs_info[:3]

    print("\n[INFO] Top 3 drugs for focal provider (by exposures):")
    print(f"{'rank':<5}{'display#':<10}{'drug_id':<20}{'exposures':<12}{'n_patients':<12}")
    for i, (drug_id, disp_num, w, npat) in enumerate(top3, start=1):
        npat_str = "NA" if npat is None else str(npat)
        print(f"{i:<5}{disp_num:<10}{str(drug_id):<20}{w:<12.0f}{npat_str:<12}")

    # Specialty colors for other providers
    palette = specialty_palette()
    spec_to_color = {}
    i = 0
    for p in coproviders:
        spec = H.nodes[p].get("specialty", "Unknown")
        if spec not in spec_to_color:
            spec_to_color[spec] = palette[i % len(palette)]
            i += 1

    # Layout (tight rings)
    pos = triple_ring_layout(focal, drug_order, coproviders, r1=R_DRUGS, r2=R_COPROV, seed=SEED)

    # Fading for non-focal elements
    alpha_fade = fade_alpha(len(coproviders))

    # ---- Draw ----
    plt.figure(figsize=(10.5, 8.5), facecolor=BG_COLOR)
    ax = plt.gca()
    ax.set_facecolor(BG_COLOR)

    # Non-focal edges (thin + faded)
    non_focal_edges = [(u, v) for u, v in H.edges() if not (u == focal or v == focal)]
    nx.draw_networkx_edges(
        H, pos,
        edgelist=non_focal_edges,
        width=0.6,
        edge_color=NONFOCAL_EDGE_COLOR,
        alpha=alpha_fade
    )

    # Focal edges (weighted widths)
    nx.draw_networkx_edges(
        H, pos,
        edgelist=list(focal_edges),
        width=[focal_widths.get((u, v), 2.5) for (u, v) in focal_edges],
        edge_color=FOCAL_EDGE_COLOR,
        alpha=0.9
    )

    # Drugs (black, labeled 1..K)
    nx.draw_networkx_nodes(
        H, pos,
        nodelist=drug_order,
        node_color=DRUG_COLOR,
        node_size=560,
        edgecolors="white",
        linewidths=1.2,
        alpha=max(0.5, alpha_fade)
    )
    drug_labels = {d: str(drug_label_map[d]) for d in drug_order}
    nx.draw_networkx_labels(
        H, pos, labels=drug_labels,
        font_color="white", font_size=10, font_weight="bold"
    )

    # Co-providers by specialty (faded)
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
            alpha=alpha_fade
        )

    # Focal provider (bright green with black outline) — NO LABEL ON GRAPH
    nx.draw_networkx_nodes(
        H, pos,
        nodelist=[focal],
        node_color=FOCAL_COLOR,
        node_size=900,
        edgecolors="black",
        linewidths=2.0,
        alpha=1.0
    )

    # --- Legend (right): Prescriber 0 + Drug + Top-7 specialties by frequency (with %) ---
    spec_series = pd.Series([H.nodes[p].get("specialty", "Unknown") for p in coproviders])
    spec_counts = spec_series.value_counts()
    total_coprov = int(spec_counts.sum())
    top_specs = spec_counts.head(7)

    legend_handles_right = [
        Line2D([0],[0], marker='o', color='w', label='Prescriber 0',
               markerfacecolor=FOCAL_COLOR, markeredgecolor='black', markersize=10),
        Line2D([0],[0], marker='o', color='w', label='Drug',
               markerfacecolor=DRUG_COLOR, markeredgecolor='white', markersize=9),
    ]
    for spec, cnt in top_specs.items():
        pct = (cnt / max(1, total_coprov)) * 100.0
        color = spec_to_color.get(spec, "#aaaaaa")
        legend_handles_right.append(
            Line2D([0],[0], marker='o', color='w',
                   label=f"{spec} — {pct:.1f}%",
                   markerfacecolor=color, markeredgecolor='white', markersize=8)
        )
    leg_right = ax.legend(
        handles=legend_handles_right,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=False,
        title="Node types / top specialties",
        borderaxespad=0.0
    )
    ax.add_artist(leg_right)  # keep when adding second legend

    # --- Legend (left): Top 3 drugs (# — concept_id) for manual naming later ---
    drug_names = {
        "40232756": "Oxycodone Hydrochloride (5 MG)", #oxycodone hydrochloride 5 MG Oral Tablet
        "1718702": "Abuse Deterent Oxycodone Hydrochloride (10 MG)", #Abuse-Deterrent 12 HR oxycodone hydrochloride 10 MG Extended Release Oral Tablet
        "40232707": "Oxycodone Hydrochloride (10 MG)" #oxycodone hydrochloride 10 MG Extended Release Oral Tablet
    }

    legend_handles_left = []
    for i, (drug_id, disp_num, w, npat) in enumerate(top3, start=1):
        legend_handles_left.append(
            Line2D([0],[0], marker='o', color='w',
                   label=f"{disp_num} — {drug_names[str(drug_id)]}",
                   markerfacecolor=DRUG_COLOR, markeredgecolor='white', markersize=8)
        )
    ax.legend(
        handles=legend_handles_left,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=False,
        title="Top 3 drugs",
        borderaxespad=0.0
    )

    # Title and tidy axes
    ax.set_title("2-Hop Ego Graph of Over-Prescriber\n"
                 "Center: focal provider • Middle ring: drugs (numbered) • Outer ring: co-providers",
                 pad=10)
    ax.axis("off")

    # Tight bounds / margins
    xs, ys = zip(*[pos[n] for n in H.nodes()])
    pad_x = 0.06 * (max(xs) - min(xs) if max(xs) > min(xs) else 1.0)
    pad_y = 0.06 * (max(ys) - min(ys) if max(ys) > min(ys) else 1.0)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect('equal')
    ax.margins(x=0.01, y=0.01)

    # Save
    out_path = OUT_DIR / f"ego_overpresriber_{str(FOCAL_PROVIDER)}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"[INFO] Saved ego graph → {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
