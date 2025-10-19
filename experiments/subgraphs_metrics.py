# subgraphs_metrics.py — Top-5 specialties (by provider count) using CSV mapping (no mapping saved)
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, List
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
HERE = Path(__file__).resolve().parents[1]
GRAPH_PATH = HERE / "data" / "provider_drug_graph.pkl"
PROVIDERS_CSV = HERE / "data" / "providers.csv"

OUT_DIR = HERE / "Results" / "subgraphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 5
DPI = 200

# Drawing — per spec: drugs slightly larger & green; providers black
FIGSIZE_ORIGINAL = (16, 9)
FIGSIZE_SUBGRAPH = (14, 8)
EDGE_ALPHA = 0.08
EDGE_WIDTH = 0.5
PROVIDER_NODE_SIZE = 20
DRUG_NODE_SIZE = 45
PROVIDER_COLOR = "black"
DRUG_COLOR = "green"

# ============================================================
# HELPERS
# ============================================================

def _sanitize_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")
    return s or "subgraph"

def ensure_node_types(G: nx.Graph, providers_csv_path: Path = PROVIDERS_CSV):
    """
    Ensure nodes have type in {'provider','drug'}.
    Priority:
      (a) existing 'type'
      (b) existing 'ntype' -> map to 'type'
      (c) 'bipartite' sides if present (0/1 or left/right), tie-break with CSV overlap
      (d) CSV overlap only
    """
    # (a) already typed?
    if any(d.get("type") in {"provider", "drug"} for _, d in G.nodes(data=True)):
        return

    # (b) migrate 'ntype' -> 'type' if present
    if any(d.get("ntype") in {"provider", "drug"} for _, d in G.nodes(data=True)):
        mapping = {}
        for n, d in G.nodes(data=True):
            ntype = d.get("ntype")
            if ntype in {"provider", "drug"}:
                mapping[n] = ntype
        if mapping:
            nx.set_node_attributes(G, mapping, name="type")
            return

    # (c) bipartite attribute?
    left = [n for n, d in G.nodes(data=True) if d.get("bipartite") in (0, "0", "left")]
    right = [n for n, d in G.nodes(data=True) if d.get("bipartite") in (1, "1", "right")]
    if (left or right) and providers_csv_path.exists():
        dfp = pd.read_csv(providers_csv_path, usecols=["provider_id"])
        ids_str = set(dfp["provider_id"].astype(str))
        left_hits = sum(str(n) in ids_str for n in left)
        right_hits = sum(str(n) in ids_str for n in right)
        providers = set(left) if left_hits >= right_hits else set(right)
        drugs = (set(left) | set(right)) - providers
        nx.set_node_attributes(G, {n: "provider" for n in providers}, name="type")
        nx.set_node_attributes(G, {n: "drug" for n in drugs}, name="type")
        return

    # (d) CSV-only overlap
    if providers_csv_path.exists():
        dfp = pd.read_csv(providers_csv_path, usecols=["provider_id"])
        ids_str = set(dfp["provider_id"].astype(str))
        providers = {n for n in G.nodes if str(n) in ids_str}
        if providers:
            drugs = set(G.nodes) - providers
            nx.set_node_attributes(G, {n: "provider" for n in providers}, name="type")
            nx.set_node_attributes(G, {n: "drug" for n in drugs}, name="type")
            return

    raise RuntimeError(
        "Unable to infer node types. Provide 'type'/'ntype' on nodes, 'bipartite' sides, "
        "or a providers.csv containing provider_id values that overlap graph nodes."
    )

def _providers(G: nx.Graph) -> List:
    return [n for n, d in G.nodes(data=True) if d.get("type") == "provider"]

def _drugs(G: nx.Graph) -> List:
    return [n for n, d in G.nodes(data=True) if d.get("type") == "drug"]

def build_provider_specialty_mapping_from_csv(
    provider_nodes: List, providers_csv_path: Path = PROVIDERS_CSV
) -> Dict:
    """
    Mirror the PPR logic: read providers.csv, prefer specialty_source_value,
    drop unknown/empty/0, then keep only providers present in the graph.
    No file is saved; mapping is returned directly.
    """
    if not providers_csv_path.exists():
        raise FileNotFoundError(f"providers.csv not found at: {providers_csv_path}")

    dfp = pd.read_csv(
        providers_csv_path,
        usecols=["provider_id", "specialty_source_value", "specialty_concept_id", "specialty_best"]
    )
    dfp["provider_id_str"] = dfp["provider_id"].astype(str)

    # Prefer specialty_source_value
    def pick_spec(row):
        ssv = str(row.get("specialty_source_value") or "").strip()
        return ssv

    dfp["specialty"] = dfp.apply(pick_spec, axis=1)

    # Exclude unknowns / empties / "0" / "UNKNOWN"
    bad = (
        dfp["specialty"].isna()
        | (dfp["specialty"].str.strip() == "")
        | (dfp["specialty"].str.strip() == "0")
        | (dfp["specialty"].str.strip().str.upper() == "UNKNOWN")
    )
    dfp = dfp[~bad].copy()

    # Intersect with graph provider nodes
    providers_str = {str(p) for p in provider_nodes}
    dfp = dfp[dfp["provider_id_str"].isin(providers_str)].copy()

    # Build mapping keyed by the actual node objects (preserve original dtype as in G)
    str_to_node = {str(p): p for p in provider_nodes}
    mapping = {str_to_node[s]: spec for s, spec in zip(dfp["provider_id_str"], dfp["specialty"])}
    return mapping

def top_specialties_by_provider_count(mapping: Dict, top_n: int = TOP_N) -> List[str]:
    """
    Rank specialties by number of unique providers in the CSV-derived mapping.
    Deterministic alphabetical tie-break on ties.
    """
    # mapping: {provider_node -> specialty}
    vc = (
        pd.Series(list(mapping.values()), dtype=str)
          .value_counts(dropna=False)
          .rename_axis("specialty")
          .reset_index(name="n_providers")
    )
    # Exclude any accidental empties that might sneak in (shouldn't, but safe)
    vc["specialty"] = vc["specialty"].astype(str).str.strip()
    vc = vc[vc["specialty"].ne("") & vc["specialty"].ne("0") & vc["specialty"].str.upper().ne("UNKNOWN")]

    vc = vc.sort_values(["n_providers", "specialty"], ascending=[False, True]).reset_index(drop=True)
    return vc["specialty"].head(top_n).tolist()


def induced_specialty_subgraph(
    G: nx.Graph, provider_specialty: Dict, specialty: str
) -> Tuple[nx.Graph, List, List]:
    """Build induced bipartite subgraph for `specialty` using the CSV-based mapping."""
    P_S = [p for p, spec in provider_specialty.items() if spec == specialty]
    # Collect drugs adjacent to those providers
    D_S = set()
    for p in P_S:
        for nbr in G.neighbors(p):
            if G.nodes[nbr].get("type") == "drug":
                D_S.add(nbr)
    H = G.subgraph(set(P_S) | D_S).copy()
    return H, P_S, list(D_S)

def bipartite_density(n_edges: int, n_prov: int, n_drug: int) -> float:
    if n_prov == 0 or n_drug == 0:
        return 0.0
    return n_edges / float(n_prov * n_drug)

def largest_component_fraction(Gsub: nx.Graph) -> float:
    n = Gsub.number_of_nodes()
    if n == 0:
        return 0.0
    comps = list(nx.connected_components(Gsub))
    if not comps:
        return 0.0
    lcc = max(comps, key=len)
    return len(lcc) / float(n)

def global_square_clustering_mean(Gsub: nx.Graph) -> float:
    """
    Mean square-based bipartite clustering over all nodes.
    Uses nx.bipartite.clustering for broad version compatibility.
    """
    if Gsub.number_of_nodes() == 0:
        return float("nan")
    cc_map = nx.bipartite.clustering(Gsub)  # dict: node -> coefficient
    if not cc_map:
        return float("nan")
    vals = list(cc_map.values())
    return float(np.mean(vals)) if vals else float("nan")

def _bipartite_layout_compat(G: nx.Graph, providers) -> Dict:
    """
    Return a bipartite layout across NetworkX versions.

    Try in order:
      1) nx.bipartite.layout.bipartite_layout(G, top_nodes=..., align=..., scale=...)
      2) nx.bipartite_layout(G, nodes=..., align=..., scale=...)
      3) nx.bipartite_layout(G, nodes)  (positional, no kwargs)
      4) Manual two-column layout as a final fallback
    """
    prov_set = set(providers)

    # (1) Modern path
    try:
        return nx.bipartite.layout.bipartite_layout(
            G, top_nodes=prov_set, align="vertical", scale=1.0
        )
    except Exception:
        pass

    # (2) Legacy with keyword 'nodes'
    try:
        return nx.bipartite_layout(
            G, nodes=prov_set, align="vertical", scale=1.0
        )
    except Exception:
        pass

    # (3) Legacy positional, maybe no align/scale support
    try:
        return nx.bipartite_layout(G, prov_set)
    except Exception:
        # (4) Manual layout fallback
        pos = {}
        providers_sorted = sorted(list(prov_set), key=str)
        drugs_sorted = sorted([n for n in G.nodes if n not in prov_set], key=str)
        nP = len(providers_sorted)
        nD = len(drugs_sorted)

        for i, p in enumerate(providers_sorted):
            y = 1.0 - (i + 1) / (nP + 1) if nP else 0.5
            pos[p] = (0.0, y)
        for j, d in enumerate(drugs_sorted):
            y = 1.0 - (j + 1) / (nD + 1) if nD else 0.5
            pos[d] = (1.0, y)
        return pos

def draw_bipartite(
    Gdraw: nx.Graph,
    providers: Iterable,
    drugs: Iterable,
    out_png: Path,
    out_svg: Path,
    title: str,
    figsize=(12, 7),
):
    providers = list(providers)
    drugs = list(drugs)
    # Layout with robust compatibility
    pos = _bipartite_layout_compat(Gdraw, providers)

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(Gdraw, pos, width=EDGE_WIDTH, alpha=EDGE_ALPHA)
    nx.draw_networkx_nodes(Gdraw, pos, nodelist=providers,
                           node_size=PROVIDER_NODE_SIZE, node_color=PROVIDER_COLOR)
    nx.draw_networkx_nodes(Gdraw, pos, nodelist=drugs,
                           node_size=DRUG_NODE_SIZE, node_color=DRUG_COLOR)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI)
    plt.savefig(out_svg)
    plt.close()

# ============================================================
# MAIN FLOW
# ============================================================

def run_subgraph_metrics(G: nx.Graph) -> Dict:
    # Ensure node typing; no specialty read from nodes here
    ensure_node_types(G)
    providers = _providers(G)
    drugs = _drugs(G)
    if not providers or not drugs:
        raise RuntimeError("Graph must contain nodes with type='provider' and type='drug' after inference.")

    # Build CSV-derived provider→specialty mapping (exclude unknowns/empties/0), intersected with providers on G
    provider_specialty = build_provider_specialty_mapping_from_csv(providers, PROVIDERS_CSV)
    if not provider_specialty:
        raise RuntimeError("No providers with valid specialties intersect the graph after CSV filtering.")

    # Select Top-N specialties by provider count (deterministic alphabetical tiebreak)
    top_specs = top_specialties_by_provider_count(provider_specialty, top_n=TOP_N)
    print("\n[Select] Top specialties by provider count (CSV-based, unknowns excluded):")
    for i, s in enumerate(top_specs, 1):
        print(f"  {i}. {s}")

    # Compute metrics per induced subgraph
    rows = []
    sizes = []  # for choosing which subgraph to draw
    for spec in top_specs:
        Gs, P_S, D_S = induced_specialty_subgraph(G, provider_specialty, spec)
        n_prov = len(P_S)
        n_drug = len(D_S)
        n_edges = Gs.number_of_edges()
        density = bipartite_density(n_edges, n_prov, n_drug)
        n_components = nx.number_connected_components(Gs) if Gs.number_of_nodes() > 0 else 0
        lcc_frac = largest_component_fraction(Gs)
        mean_sq_cc = global_square_clustering_mean(Gs)

        rows.append({
            "specialty": spec,
            "n_providers": n_prov,
            "n_drugs": n_drug,
            "n_edges": n_edges,
            "density": density,
            "mean_square_cc": mean_sq_cc,
            "n_components": n_components,
            "largest_comp_frac": lcc_frac,
        })
        sizes.append((spec, n_prov))

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(["n_providers", "n_drugs", "n_edges", "specialty"],
                               ascending=[False, False, False, True]).reset_index(drop=True)

    # Print table (rounded)
    print("\n[Metrics] Top-5 Specialty Subgraphs — density, global square clustering, components")
    with pd.option_context("display.max_columns", None):
        print(df_sorted.round(3).to_string(index=False))

    # Save CSV (in OUT_DIR root)
    csv_path = OUT_DIR / "top5_specialty_bipartite_metrics.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n[Save] CSV written: {csv_path}")

    # Draw full original graph
    title_full = f"Original Provider–Drug Graph (providers={len(providers)}, drugs={len(drugs)}, edges={G.number_of_edges()})"
    out_png_full = OUT_DIR / "original_graph.png"
    out_svg_full = OUT_DIR / "original_graph.svg"
    print(f"[Draw] Full graph → {out_png_full.name}, {out_svg_full.name}")
    draw_bipartite(Gdraw=G, providers=providers, drugs=drugs,
                   out_png=out_png_full, out_svg=out_svg_full,
                   title=title_full, figsize=FIGSIZE_ORIGINAL)

    # Draw one induced subgraph (largest by provider count among Top-5)
    if sizes:
        chosen_spec = sorted(sizes, key=lambda x: (-x[1], x[0]))[0][0]
        Gs, P_S, D_S = induced_specialty_subgraph(G, provider_specialty, chosen_spec)
        title_sub = f"{chosen_spec} Subgraph (providers={len(P_S)}, drugs={len(D_S)}, edges={Gs.number_of_edges()})"
        base = _sanitize_filename(chosen_spec)
        out_png_sub = OUT_DIR / f"subgraph_{base}.png"
        out_svg_sub = OUT_DIR / f"subgraph_{base}.svg"
        print(f"[Draw] Induced subgraph ({chosen_spec}) → {out_png_sub.name}, {out_svg_sub.name}")
        draw_bipartite(Gdraw=Gs, providers=P_S, drugs=D_S,
                       out_png=out_png_sub, out_svg=out_svg_sub,
                       title=title_sub, figsize=FIGSIZE_SUBGRAPH)

    return {
        "csv_metrics": str(csv_path),
        "original_png": str(out_png_full),
        "original_svg": str(out_svg_full),
        "top_specs": top_specs,
    }

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("[Load] Reading bipartite graph...")
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    print(f"[Info] Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    print("[Run] Computing subgraph metrics (CSV-based specialties, unknowns excluded)...")
    results = run_subgraph_metrics(G)

    print("\n[Done] Outputs saved:")
    for k, v in results.items():
        print(f"  {k:15s}: {v}")
