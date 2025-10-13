"""
Personalized PageRank (PPR) analysis for provider–drug bipartite graphs.

Outputs:
- Global bar chart: specialty proximity to top 3% global prescribers
- Per-specialty heatmap: proximity of each top prescriber to specialties (row-normalized to %)
- CSV exports for reproducibility

Assumptions:
- You already built a bipartite NetworkX graph G with nodes of type {'provider','drug'}
- Edges have attribute 'weight' = # of exposures (patient count) for that provider–drug
- provider_specialty: dict mapping provider_id -> specialty string (preferred: specialty_source_value)

Usage (example):
    from PPR_Pipeline import run_ppr_analysis
    run_ppr_analysis(G, provider_specialty)

Author: Project 1 Network Analysis
"""
from __future__ import annotations
import os
from typing import Dict, Iterable, Set
import math
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import numpy as np

# =====================
# ---- CONFIG AREA ----
# =====================

# Output directories (change these as needed)
HERE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HERE / "Results" / "ppr"
FIG_DIR = os.path.join(OUTPUT_DIR, "figs")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

# Seed selection
TOP_GLOBAL_PCT = 0.03   # top 3% providers by exposure volume
PER_SPECIALTY_TOP_1 = True  # always picks the single top prescriber per specialty

# Transition weighting toggle
USE_WEIGHTED_TRANSITIONS = True  # if False, transitions are uniform regardless of edge 'weight'
ALPHA = 0.85  # teleport/back probability for PageRank

# Specialty inclusion thresholding
TOP_K_SPECIALTIES_BY_EXPOSURES = 7  # keep only the top-K specialties (by total exposures) for charts

# Figure settings
DPI = 200

# =====================
# ---- CORE LOGIC  ----
# =====================


def _ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)


def _node_type(G: nx.Graph, n) -> str:
    t = G.nodes[n].get("type")
    if t not in {"provider", "drug"}:
        raise ValueError(f"Node {n} missing valid 'type' attribute ('provider'|'drug'). Got: {t}")
    return t


def compute_provider_exposures(G: nx.Graph) -> Dict:
    """Sum of incident edge weights per provider node."""
    total = {}
    for u in G.nodes:
        if _node_type(G, u) == "provider":
            s = 0.0
            for _, v, d in G.edges(u, data=True):
                w = d.get("weight", 1.0)
                s += float(w)
            total[u] = s
    return total


def exposures_by_specialty(provider_exposures: Dict, provider_specialty: Dict) -> pd.Series:
    df = (
        pd.Series(provider_exposures, name="exposures")
          .rename_axis("provider")
          .to_frame()
          .assign(specialty=lambda x: x.index.map(lambda p: provider_specialty.get(p, "UNKNOWN")))
          .groupby("specialty", dropna=False)["exposures"].sum()
          .sort_values(ascending=False)
    )
    return df


def select_top_global_seeds(provider_exposures: Dict, pct: float) -> Set:
    if not (0 < pct <= 1):
        raise ValueError("pct must be in (0,1]")
    n = max(1, math.floor(len(provider_exposures) * pct))
    ranked = sorted(provider_exposures.items(), key=lambda kv: kv[1], reverse=True)
    seeds = {p for p, _ in ranked[:n]}
    return seeds


def select_top1_per_specialty(provider_exposures: Dict, provider_specialty: Dict) -> Dict[str, str]:
    """
    Return mapping seed_specialty -> provider_id (top prescriber in that specialty).
    Only considers providers present in provider_specialty (UNKNOWNs already excluded upstream).
    """
    # Keep only providers that have a specialty mapping
    filtered = {p: exp for p, exp in provider_exposures.items() if p in provider_specialty}

    # Build a frame with provider as a real column (not index)
    df = (pd.Series(filtered, name="exposures")
            .rename_axis("provider")
            .reset_index()
            .assign(specialty=lambda x: x["provider"].map(provider_specialty)))

    # Defensive: drop rows with missing specialty after mapping (should be none)
    df = df.dropna(subset=["specialty"])

    # For each specialty, take the provider with max exposures
    df_sorted = df.sort_values(["specialty", "exposures"], ascending=[True, False])
    top_rows = df_sorted.groupby("specialty", as_index=False).head(1)

    # Build specialty -> provider_id map
    return {str(row["specialty"]): row["provider"] for _, row in top_rows.iterrows()}



def pagerank_for_seeds(
    G: nx.Graph,
    seed_nodes: Iterable,
    use_weighted: bool = True,
    alpha: float = 0.85,
) -> Dict:
    """Run Personalized PageRank with a uniform personalization over the provided seed nodes."""
    seed_nodes = list(seed_nodes)
    if len(seed_nodes) == 0:
        raise ValueError("No seed nodes provided for PPR.")

    # Personalization: uniform mass over seeds
    mass = 1.0 / len(seed_nodes)
    personalization = {n: 0.0 for n in G.nodes}
    for s in seed_nodes:
        personalization[s] = mass

    weight_key = "weight" if use_weighted else None
    pr = nx.pagerank(G, alpha=alpha, personalization=personalization, weight=weight_key)
    return pr


def aggregate_provider_scores_to_specialty(scores: Dict, provider_specialty: Dict) -> pd.Series:
    df = (
        pd.Series(scores, name="score")
          .rename_axis("node")
          .to_frame()
    )
    # keep only providers
    df = df.loc[[n for n in df.index if n in provider_specialty], :].copy()
    df["specialty"] = df.index.map(lambda p: provider_specialty.get(p, "UNKNOWN"))
    agg = df.groupby("specialty")["score"].sum().sort_values(ascending=False)
    return agg


# =====================
# ---- PLOTTING     ----
# =====================


def _save_bar_chart(series: pd.Series, title: str, path: str):
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel("PPR score (aggregated by specialty)")
    plt.xlabel("Specialty")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def _save_heatmap(df: pd.DataFrame, title: str, path: str):
    plt.figure(figsize=(10, max(4, 0.5 * len(df))))
    plt.imshow(df.values, aspect='auto')
    plt.colorbar(label="Percent of PPR mass")
    plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha='right')
    plt.yticks(range(df.shape[0]), df.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


# =====================
# ---- MAIN RUNNER  ----
# =====================


def run_ppr_analysis(
    G: nx.Graph,
    provider_specialty: Dict,
    output_dir: str = OUTPUT_DIR,
    fig_dir: str = FIG_DIR,
    csv_dir: str = CSV_DIR,
    use_weighted_transitions: bool = USE_WEIGHTED_TRANSITIONS,
    alpha: float = ALPHA,
    top_global_pct: float = TOP_GLOBAL_PCT,
    top_k_specialties: int = TOP_K_SPECIALTIES_BY_EXPOSURES,
):
    """Execute both PPR modes and save figures + CSVs into Results/ppr.

    - Global: seeds = top 3% providers globally by exposures
    - Per-specialty: one run per (top prescriber of specialty), aggregate to specialties
    - Heatmap rows normalized to sum=100 (%), columns restricted to top-K specialties (by exposures)
    """
    global OUTPUT_DIR, FIG_DIR, CSV_DIR
    # allow caller to override dirs
    global OUTPUT_DIR, FIG_DIR, CSV_DIR
    OUTPUT_DIR = output_dir
    FIG_DIR = fig_dir
    CSV_DIR = csv_dir
    _ensure_dirs()

    # 1) Compute exposures & specialty ranking for thresholding
    prov_expo = compute_provider_exposures(G)
    spec_expo = exposures_by_specialty(prov_expo, provider_specialty)
    top_specs = spec_expo.head(top_k_specialties).index.tolist()

    # 2) GLOBAL PPR (top 3% providers)
    global_seeds = select_top_global_seeds(prov_expo, pct=top_global_pct)
    pr_global = pagerank_for_seeds(G, global_seeds, use_weighted=use_weighted_transitions, alpha=alpha)
    spec_scores_global = aggregate_provider_scores_to_specialty(pr_global, provider_specialty)

    # keep only top-K specialties for the global chart (improves readability)
    spec_scores_global_top = spec_scores_global.loc[spec_scores_global.index.intersection(top_specs)]
    spec_scores_global_top = spec_scores_global_top.sort_values(ascending=False)

    # Save CSV + figure
    spec_scores_global.to_csv(os.path.join(CSV_DIR, "global_ppr_specialty_scores_all.csv"))
    spec_scores_global_top.to_csv(os.path.join(CSV_DIR, "global_ppr_specialty_scores_topK.csv"))
    _save_bar_chart(
        spec_scores_global_top,
        title=f"Global PPR (seeds = top {int(top_global_pct*100)}% providers)\nTransition weighting: {'weighted' if use_weighted_transitions else 'uniform'}",
        path=os.path.join(FIG_DIR, "global_ppr_bar_topK.png"),
    )

    # 3) PER-SPECIALTY PPR (top prescriber per specialty)
    top1_map = select_top1_per_specialty(prov_expo, provider_specialty)  # {specialty -> provider}

    # Limit rows to the same top-K specialties (seed specialties)
    seed_specialties = [s for s in top_specs if s in top1_map]

    rows = []
    for s in seed_specialties:
        seed_provider = top1_map[s]
        pr_row = pagerank_for_seeds(G, [seed_provider], use_weighted=use_weighted_transitions, alpha=alpha)
        agg_row = aggregate_provider_scores_to_specialty(pr_row, provider_specialty)
        # align to columns = top_specs (fill missing with 0)
        agg_row = agg_row.reindex(top_specs, fill_value=0.0)
        # normalize row to 100%
        total = agg_row.sum()
        if total > 0:
            agg_row = (agg_row / total) * 100.0
        rows.append(agg_row.rename(s))

    heatmap_df = pd.DataFrame(rows)

    # Save CSV + heatmap
    heatmap_df.to_csv(os.path.join(CSV_DIR, "per_specialty_ppr_heatmap_percent_topKxTopK.csv"))
    _save_heatmap(
        heatmap_df,
        title=f"Per-Specialty PPR (rows: top prescriber of each specialty)\nColumns: specialties (Top-{top_k_specialties} by exposures) — Row normalized to %\nTransition weighting: {'weighted' if use_weighted_transitions else 'uniform'}",
        path=os.path.join(FIG_DIR, "per_specialty_ppr_heatmap_topK.png"),
    )

    # 4) Also store provider-level CSVs for auditing (optional, only for seed specialties)
    #    This saves the full PPR vector per seed provider.
    prov_scores_dir = os.path.join(CSV_DIR, "per_specialty_provider_scores")
    os.makedirs(prov_scores_dir, exist_ok=True)
    for s in seed_specialties:
        seed_provider = top1_map[s]
        pr_row = pagerank_for_seeds(G, [seed_provider], use_weighted=use_weighted_transitions, alpha=alpha)
        # keep only providers
        pr_prov = {p: sc for p, sc in pr_row.items() if p in provider_specialty}
        pd.Series(pr_prov, name="ppr_score").sort_values(ascending=False).to_csv(
            os.path.join(prov_scores_dir, f"ppr_provider_scores_seed_{str(s).replace('/', '_')}.csv")
        )

    # Quick return payload
    return {
        "top_specialties": top_specs,
        "global_bar_path": os.path.join(FIG_DIR, "global_ppr_bar_topK.png"),
        "heatmap_path": os.path.join(FIG_DIR, "per_specialty_ppr_heatmap_topK.png"),
        "csv_global_all": os.path.join(CSV_DIR, "global_ppr_specialty_scores_all.csv"),
        "csv_global_topK": os.path.join(CSV_DIR, "global_ppr_specialty_scores_topK.csv"),
        "csv_heatmap": os.path.join(CSV_DIR, "per_specialty_ppr_heatmap_percent_topKxTopK.csv"),
    }


# =====================
# ---- VALIDATION   ----
# =====================

if __name__ == "__main__":
    import os
    from pathlib import Path
    import pickle
    import numpy as np
    import pandas as pd
    import networkx as nx

    # -----------------------------
    # Paths (adjust if needed)
    # -----------------------------
    HERE = Path(__file__).resolve().parents[1]
    GRAPH_PATH = HERE / "data" / "provider_drug_graph.pkl"
    PROVIDERS_CSV = HERE / "data" / "providers.csv"
    OUT_ROOT = HERE / "Results"
    OUTPUT_DIR = OUT_ROOT / "ppr"
    MAP_PATH = OUT_ROOT / "data" / "provider_specialty_mapping.pkl"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MAP_PATH.parent, exist_ok=True)

    # -----------------------------
    # Load graph
    # -----------------------------
    print("[Load] Reading bipartite graph...")
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # -----------------------------
    # Identify providers on the graph
    # -----------------------------
    provider_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "provider"]

    if not provider_nodes:
        # Fallback: infer providers via overlap with providers.csv
        if not PROVIDERS_CSV.exists():
            raise RuntimeError(
                "No nodes labeled type='provider' and data/providers.csv not found. "
                "Either add node attribute 'type' or provide providers.csv."
            )
        df_ids = pd.read_csv(PROVIDERS_CSV, usecols=["provider_id"])
        ids_int = pd.to_numeric(df_ids["provider_id"], errors="coerce").dropna().astype(int).tolist()
        ids_str = df_ids["provider_id"].astype(str).tolist()

        nodes = list(G.nodes)
        overlap_int = [n for n in nodes if isinstance(n, (int, np.integer)) and n in set(ids_int)]
        overlap_str = [n for n in nodes if isinstance(n, str) and n in set(ids_str)]

        provider_nodes = overlap_int if len(overlap_int) >= len(overlap_str) else overlap_str
        if not provider_nodes:
            raise RuntimeError("Unable to infer provider nodes from graph/CSV overlap.")

        # Label types for downstream code
        nx.set_node_attributes(G, {n: "provider" for n in provider_nodes}, name="type")
        nx.set_node_attributes(G, {n: "drug" for n in set(G.nodes) - set(provider_nodes)}, name="type")

    exemplar = provider_nodes[0]
    providers_are_int = isinstance(exemplar, (int, np.integer))

    # -----------------------------
    # Build provider→specialty mapping (exclude unknowns)
    # -----------------------------
    if not MAP_PATH.exists():
        print("[Build] Creating provider→specialty mapping from providers.csv ...")
        usecols = ["provider_id", "specialty_source_value", "specialty_concept_id", "specialty_best"]
        dfp = pd.read_csv(PROVIDERS_CSV, usecols=usecols)

        # Normalize provider_id dtype to match graph nodes
        if providers_are_int:
            dfp["provider_id"] = pd.to_numeric(dfp["provider_id"], errors="coerce").astype("Int64")
            dfp = dfp.dropna(subset=["provider_id"])
            dfp["provider_id"] = dfp["provider_id"].astype(int)
        else:
            dfp["provider_id"] = dfp["provider_id"].astype(str)

        # Choose specialty: prefer specialty_source_value
        def pick_spec(row):
            ssv = str(row.get("specialty_source_value") or "").strip()
            return ssv

        dfp["specialty"] = dfp.apply(pick_spec, axis=1)

        # Exclude unknowns / empties / "0" / NaN
        bad = dfp["specialty"].isna() | (dfp["specialty"].str.strip() == "") | (dfp["specialty"].str.strip() == "0")
        dfp = dfp[~bad].copy()

        # Keep only providers present in the graph’s provider node set
        keep = set(provider_nodes)
        mapping = {pid: spec for pid, spec in zip(dfp["provider_id"], dfp["specialty"]) if pid in keep}

        with open(MAP_PATH, "wb") as f:
            pickle.dump(mapping, f)

    print("[Load] Reading provider→specialty mapping...")
    with open(MAP_PATH, "rb") as f:
        provider_specialty = pickle.load(f)

    # Safety: ensure mapping keys are a subset of graph providers
    missing_in_graph = set(provider_specialty.keys()) - set(provider_nodes)
    if missing_in_graph:
        # drop any that don't exist in the graph (should be rare after filtering)
        for k in list(missing_in_graph):
            provider_specialty.pop(k, None)

    # -----------------------------
    # Run Personalized PageRank
    # -----------------------------
    print("[Run] Executing Personalized PageRank analysis...")
    results = run_ppr_analysis(G, provider_specialty)

    # -----------------------------
    # Output summary
    # -----------------------------
    print("\n[Done] Outputs saved:")
    for k, v in results.items():
        print(f"  {k:25s}: {v}")
