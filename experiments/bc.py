# bc.py — global betweenness centrality (aligned with your graph + PPR wiring)
from __future__ import annotations
import json, pickle
from pathlib import Path
from typing import Any, Iterable, Dict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# Paths (same style as PPR)
# -----------------------------
HERE = Path(__file__).resolve().parents[1]
GRAPH_PATH   = HERE / "data" / "provider_drug_graph.pkl"
PROVIDERS_CSV= HERE / "data" / "providers.csv"
OUT_ROOT     = HERE / "Results"
OUTPUT_DIR   = OUT_ROOT / "bc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Toggles
# -----------------------------
USE_WEIGHTS   = True
INVERT_WEIGHTS= True      # if True, shortest-path length = 1/(weight+EPS)
EPS           = 1e-9
APPROXIMATE   = False     # use sampling for BC on huge graphs
K_SAMPLES     = 1000
RANDOM_SEED   = 42
Z_OUTLIER     = 2.0
DPI           = 200

# -----------------------------
# Load graph
# -----------------------------
def _load_graph(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.Graph):
        raise TypeError(f"Loaded object is not a NetworkX graph: {type(G)}")
    return G

# -----------------------------
# Provider detection (mirror PPR intent + your diag)
# -----------------------------
def _identify_providers(G: nx.Graph) -> Iterable[Any]:
    # 1) Primary: your graph uses node attribute ntype='provider'
    by_ntype = [n for n, d in G.nodes(data=True) if d.get("ntype") == "provider"]
    if by_ntype:
        return by_ntype

    # 2) Fallback: overlap with providers.csv (union of int + str)
    if not PROVIDERS_CSV.exists():
        return []
    df = pd.read_csv(PROVIDERS_CSV, usecols=["provider_id"])
    ids_int = set(pd.to_numeric(df["provider_id"], errors="coerce").dropna().astype(int).tolist())
    ids_str = set(df["provider_id"].astype(str).tolist())
    nodes   = list(G.nodes)
    over_int= [n for n in nodes if isinstance(n, (int, np.integer)) and n in ids_int]
    over_str= [n for n in nodes if isinstance(n, str) and n in ids_str]
    prov = list({*over_int, *over_str})  # union, don’t throw one set away
    return prov

# -----------------------------
# Specialty resolution
# -----------------------------
def _node_specialty(G: nx.Graph, n: Any) -> str:
    # Prefer node attribute (your graph already stores 'specialty' on nodes)
    s = G.nodes[n].get("specialty")
    if s is not None:
        s = str(s).strip()
        if s and s not in {"0", "nan", "None"}:
            return s
    # Fallback: providers.csv (keyed to node dtype)
    if not PROVIDERS_CSV.exists():
        return "Unknown"
    df = pd.read_csv(PROVIDERS_CSV, usecols=[c for c in ["provider_id","specialty_source_value","specialty_best"] if c in pd.read_csv(PROVIDERS_CSV, nrows=0).columns])
    if df.empty:
        return "Unknown"
    # Compose specialty, prefer source, fallback to best
    spec = (df.get("specialty_source_value", pd.Series([""]*len(df))).fillna("").astype(str))
    if "specialty_best" in df.columns:
        spec = spec.where(spec.str.strip().ne(""), df["specialty_best"].astype(str))
    spec = spec.fillna("").astype(str)
    bad = spec.str.strip().isin(["", "0", "nan", "None"])
    spec = spec.mask(bad, "Unknown")
    # Match dtype of node
    if isinstance(n, (int, np.integer)):
        df["key"] = pd.to_numeric(df["provider_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["key"]); df["key"] = df["key"].astype(int)
    else:
        df["key"] = df["provider_id"].astype(str)
    m = dict(zip(df["key"], spec))
    return m.get(n, "Unknown")

# -----------------------------
# BC computation
# -----------------------------
def _edge_weight_to_length(G: nx.Graph):
    for _, _, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["length"] = 1.0 / (w + EPS)

def _compute_bc(G: nx.Graph) -> Dict[Any, float]:
    weight_key = None
    if USE_WEIGHTS:
        if INVERT_WEIGHTS:
            _edge_weight_to_length(G)
            weight_key = "length"
        else:
            weight_key = "weight"
    if APPROXIMATE:
        nodes = list(G.nodes())
        k = None if K_SAMPLES >= len(nodes) else K_SAMPLES
        return nx.betweenness_centrality(G, k=k, normalized=True, weight=weight_key, seed=RANDOM_SEED)
    return nx.betweenness_centrality(G, normalized=True, weight=weight_key)

# -----------------------------
# Utilities
# -----------------------------
def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd is None or sd <= 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def _plot_hist(prov_df: pd.DataFrame, path: Path):
    plt.figure(figsize=(8,5))
    plt.hist(prov_df["betweenness"].values, bins=60)
    plt.xlabel("Provider Betweenness Centrality")
    plt.ylabel("Count")
    plt.title("Distribution of Provider Betweenness Centrality")
    mu, sd = prov_df["betweenness"].mean(), prov_df["betweenness"].std()
    if sd and sd > 0:
        thr = mu + Z_OUTLIER * sd
        plt.axvline(thr, linestyle="--")
        plt.text(thr, plt.ylim()[1]*0.95, f"z>{Z_OUTLIER:.1f}", rotation=90, va="top")
    plt.tight_layout(); plt.savefig(path, dpi=DPI); plt.close()

def _plot_spec_bars(spec_df: pd.DataFrame, path: Path):
    plt.figure(figsize=(10, max(4, 0.25*len(spec_df))))
    y = np.arange(len(spec_df))
    plt.barh(y, spec_df["bc_summary"].values)
    plt.yticks(y, spec_df["specialty"].values)
    plt.xlabel("Betweenness (summary)")
    plt.title("Specialty-level Betweenness Summary")
    plt.gca().invert_yaxis()
    plt.tight_layout(); plt.savefig(path, dpi=240); plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    G = _load_graph(GRAPH_PATH)

    # Identify providers (ntype first, then union overlap int+str)
    providers = _identify_providers(G)
    if not providers:
        raise RuntimeError("No providers found: check ntype='provider' or providers.csv overlap.")
    # (Optional) stamp a standard 'type' attr for downstream tools
    prov_set = set(providers)
    nx.set_node_attributes(G, {n: ("provider" if n in prov_set else "drug") for n in G.nodes}, name="type")

    # Global BC
    bc = _compute_bc(G)

    # Provider table keyed by actual node labels + node specialty
    prov_bc = pd.DataFrame({
        "provider_node": providers,
        "betweenness": [float(bc.get(n, 0.0)) for n in providers],
        "specialty":   [ _node_specialty(G, n) for n in providers ],
    })
    prov_bc["z"] = _zscore(prov_bc["betweenness"])
    prov_bc["rank_desc"] = prov_bc["betweenness"].rank(ascending=False, method="min").astype(int)
    prov_bc["is_outlier"] = prov_bc["z"] > Z_OUTLIER

    # Save CSV
    out_csv = OUTPUT_DIR / "providers_bc.csv"
    prov_bc.sort_values("betweenness", ascending=False).to_csv(out_csv, index=False)

    # Plots
    _plot_hist(prov_bc, OUTPUT_DIR / "providers_bc_hist.png")
    # specialty summary (median by default)
    spec_df = (prov_bc.groupby("specialty", as_index=False)["betweenness"]
                        .median().rename(columns={"betweenness":"bc_summary"})
                        .sort_values("bc_summary", ascending=False))
    _plot_spec_bars(spec_df, OUTPUT_DIR / "specialty_bc_bar.png")

    # Summary JSON
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "n_providers": int(prov_bc.shape[0]),
            "outliers_z_gt": float(Z_OUTLIER),
            "outliers_count": int(prov_bc["is_outlier"].sum()),
            "top5_providers": prov_bc.nlargest(5, "betweenness")[["provider_node","specialty","betweenness","z","rank_desc"]].to_dict(orient="records"),
        }, f, indent=2)

    print("Saved:\n ", out_csv, "\n ", OUTPUT_DIR / "providers_bc_hist.png",
          "\n ", OUTPUT_DIR / "specialty_bc_bar.png", "\n ", OUTPUT_DIR / "summary.json")

if __name__ == "__main__":
    main()
