# export_node_edge_lists.py
from __future__ import annotations
from pathlib import Path
import pickle
import networkx as nx
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
HERE = Path(__file__).resolve().parents[1]
DATA_DIR = HERE / "data"
GRAPH_PATH = DATA_DIR / "provider_drug_graph.pkl"

NODE_CSV = DATA_DIR / "nodes.csv"
EDGE_CSV = DATA_DIR / "edges.csv"

# -----------------------------
# Load graph safely
# -----------------------------
print(f"[Load] Reading graph from {GRAPH_PATH} ...")
with open(GRAPH_PATH, "rb") as f:
    G: nx.Graph = pickle.load(f)
print(f"[Info] Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

# -----------------------------
# Node list
# -----------------------------
nodes_data = []
for node, attrs in G.nodes(data=True):
    node_type = attrs.get("type", "unknown")
    specialty = attrs.get("specialty", None)
    nodes_data.append({"node_id": node, "type": node_type, "specialty": specialty})

nodes_df = pd.DataFrame(nodes_data)
nodes_df.to_csv(NODE_CSV, index=False)
print(f"[Write] Node list written to: {NODE_CSV} ({len(nodes_df):,} rows)")

# -----------------------------
# Edge list
# -----------------------------
edges_data = []
for u, v, attrs in G.edges(data=True):
    weight = attrs.get("weight", 1)
    edges_data.append({"source": u, "target": v, "weight": weight})

edges_df = pd.DataFrame(edges_data)
edges_df.to_csv(EDGE_CSV, index=False)
print(f"[Write] Edge list written to: {EDGE_CSV} ({len(edges_df):,} rows)")

print("[Done] Export complete.")
