"""Build the heterogeneous user-vehicle-property graph."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from vehicle_ranking.config import GraphConfig
from vehicle_ranking.data.export import ExportData

BEHAVIOR_RELATIONS = {
    "bid": ("user", "BID", "item"),
    "purchase": ("user", "PURCHASE", "item"),
    "fav": ("user", "FAVORITE", "item"),
    "alert": ("user", "ALERT", "item"),
}
PROPERTY_RELATION = ("item", "HAS_PROPERTY", "property")
BID = BEHAVIOR_RELATIONS["bid"]


def numeric_feature_columns(df: pd.DataFrame, id_col: str) -> list[str]:
    return [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]


def behavior_edge_weights(name: str, edges: pd.DataFrame, cfg: GraphConfig) -> np.ndarray:
    """Commitment weights: bid 3 * (1 + 0.2 * ln(1 + bid count)), purchase 10, favorite 2, alert 1."""
    if name == "bid":
        count = edges["count"].to_numpy(float) if "count" in edges.columns else np.ones(len(edges))
        return cfg.weight_bid * (1.0 + cfg.bid_count_amplification * np.log1p(count))
    weight = {"purchase": cfg.weight_purchase, "fav": cfg.weight_fav, "alert": cfg.weight_alert}[name]
    return np.full(len(edges), weight)


def _index(src: np.ndarray, dst: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.stack([src, dst]), dtype=torch.long)


def build_graph(export: ExportData, user_frame: pd.DataFrame, cfg: GraphConfig) -> HeteroData:
    """user_frame holds the user features at the cutoff the graph is used for, one row per u_idx."""
    user_frame = user_frame.sort_values("u_idx")
    if not np.array_equal(user_frame["u_idx"].to_numpy(), np.arange(export.n_users)):
        raise ValueError("user_frame must contain every u_idx exactly once")

    data = HeteroData()
    data["user"].x = torch.from_numpy(
        user_frame[numeric_feature_columns(user_frame, "u_idx")].fillna(0.0).to_numpy(np.float32))
    data["item"].x = torch.from_numpy(
        export.items[numeric_feature_columns(export.items, "i_idx")].fillna(0.0).to_numpy(np.float32))
    data["property"].num_nodes = len(export.props)

    for name, edge_type in BEHAVIOR_RELATIONS.items():
        df = export.edges[name]
        store = data[edge_type]
        store.edge_index = _index(df["u_idx"].to_numpy(), df["i_idx"].to_numpy())
        store.ts = torch.as_tensor(df["ts"].to_numpy(), dtype=torch.long)
        store.edge_weight = torch.as_tensor(behavior_edge_weights(name, df, cfg), dtype=torch.float32)
    data[PROPERTY_RELATION].edge_index = _index(export.item_props["i_idx"].to_numpy(), export.item_props["p_idx"].to_numpy())

    add_reverse_edges(data, cfg.reverse_relations)
    apply_relation_weights(data, cfg.mp_rel_weights)
    return data


def add_reverse_edges(data: HeteroData, relations: Iterable[str]) -> HeteroData:
    """Add (dst, REL_rev, src) for the selected relations, copying per-edge attributes."""
    for src, rel, dst in list(data.edge_types):
        if rel not in relations:
            continue
        store = data[(src, rel, dst)]
        rev = data[(dst, f"{rel}_rev", src)]
        rev.edge_index = store.edge_index.flip(0).clone()
        for key, value in store.items():
            if key != "edge_index" and isinstance(value, torch.Tensor) and value.size(0) == store.num_edges:
                rev[key] = value.clone()
    return data


def apply_relation_weights(data: HeteroData, rel_weights: Dict[str, float]) -> HeteroData:
    """Multiply edge_weight by the relation weight (a reverse relation uses its base relation's value)."""
    for edge_type in data.edge_types:
        store = data[edge_type]
        if store.num_edges == 0:
            continue
        weight = float(rel_weights.get(edge_type[1].removesuffix("_rev"), 1.0))
        if "edge_weight" in store:
            store.edge_weight = store.edge_weight * weight
        else:
            store.edge_weight = torch.full((store.num_edges,), weight, dtype=torch.float32)
    return data
