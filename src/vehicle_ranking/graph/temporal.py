"""Leakage guards: cutoff visibility, time-sliced message-passing graphs and target-edge removal."""
from __future__ import annotations

from typing import Dict

import torch
from torch_geometric.data import HeteroData


def visible_nodes_by_cutoff(data: HeteroData, cutoff_ts: int) -> Dict[str, torch.Tensor]:
    """A node is visible if it touches a timestamped edge with ts <= cutoff.

    Node types that no timestamped edge type touches (property) are static schema nodes and are
    always visible.
    """
    temporal_types = set()
    for edge_type in data.edge_types:
        if data[edge_type].get("ts", None) is not None:
            temporal_types.update((edge_type[0], edge_type[2]))

    visible = {
        ntype: torch.full((int(data[ntype].num_nodes),), ntype not in temporal_types, dtype=torch.bool)
        for ntype in data.node_types
    }
    for edge_type in data.edge_types:
        store = data[edge_type]
        ts = store.get("ts", None)
        if ts is None or store.num_edges == 0:
            continue
        mask = (ts > 0) & (ts <= cutoff_ts)
        edge_index = store.edge_index[:, mask]
        visible[edge_type[0]][edge_index[0]] = True
        visible[edge_type[2]][edge_index[1]] = True
    return visible


def _copy_edges(src_store, dst_store, mask: torch.Tensor) -> None:
    n = src_store.edge_index.size(1)
    dst_store.edge_index = src_store.edge_index[:, mask].clone()
    for key, value in src_store.items():
        if key != "edge_index" and isinstance(value, torch.Tensor) and value.size(0) == n:
            dst_store[key] = value[mask].clone()


def create_time_sliced_graph(data: HeteroData, cutoff_ts: int) -> HeteroData:
    """Message-passing graph at a cutoff.

    All nodes and node features are kept. Timestamped edges are kept when ts <= cutoff. An edge
    without a timestamp is kept when its source node is visible at the cutoff, so vehicle-to-property
    edges enter only from vehicles with earlier interactions, while property-to-vehicle edges reach
    every vehicle, including newly listed ones.
    """
    sliced = HeteroData()
    for ntype in data.node_types:
        sliced[ntype].num_nodes = data[ntype].num_nodes
        for key, value in data[ntype].items():
            if key != "num_nodes" and isinstance(value, torch.Tensor):
                sliced[ntype][key] = value.clone()

    visible = visible_nodes_by_cutoff(data, cutoff_ts)
    for edge_type in data.edge_types:
        store = data[edge_type]
        if store.num_edges == 0:
            continue
        ts = store.get("ts", None)
        if ts is not None:
            mask = (ts > 0) & (ts <= cutoff_ts)
        else:
            mask = visible[edge_type[0]][store.edge_index[0]]
        if torch.any(mask):
            _copy_edges(store, sliced[edge_type], mask)
    return sliced


def drop_item_behavior(graph: HeteroData, items: torch.Tensor) -> None:
    """Remove every behavior edge (user-item and item-user) of the given vehicles.

    Target vehicles then look like newly listed ones: they receive messages only from their
    property nodes, which is the situation of every candidate at decision time.
    """
    drop = torch.zeros(int(graph["item"].num_nodes), dtype=torch.bool)
    drop[items] = True
    for edge_type in graph.edge_types:
        if {edge_type[0], edge_type[2]} != {"user", "item"}:
            continue
        store = graph[edge_type]
        if store.num_edges == 0:
            continue
        item_side = store.edge_index[1] if edge_type[2] == "item" else store.edge_index[0]
        keep = ~drop[item_side]
        n = store.edge_index.size(1)
        store.edge_index = store.edge_index[:, keep]
        for key, value in list(store.items()):
            if key != "edge_index" and isinstance(value, torch.Tensor) and value.size(0) == n:
                store[key] = value[keep]
