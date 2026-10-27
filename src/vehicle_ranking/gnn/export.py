"""Point-in-time graphs and embedding export for the ranking stage.

The GNN used for ranking sees no data after T_emb = T_train - W days: its weights, node features,
scalers and message-passing graph all end at T_emb. Embeddings are exported twice with the same
weights: at T_emb for LambdaRank training rows, whose targets fall in (T_emb, T_train], and at
T_train for validation, test and holdout rows, whose targets fall after T_train.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

from vehicle_ranking.config import GnnTrainingConfig, GraphConfig
from vehicle_ranking.data.export import ExportData
from vehicle_ranking.features.users import user_features
from vehicle_ranking.gnn.model import HeteroGNN
from vehicle_ranking.gnn.train import NodeScalers
from vehicle_ranking.graph.build import build_graph
from vehicle_ranking.graph.temporal import create_time_sliced_graph
from vehicle_ranking.utils import DAY_SECONDS, Cutoffs

EMBEDDING_SNAPSHOTS = {"train_rows": "embeddings_t_emb", "eval_rows": "embeddings_t_train"}


def embedding_cutoff(train_cutoff: int, window_days: int) -> int:
    return train_cutoff - window_days * DAY_SECONDS


def gnn_cutoffs(spec: Cutoffs, cfg: GnnTrainingConfig) -> Cutoffs:
    """GNN training ends at T_emb; validation and test take the following val_days and test_days."""
    train = embedding_cutoff(spec.train, cfg.emb_cutoff_days)
    val = train + cfg.val_days * DAY_SECONDS
    test = val + cfg.test_days * DAY_SECONDS
    if test > spec.train:
        raise ValueError("GNN validation and test windows must end before the ranking training cutoff")
    return Cutoffs(train=train, val=val, test=test)


def graph_at(export: ExportData, cutoff: int, cfg: GraphConfig, scalers: Optional[NodeScalers] = None):
    """Full graph with user features as of the cutoff, scaled with scalers fitted at the cutoff if none are given."""
    data = build_graph(export, user_features(export, cutoff), cfg)
    scalers = scalers or NodeScalers.fit(data, cutoff)
    scalers.apply(data)
    return data, scalers


@torch.no_grad()
def export_embeddings(model: HeteroGNN, data: HeteroData, cutoff_ts: int, out_dir: Path) -> None:
    """Run the trained weights on the graph sliced at cutoff_ts; behavioral edges after it are absent."""
    model = model.cpu().eval()
    graph = create_time_sliced_graph(data, cutoff_ts).cpu()
    weights = {et: graph[et].edge_weight for et in graph.edge_types if "edge_weight" in graph[et]}
    emb = model(graph.x_dict, graph.edge_index_dict, edge_weight_dict=weights)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "user_embeddings.npy", emb["user"].numpy())
    np.save(out_dir / "item_embeddings.npy", emb["item"].numpy())
