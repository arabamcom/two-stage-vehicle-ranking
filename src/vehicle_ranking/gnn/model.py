"""Heterogeneous GNN encoder for the user-vehicle-property graph."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, HeteroConv, Linear

from vehicle_ranking.config import ModelConfig

EdgeType = Tuple[str, str, str]


class HeteroGNN(nn.Module):
    """Type-specific input projections, relation-wise GraphConv layers averaged per node type,
    residual connections, and L2-normalized 128-dimensional user and vehicle outputs.

    Property nodes have no input features; they use a learnable embedding table.
    """

    def __init__(self, data: HeteroData, cfg: ModelConfig):
        super().__init__()
        self.residual = cfg.residual
        self.prop_embedding = nn.Embedding(int(data["property"].num_nodes), cfg.prop_embedding_dim)
        nn.init.xavier_uniform_(self.prop_embedding.weight)

        self.user_mlp = nn.Sequential(Linear(data["user"].num_features, cfg.hidden_channels), nn.ReLU())
        self.item_mlp = nn.Sequential(Linear(data["item"].num_features, cfg.hidden_channels), nn.ReLU())
        self.prop_mlp = nn.Sequential(Linear(cfg.prop_embedding_dim, cfg.hidden_channels), nn.ReLU())
        self.dropout = nn.Dropout(cfg.dropout)
        self.convs = nn.ModuleList([
            HeteroConv(
                {edge_type: GraphConv((-1, -1), cfg.hidden_channels) for edge_type in data.edge_types},
                aggr="mean",
            )
            for _ in range(cfg.num_layers)
        ])
        self.out = Linear(cfg.hidden_channels, cfg.out_channels)
        self.register_buffer("tau", torch.tensor(float(cfg.temperature)))

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        n_id_dict: Optional[Dict[str, torch.Tensor]] = None,
        edge_weight_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if n_id_dict is not None and "property" in n_id_dict:
            prop_x = self.prop_mlp(self.prop_embedding(n_id_dict["property"]))
        else:
            prop_x = self.prop_mlp(self.prop_embedding.weight)
        h = {"user": self.user_mlp(x_dict["user"]), "item": self.item_mlp(x_dict["item"]), "property": prop_x}

        for conv in self.convs:
            out = conv(h, edge_index_dict, edge_weight_dict=edge_weight_dict or {})
            out = {k: self.dropout(F.relu(v)) for k, v in out.items()}
            for k, v in out.items():
                h[k] = h[k] + v if self.residual else v

        return {
            "user": F.normalize(self.out(h["user"]), p=2, dim=-1),
            "item": F.normalize(self.out(h["item"]), p=2, dim=-1),
        }


def link_scores(user_emb: torch.Tensor, item_emb: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
    """Inner product of normalized embeddings, that is, cosine similarity."""
    return (user_emb[edge_label_index[0]] * item_emb[edge_label_index[1]]).sum(dim=-1)
