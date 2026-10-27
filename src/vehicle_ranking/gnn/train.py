"""GNN training with a symmetric InfoNCE objective on bid edges and in-batch negatives.

Message-passing edges and supervision edges are disjoint. In every epoch a random share of the
vehicles with training bids becomes the target set: their bids are the supervision targets and all
their behavior edges are removed from the message-passing graph. A target vehicle then looks like
a newly listed one, which receives messages only from its property nodes, while users keep
collaborative messages from their other vehicles. Validation and test targets are vehicles listed
after the training cutoff, and their sampled negatives come from the roster of the same auction
day, so positives and negatives are equally new.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from vehicle_ranking.config import GnnTrainingConfig
from vehicle_ranking.gnn.model import HeteroGNN, link_scores
from vehicle_ranking.graph.build import BID
from vehicle_ranking.graph.temporal import create_time_sliced_graph, drop_item_behavior, visible_nodes_by_cutoff
from vehicle_ranking.utils import Cutoffs

logger = logging.getLogger(__name__)


@dataclass
class NodeScalers:
    """Vehicle features: clipping to the 0.1% and 99.9% quantiles and standardization.
    User features: standardization and clipping to [-5, 5]. Both are fitted on the nodes visible at
    the fit cutoff and reused unchanged for graphs at later cutoffs."""
    item_lo: np.ndarray
    item_hi: np.ndarray
    item_scaler: StandardScaler
    user_mean: np.ndarray
    user_std: np.ndarray

    @classmethod
    def fit(cls, data: HeteroData, cutoff: int) -> "NodeScalers":
        visible = visible_nodes_by_cutoff(data, cutoff)
        item_x = data["item"].x.numpy()[visible["item"].numpy()]
        lo, hi = np.nanquantile(item_x, 0.001, axis=0), np.nanquantile(item_x, 0.999, axis=0)
        user_x = data["user"].x.numpy()[visible["user"].numpy()]
        return cls(lo, hi, StandardScaler().fit(np.clip(item_x, lo, hi)),
                   user_x.mean(axis=0), np.maximum(user_x.std(axis=0, ddof=1), 1e-6))

    def apply(self, data: HeteroData) -> None:
        item_x = np.clip(data["item"].x.numpy(), self.item_lo, self.item_hi)
        data["item"].x = torch.from_numpy(self.item_scaler.transform(item_x).astype(np.float32))
        user_x = (data["user"].x.numpy() - self.user_mean) / self.user_std
        data["user"].x = torch.from_numpy(np.clip(user_x, -5.0, 5.0).astype(np.float32))


def positives_by_user(edge_index: torch.Tensor) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = defaultdict(set)
    for u, i in edge_index.t().tolist():
        out[u].add(i)
    return out


def masked_infonce(
    node_emb: Dict[str, torch.Tensor], batch, known_pos: Dict[int, Set[int]], tau: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    """Symmetric InfoNCE over the B x B cosine matrix of a batch of positive pairs.

    Other known positives of the same user are masked so they are not treated as negatives.
    """
    eidx = batch[BID].edge_label_index
    b = eidx.size(1)
    if b == 0:
        return torch.zeros((), device=eidx.device), 0
    scores = (node_emb["user"][eidx[0]] @ node_emb["item"][eidx[1]].T) / tau

    users = batch["user"].n_id[eidx[0]].tolist()
    items = batch["item"].n_id[eidx[1]].tolist()
    cols_of_item: Dict[int, List[int]] = defaultdict(list)
    for col, item in enumerate(items):
        cols_of_item[item].append(col)
    rows, cols = [], []
    for row, user in enumerate(users):
        for item in known_pos.get(user, ()):
            for col in cols_of_item.get(item, ()):
                if col != row:
                    rows.append(row)
                    cols.append(col)
    if rows:
        mask = torch.zeros(b, b, dtype=torch.bool, device=scores.device)
        mask[rows, cols] = True
        scores = scores.masked_fill(mask, float("-inf"))

    labels = torch.arange(b, device=scores.device)
    return 0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)), b


def split_supervision(train_pos: torch.Tensor, ratio: float, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random share of the vehicles with training bids; returns their bids (targets) and the vehicles."""
    items = torch.unique(train_pos[1])
    k = max(1, int(round(len(items) * ratio)))
    chosen = items[torch.randperm(len(items), generator=generator)[:k]]
    return train_pos[:, torch.isin(train_pos[1], chosen)], chosen


def roster_negatives(pos: torch.Tensor, rosters: Dict[int, np.ndarray], listing_day: pd.Series,
                     known: Dict[int, Set[int]], ratio: float, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Positives plus `ratio` negatives each, drawn from the roster of the positive vehicle's listing day.

    known holds every (user, vehicle) pair with any behavior; those pairs are never drawn as negatives.
    """
    n_neg = int(ratio)
    src, dst = [], []
    for u, i in pos.t().tolist():
        pool = rosters.get(int(listing_day.get(i, -1)), np.empty(0, dtype=np.int64))
        pool = pool[~np.isin(pool, list(known.get(u, ()) | {i}))]
        for j in rng.choice(pool, size=min(n_neg, len(pool)), replace=False):
            src.append(u)
            dst.append(int(j))
    index = torch.cat([pos, torch.tensor([src, dst], dtype=torch.long)], dim=1)
    labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(len(src))])
    return index, labels


def _batch_inputs(batch):
    n_id = {nt: batch[nt].n_id for nt in batch.node_types if "n_id" in batch[nt]}
    weights = {et: batch[et].edge_weight for et in batch.edge_types if "edge_weight" in batch[et]}
    return batch.x_dict, batch.edge_index_dict, n_id, weights


def _loader(graph: HeteroData, edges: torch.Tensor, cfg: GnnTrainingConfig, shuffle: bool, labels=None):
    return LinkNeighborLoader(
        graph,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.batch_size,
        edge_label_index=(BID, edges),
        edge_label=labels,
        shuffle=shuffle,
    )


class GnnTrainer:
    def __init__(self, model: HeteroGNN, cfg: GnnTrainingConfig, output_dir: Path):
        self.model = model
        self.cfg = cfg
        self.output_dir = output_dir
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        self.optimizer = None

    def _forward(self, batch):
        x, ei, n_id, w = _batch_inputs(batch)
        return self.model(x, ei, n_id, w)

    def _epoch(self, loader, known_pos) -> float:
        self.model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss, b = masked_infonce(self._forward(batch), batch, known_pos, self.model.tau)
            if b == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total += float(loss) * b
            count += b
        return total / max(count, 1)

    @torch.no_grad()
    def _infonce(self, loader, known_pos) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            loss, b = masked_infonce(self._forward(batch), batch, known_pos, self.model.tau)
            total += float(loss) * b
            count += b
        return total / count if count else float("inf")

    @torch.no_grad()
    def link_prediction(self, loader) -> Tuple[float, float]:
        """ROC-AUC and average precision on positives plus sampled negatives."""
        self.model.eval()
        scores, labels = [], []
        for batch in loader:
            batch = batch.to(self.device)
            emb = self._forward(batch)
            scores.append(link_scores(emb["user"], emb["item"], batch[BID].edge_label_index).cpu())
            labels.append(batch[BID].edge_label.cpu())
        y, p = torch.cat(labels).numpy(), torch.cat(scores).numpy()
        return float(roc_auc_score(y, p)), float(average_precision_score(y, p))

    def _epoch_loader(self, train_mp: HeteroData, train_pos: torch.Tensor, generator: torch.Generator):
        targets, items = split_supervision(train_pos, self.cfg.supervision_ratio, generator)
        graph = train_mp.clone()
        drop_item_behavior(graph, items)
        return _loader(graph, targets, self.cfg, shuffle=True)

    def fit(self, data: HeteroData, cutoffs: Cutoffs, rosters: Dict[int, np.ndarray], listing_day: pd.Series) -> dict:
        """Train on bids up to cutoffs.train, select on validation InfoNCE, report sampled link prediction."""
        ts = data[BID].ts
        bid_index = data[BID].edge_index
        train_pos = bid_index[:, ts <= cutoffs.train]
        val_pos = bid_index[:, (ts > cutoffs.train) & (ts <= cutoffs.val)]
        test_pos = bid_index[:, (ts > cutoffs.val) & (ts <= cutoffs.test)]
        if min(train_pos.size(1), val_pos.size(1), test_pos.size(1)) == 0:
            raise ValueError("Train, validation and test bid splits must all be non-empty")
        logger.info("Bid splits: train=%d val=%d test=%d", train_pos.size(1), val_pos.size(1), test_pos.size(1))

        train_mp = create_time_sliced_graph(data, cutoffs.train)
        test_mp = create_time_sliced_graph(data, cutoffs.val)
        known_train = positives_by_user(train_pos)
        known_train_val = positives_by_user(torch.cat([train_pos, val_pos], dim=1))
        rng = np.random.default_rng(self.cfg.seed)
        known_all = positives_by_user(torch.cat(
            [data[et].edge_index for et in data.edge_types if et[0] == "user" and et[2] == "item"], dim=1))
        val_pos_loader = _loader(train_mp, val_pos, self.cfg, shuffle=False)
        val_edges, val_labels = roster_negatives(val_pos, rosters, listing_day, known_all, self.cfg.eval_neg_ratio, rng)
        val_loader = _loader(train_mp, val_edges, self.cfg, shuffle=False, labels=val_labels)
        test_edges, test_labels = roster_negatives(test_pos, rosters, listing_day, known_all, self.cfg.eval_neg_ratio, rng)
        test_loader = _loader(test_mp, test_edges, self.cfg, shuffle=False, labels=test_labels)

        generator = torch.Generator().manual_seed(self.cfg.seed)
        self.model.to(self.device)
        with torch.no_grad():
            self._forward(next(iter(val_pos_loader)).to(self.device))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )

        checkpoint = self.output_dir / "best_model.pt"
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss = self._epoch(self._epoch_loader(train_mp, train_pos, generator), known_train)
            val_loss = self._infonce(val_pos_loader, known_train_val)
            val_auc, val_ap = self.link_prediction(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "val_auc": val_auc, "val_ap": val_ap})
            logger.info("Epoch %03d train %.4f val %.4f val AUC %.4f val AP %.4f",
                        epoch, train_loss, val_loss, val_auc, val_ap)
            if val_loss < best - self.cfg.min_delta:
                best, stale = val_loss, 0
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                            "val_auc": val_auc, "val_ap": val_ap, "val_loss": val_loss}, checkpoint)
            else:
                stale += 1
                if stale >= self.cfg.patience:
                    break

        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        test_auc, test_ap = self.link_prediction(test_loader)
        logger.info("Best epoch %d | test AUC %.4f | test AP %.4f", state["epoch"], test_auc, test_ap)
        return {
            "best_epoch": int(state["epoch"]),
            "best_validation_auc": float(state["val_auc"]),
            "best_validation_ap": float(state["val_ap"]),
            "test_auc": test_auc,
            "test_ap": test_ap,
            "n_train_positives": int(train_pos.size(1)),
            "n_val_positives": int(val_pos.size(1)),
            "n_test_positives": int(test_pos.size(1)),
            "history": history,
        }
