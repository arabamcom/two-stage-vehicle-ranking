"""LambdaRank dataset: positives, sampled non-interactions and full-roster evaluation tasks.

For training cutoff T_train, embedding cutoff T_emb = T_train - W days and cutoffs from
cutoff_spec.json, the splits are
    train    (T_emb, T_train]                   features as of T_emb
    val      (T_train, T_val]                   features as of T_train
    test     (T_val, T_test - holdout_days]     features as of T_train
    holdout  (T_test - holdout_days, T_test]    features as of T_train, evaluation only
A row's auction day is the listing day of its vehicle in the decision-time roster.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

from vehicle_ranking.config import RankingDatasetConfig
from vehicle_ranking.data.export import ExportData
from vehicle_ranking.gnn.export import EMBEDDING_SNAPSHOTS
from vehicle_ranking.ranking.features import Snapshot, build_snapshot, item_categories, join_features
from vehicle_ranking.utils import BEHAVIOR_EDGES, DAY_SECONDS, Cutoffs

logger = logging.getLogger(__name__)

EVAL_KEYS = ["u_idx", "i_idx", "label", "i_pos", "is_warm", "epoch_day"]


@dataclass(frozen=True)
class RankingWindows:
    train_start: int
    train_end: int
    val_end: int
    test_end: int
    holdout_end: int

    @classmethod
    def from_cutoffs(cls, cutoffs: Cutoffs, window_days: int, holdout_days: int) -> "RankingWindows":
        return cls(
            train_start=cutoffs.train - window_days * DAY_SECONDS,
            train_end=cutoffs.train,
            val_end=cutoffs.val,
            test_end=cutoffs.test - holdout_days * DAY_SECONDS,
            holdout_end=cutoffs.test,
        )


def get_positives(export: ExportData, label_map: Dict[str, int], split: str, start: int, end: int) -> pd.DataFrame:
    """Interactions in (start, end]; a pair with several behaviors keeps its highest label and latest ts.

    The auction day of a positive is the listing day of its vehicle; vehicles never listed are dropped.
    """
    frames = [export.edges_between(name, start, end)[["u_idx", "i_idx", "ts"]].assign(label=label)
              for name, label in label_map.items()]
    pos = pd.concat(frames).groupby(["u_idx", "i_idx"], as_index=False).agg(label=("label", "max"), ts=("ts", "max"))
    pos["epoch_day"] = pos["i_idx"].map(export.listing_day())
    return pos.dropna(subset=["epoch_day"]).astype({"epoch_day": np.int64}).assign(split=split)


def cap_positives(pos: pd.DataFrame, cfg: RankingDatasetConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Keep at most max_pos_per_user positives per user, sampled with probability proportional to label priority."""
    kept = []
    for _, group in pos.groupby("u_idx"):
        if len(group) > cfg.max_pos_per_user:
            weights = group["label"].map(cfg.label_priority).to_numpy(float)
            group = group.iloc[rng.choice(len(group), cfg.max_pos_per_user, replace=False, p=weights / weights.sum())]
        kept.append(group)
    return pd.concat(kept, ignore_index=True)


def known_pairs_until(export: ExportData, cutoff: int) -> Dict[int, Set[int]]:
    """Every (user, vehicle) pair with any behavior up to the cutoff."""
    pairs = pd.concat([export.edges_until(n, cutoff)[["u_idx", "i_idx"]] for n in BEHAVIOR_EDGES]).drop_duplicates()
    return {int(u): set(g.tolist()) for u, g in pairs.groupby("u_idx")["i_idx"]}


def candidate_pool(rosters: Dict[int, np.ndarray], day: int, min_size: int) -> np.ndarray:
    """The day's roster, extended with the adjacent days when it has fewer than min_size vehicles."""
    pool = rosters.get(day, np.empty(0, dtype=np.int64))
    if len(pool) >= min_size:
        return pool
    near = [rosters[d] for d in (day - 1, day + 1) if d in rosters]
    return np.unique(np.concatenate([pool] + near)) if near else pool


def sample_negatives(pos: pd.DataFrame, split: str, known: Dict[int, Set[int]], rosters: Dict[int, np.ndarray],
                     cfg: RankingDatasetConfig, rng: np.random.Generator, snap: Snapshot) -> pd.DataFrame:
    """For each user-day with n positives, draw n * neg_per_pos distinct vehicles from that day's roster.

    Vehicles the user interacted with up to the split cutoff are excluded; when fewer remain, all
    are used. These rows are reference non-interactions, not verified dislikes. With
    hard_neg_ratio > 0, that share is taken as the candidates closest to the user in embedding space.
    """
    rows = []
    for (u, day), group in pos.groupby(["u_idx", "epoch_day"]):
        pool = candidate_pool(rosters, int(day), cfg.in_day_min_pool_size)
        user_known = known.get(int(u), set())
        candidates = pool[~np.isin(pool, np.fromiter(user_known, dtype=np.int64, count=len(user_known)))]
        n_neg = min(max(len(group) * cfg.neg_per_pos, cfg.min_neg_per_query), len(candidates))
        if n_neg == 0:
            continue
        n_hard = int(round(n_neg * cfg.hard_neg_ratio))
        hard = candidates[np.argsort(-(snap.item_emb[candidates] @ snap.user_emb[u]))[:n_hard]] if n_hard else np.empty(0, np.int64)
        rest = candidates[~np.isin(candidates, hard)]
        picks = rng.choice(rest, size=n_neg - len(hard), replace=False)
        for i, kind in [(i, "hard") for i in hard] + [(i, "random") for i in picks]:
            rows.append((int(u), int(i), 0, split, kind, int(day)))
    return pd.DataFrame(rows, columns=["u_idx", "i_idx", "label", "split", "neg_type", "epoch_day"])


def evaluation_tasks(pos: pd.DataFrame, rosters: Dict[int, np.ndarray], known_before: Dict[int, Set[int]],
                     train_users: Set[int], rng: np.random.Generator, max_positives: int, min_pool_size: int,
                     warm_fraction: float, split: str) -> pd.DataFrame:
    """One task per positive: the positive against every other vehicle of its day's roster.

    The user's other positives of the window and vehicles the user interacted with before T_train
    are removed from the candidates, so a perfect ranker reaches NDCG 1. max_positives > 0 samples
    that many positives, warm_fraction of them from ranking-warm users (users with LambdaRank
    training positives).
    """
    pos = pos.assign(is_warm=pos["u_idx"].isin(train_users))
    window_pos = pos.groupby("u_idx")["i_idx"].apply(set).to_dict()
    if max_positives > 0 and len(pos) > max_positives:
        warm, cold = pos[pos["is_warm"]], pos[~pos["is_warm"]]
        n_cold = min(int(max_positives * (1.0 - warm_fraction)), len(cold))
        n_warm = min(max_positives - n_cold, len(warm))
        pos = pd.concat([warm.sample(n_warm, random_state=int(rng.integers(2**31 - 1))),
                         cold.sample(n_cold, random_state=int(rng.integers(2**31 - 1)))])

    rows = []
    for r in pos.itertuples(index=False):
        pool = rosters.get(int(r.epoch_day))
        if pool is None or len(pool) < min_pool_size:
            continue
        excluded = (window_pos.get(r.u_idx, set()) | known_before.get(r.u_idx, set())) - {r.i_idx}
        for i in sorted((set(pool.tolist()) | {int(r.i_idx)}) - excluded):
            rows.append((int(r.u_idx), i, int(i == r.i_idx), int(r.i_idx), bool(r.is_warm), int(r.epoch_day)))
    return pd.DataFrame(rows, columns=EVAL_KEYS).assign(split=split)


def add_query_weights(dataset: pd.DataFrame, pos_train: pd.DataFrame, train_cutoff: int, half_life_days: int) -> pd.DataFrame:
    """Training rows weighted by 2^(-days since the user's last training positive / half-life), floored at 0.1."""
    days_ago = (train_cutoff - pos_train.groupby("u_idx")["ts"].max()) / DAY_SECONDS
    weight = np.power(2.0, -days_ago / half_life_days).clip(lower=0.1)
    dataset["query_weight"] = 1.0
    train = dataset["split"] == "train"
    dataset.loc[train, "query_weight"] = dataset.loc[train, "u_idx"].map(weight).fillna(0.1)
    return dataset


def build_ranking_dataset(export: ExportData, gnn_run: Path, window_days: int,
                          cfg: RankingDatasetConfig, out_dir: Path) -> dict:
    """Build and save the training dataset and the test and holdout evaluation tasks."""
    rng = np.random.default_rng(cfg.seed)
    windows = RankingWindows.from_cutoffs(export.cutoffs, window_days, cfg.holdout_days)
    rosters = export.daily_rosters()
    categories, cat_cols = item_categories(export, cfg.prop_types_to_keep)
    snapshots = {
        "train_rows": build_snapshot(export, windows.train_start, cfg, categories, cat_cols,
                                     gnn_run / EMBEDDING_SNAPSHOTS["train_rows"]),
        "eval_rows": build_snapshot(export, windows.train_end, cfg, categories, cat_cols,
                                    gnn_run / EMBEDDING_SNAPSHOTS["eval_rows"]),
    }

    pos_train_raw = get_positives(export, cfg.label_map, "train", windows.train_start, windows.train_end)
    pos_train = cap_positives(pos_train_raw, cfg, rng)
    pos_val = get_positives(export, cfg.label_map, "val", windows.train_end, windows.val_end)
    pos_test = get_positives(export, cfg.label_map, "test", windows.val_end, windows.test_end)
    pos_holdout = get_positives(export, cfg.label_map, "holdout", windows.test_end, windows.holdout_end)

    parts = []
    for pos, split, cutoff, snap in ((pos_train, "train", windows.train_end, "train_rows"),
                                     (pos_val, "val", windows.val_end, "eval_rows"),
                                     (pos_test, "test", windows.test_end, "eval_rows")):
        neg = sample_negatives(pos, split, known_pairs_until(export, cutoff), rosters, cfg, rng, snapshots[snap])
        parts.append(pd.concat([pos.drop(columns="ts"), neg]).assign(snapshot=snap))

    train_users = set(pos_train["u_idx"].tolist())
    known_before = known_pairs_until(export, windows.train_end)
    tasks = [
        evaluation_tasks(pos_test, rosters, known_before, train_users, rng, cfg.max_test_pool_positives,
                         cfg.pool_eval_min_size, cfg.test_pool_warm_fraction, "test_pool"),
        evaluation_tasks(pos_holdout, rosters, known_before, train_users, rng, cfg.max_holdout_pool_positives,
                         cfg.pool_eval_min_size, cfg.test_pool_warm_fraction, "holdout_pool"),
    ]
    dataset = pd.concat(parts + [t.assign(snapshot="eval_rows") for t in tasks], ignore_index=True)
    dataset = add_query_weights(dataset, pos_train_raw, windows.train_end, cfg.recency_half_life_days)
    dataset, feat_cols = join_features(dataset, export, cfg, snapshots, categories, cat_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    main = dataset[dataset["split"].isin(["train", "val", "test"])]
    main[["u_idx", "i_idx", "label", "split", "epoch_day", "query_weight", "neg_type"] + feat_cols].to_parquet(
        out_dir / "ranking_dataset.parquet", index=False)
    for split in ("test_pool", "holdout_pool"):
        dataset.loc[dataset["split"] == split, EVAL_KEYS + feat_cols].to_parquet(out_dir / f"{split}.parquet", index=False)

    summary = {
        "feature_cols": feat_cols,
        "n_features": len(feat_cols),
        "window_days": window_days,
        "windows": windows.__dict__,
        "rows_by_split": main["split"].value_counts().to_dict(),
        "n_test_pool_tasks": int(tasks[0].groupby(["u_idx", "i_pos"]).ngroups) if len(tasks[0]) else 0,
        "n_holdout_pool_tasks": int(tasks[1].groupby(["u_idx", "i_pos"]).ngroups) if len(tasks[1]) else 0,
        "n_train_users_capped": int((pos_train_raw.groupby("u_idx").size() > cfg.max_pos_per_user).sum()),
    }
    logger.info("Ranking dataset: %s", {k: v for k, v in summary.items() if k != "feature_cols"})
    return summary
