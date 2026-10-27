"""LightGBM LambdaRank training on user-day (or user-only) query groups."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

from vehicle_ranking.config import RankingTrainingConfig

logger = logging.getLogger(__name__)

QUERY_GROUP_KEYS = {"user": ["u_idx"], "user_day": ["u_idx", "epoch_day"]}


def group_key_columns(query_group: str) -> List[str]:
    if query_group not in QUERY_GROUP_KEYS:
        raise ValueError(f"query_group must be one of {sorted(QUERY_GROUP_KEYS)}, got {query_group!r}")
    return list(QUERY_GROUP_KEYS[query_group])


def shuffle_within_groups(df: pd.DataFrame, keys: List[str], seed: int) -> pd.DataFrame:
    """Seeded random row order inside each group, so tied scores do not favor positives."""
    tie_break = np.random.default_rng(seed).permutation(len(df))
    return df.assign(_t=tie_break).sort_values(keys + ["_t"]).drop(columns="_t").reset_index(drop=True)


def load_splits(dataset_path: Path, feat_cols: List[str], cfg: RankingTrainingConfig) -> Dict[str, dict]:
    """Per split: features, labels, query sizes, weights and the ordered frame."""
    keys = group_key_columns(cfg.query_group)
    df = pd.read_parquet(dataset_path)
    out = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].sort_values(keys + ["label"], ascending=[True] * len(keys) + [False])
        sub = shuffle_within_groups(sub[sub.groupby(keys).cumcount() < cfg.max_query_rows], keys, cfg.seed)
        out[split] = {
            "X": sub[feat_cols].to_numpy(np.float32),
            "y": sub["label"].to_numpy(np.float32),
            "groups": sub.groupby(keys, sort=False).size().to_numpy(),
            "weight": sub["query_weight"].to_numpy(np.float32),
            "frame": sub,
        }
        logger.info("[%s] rows=%d queries=%d", split, len(sub), len(out[split]["groups"]))
    return out


def train_lambdarank(splits: Dict[str, dict], feat_cols: List[str], cfg: RankingTrainingConfig) -> lgb.Booster:
    params = dict(cfg.lgb_params, seed=cfg.seed)
    train_set = lgb.Dataset(splits["train"]["X"], label=splits["train"]["y"], group=splits["train"]["groups"],
                            weight=splits["train"]["weight"], feature_name=feat_cols, free_raw_data=False)
    val_set = lgb.Dataset(splits["val"]["X"], label=splits["val"]["y"], group=splits["val"]["groups"],
                          feature_name=feat_cols, reference=train_set)
    return lgb.train(
        params, train_set,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        num_boost_round=cfg.num_boost_round,
        callbacks=[lgb.early_stopping(cfg.early_stopping), lgb.log_evaluation(cfg.log_period)],
    )
