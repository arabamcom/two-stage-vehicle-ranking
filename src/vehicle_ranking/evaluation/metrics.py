"""Ranking metrics for sampled query lists and for single-positive full-pool tasks."""
from __future__ import annotations

from math import comb
from typing import Dict, List

import numpy as np
import pandas as pd


def dcg_at_k(rel: np.ndarray, k: int) -> float:
    rel = rel[:k]
    return float(np.sum((2.0 ** rel - 1.0) / np.log2(np.arange(len(rel)) + 2.0)))


def ndcg_at_k(rel: np.ndarray, scores: np.ndarray, k: int) -> float:
    ideal = dcg_at_k(np.sort(rel)[::-1], k)
    return dcg_at_k(rel[np.argsort(-scores, kind="stable")], k) / ideal if ideal > 0 else 0.0


def mrr(rel: np.ndarray, scores: np.ndarray) -> float:
    hits = np.where(rel[np.argsort(-scores, kind="stable")] > 0)[0]
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def hit_at_k(rel: np.ndarray, scores: np.ndarray, k: int) -> float:
    return float(np.any(rel[np.argsort(-scores, kind="stable")[:k]] > 0))


def precision_at_k(rel: np.ndarray, scores: np.ndarray, k: int) -> float:
    top = rel[np.argsort(-scores, kind="stable")[: min(k, len(rel))]]
    return float(np.sum(top > 0)) / min(k, len(rel))


def recall_at_k(rel: np.ndarray, scores: np.ndarray, k: int) -> float:
    n_pos = int(np.sum(rel > 0))
    top = rel[np.argsort(-scores, kind="stable")[: min(k, len(rel))]]
    return float(np.sum(top > 0)) / n_pos if n_pos else 0.0


def random_hit_at_k(n: int, p: int, k: int) -> float:
    """Exact Hit@k of a uniformly random order of n candidates with p positives."""
    k = min(k, n)
    return 1.0 - (comb(n - p, k) / comb(n, k) if n - p >= k else 0.0)


def random_ndcg_at_k(rel: np.ndarray, k: int) -> float:
    """Expected NDCG@k of a uniformly random order: every position has the mean gain of the list."""
    ideal = dcg_at_k(np.sort(rel)[::-1], k)
    if ideal == 0:
        return 0.0
    k = min(k, len(rel))
    return float(np.mean(2.0 ** rel - 1.0) * np.sum(1.0 / np.log2(np.arange(k) + 2.0)) / ideal)


def sampled_list_metrics(y: np.ndarray, scores: np.ndarray, groups: np.ndarray, k: int = 10) -> Dict[str, float]:
    """Mean metrics over query groups that contain a positive and a non-positive, with analytic random controls."""
    totals: Dict[str, float] = {}
    n_groups, offset = 0, 0
    for size in groups:
        rel, s = y[offset: offset + size], scores[offset: offset + size]
        offset += size
        if rel.max() == 0 or rel.min() == rel.max():
            continue
        n_groups += 1
        p = int(np.sum(rel > 0))
        values = {
            f"ndcg@{k}": ndcg_at_k(rel, s, k), f"precision@{k}": precision_at_k(rel, s, k),
            f"recall@{k}": recall_at_k(rel, s, k), f"hit@{k}": hit_at_k(rel, s, k), "mrr": mrr(rel, s),
            f"random_ndcg@{k}": random_ndcg_at_k(rel, k), f"random_precision@{k}": p / len(rel),
            f"random_hit@{k}": random_hit_at_k(len(rel), p, k),
        }
        for key, v in values.items():
            totals[key] = totals.get(key, 0.0) + v
    out = {key: v / max(n_groups, 1) for key, v in totals.items()}
    out["n_groups"] = n_groups
    return out


def pool_positive_counts(pool: pd.DataFrame, score_col: str, keys: List[str]) -> pd.DataFrame:
    """Per task: pool vehicles scored above (n_greater) and equal to (n_equal) the positive."""
    if not np.isfinite(pool[score_col].to_numpy(float)).all():
        raise ValueError(f"{score_col} contains non-finite scores; every candidate needs a comparable score")
    is_pos = pool["i_idx"].values == pool["i_pos"].values
    pos = pool.loc[is_pos, keys + [score_col]].rename(columns={score_col: "_pos"})
    others = pool.loc[~is_pos, keys + [score_col]].merge(pos, on=keys, how="inner")
    grouped = [others[k] for k in keys]
    counts = pd.DataFrame({
        "n_greater": (others[score_col] > others["_pos"]).groupby(grouped).sum(),
        "n_equal": (others[score_col] == others["_pos"]).groupby(grouped).sum(),
    }).reset_index()
    return pos[keys].merge(counts, on=keys, how="left").fillna({"n_greater": 0, "n_equal": 0}).astype(
        {"n_greater": np.int64, "n_equal": np.int64}
    )


def expected_ndcg_at_k(n_greater: np.ndarray, n_equal: np.ndarray, k: int) -> np.ndarray:
    """NDCG@k of a single relevant vehicle averaged over a uniformly random order of its ties."""
    n_greater = np.asarray(n_greater, dtype=np.int64)
    n_equal = np.asarray(n_equal, dtype=np.int64)
    cumulative = np.cumsum(np.concatenate([[0.0], 1.0 / np.log2(np.arange(1, k + 1) + 1.0)]))
    upto = lambda r: cumulative[np.minimum(r, k)]  # noqa: E731
    return (upto(n_greater + n_equal + 1) - upto(n_greater)) / (n_equal + 1)


def expected_hit_at_k(n_greater: np.ndarray, n_equal: np.ndarray, k: int) -> np.ndarray:
    return np.clip((k - np.asarray(n_greater)) / (np.asarray(n_equal) + 1), 0.0, 1.0)
