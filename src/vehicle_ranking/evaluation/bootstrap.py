"""User-clustered paired bootstrap for task-level metrics."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def clustered_bootstrap(values: pd.DataFrame, cluster_col: str, value_cols: List[str],
                        n_resamples: int, seed: int) -> Dict[str, np.ndarray]:
    """Task-weighted mean of each column over clusters drawn with replacement; a drawn cluster brings all its tasks.

    All columns share the same draws, so differences between columns are paired.
    """
    per_cluster = values.groupby(cluster_col)[value_cols].sum()
    task_count = values.groupby(cluster_col).size().reindex(per_cluster.index).to_numpy(np.float64)
    sums = per_cluster.to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(task_count), size=(n_resamples, len(task_count)))
    weights = np.stack([np.bincount(d, minlength=len(task_count)) for d in draws]).astype(np.float64)
    totals = weights @ sums
    counts = weights @ task_count
    return {col: totals[:, j] / counts for j, col in enumerate(value_cols)}


def interval(samples: np.ndarray) -> Dict[str, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "excludes_zero": bool(low > 0 or high < 0)}
