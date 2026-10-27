"""Full-roster evaluation: each retained positive is ranked against every vehicle of its day's roster."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from vehicle_ranking.evaluation.bootstrap import clustered_bootstrap, interval
from vehicle_ranking.evaluation.metrics import expected_hit_at_k, expected_ndcg_at_k, pool_positive_counts

KEYS = ["u_idx", "i_pos"]
METRICS = ("ndcg", "hit")


def system_task_values(pool: pd.DataFrame, scores: np.ndarray, k: int) -> pd.DataFrame:
    counts = pool_positive_counts(pool[KEYS + ["i_idx"]].assign(_s=scores), "_s", KEYS)
    return counts[KEYS].assign(ndcg=expected_ndcg_at_k(counts["n_greater"], counts["n_equal"], k),
                               hit=expected_hit_at_k(counts["n_greater"], counts["n_equal"], k))


def random_task_values(tasks: pd.DataFrame, k: int) -> pd.DataFrame:
    """Uniformly random order: every candidate ties with the positive."""
    n = tasks["pool_size"].to_numpy(np.int64)
    return tasks[KEYS].assign(ndcg=expected_ndcg_at_k(np.zeros(len(n), dtype=np.int64), n - 1, k),
                              hit=np.minimum(k / n, 1.0))


def evaluate_systems(pool: pd.DataFrame, system_scores: Dict[str, np.ndarray], comparisons: List[tuple],
                     k: int, n_resamples: int, seed: int) -> dict:
    """Segment means for every system and user-clustered paired intervals for the comparisons."""
    tasks = pool.loc[pool["i_idx"] == pool["i_pos"], KEYS + ["is_warm"]].merge(
        pool.groupby(KEYS).size().rename("pool_size").reset_index(), on=KEYS)
    tasks["is_warm"] = tasks["is_warm"].astype(bool)
    frames = {name: system_task_values(pool, s, k) for name, s in system_scores.items()}
    frames["random"] = random_task_values(tasks, k)
    per_task = tasks.copy()
    for name, frame in frames.items():
        per_task = per_task.merge(frame.rename(columns={m: f"{name}__{m}" for m in METRICS}), on=KEYS)

    segments = {"aggregate": per_task, "ranking_warm": per_task[per_task["is_warm"]],
                "ranking_cold": per_task[~per_task["is_warm"]]}
    report = {
        "n_tasks": int(len(per_task)),
        "mean_candidates": float(per_task["pool_size"].mean()),
        "segments": {s: {"n_tasks": int(len(f)), "n_users": int(f["u_idx"].nunique())} for s, f in segments.items()},
        "systems": {name: {s: {m: float(f[f"{name}__{m}"].mean()) for m in METRICS}
                           for s, f in segments.items() if len(f)} for name in frames},
        "comparisons": {},
    }
    for seg, frame in segments.items():
        if len(frame) == 0:
            continue
        samples = clustered_bootstrap(frame, "u_idx", [f"{n}__ndcg" for n in frames], n_resamples, seed)
        for a, b in comparisons:
            diff = samples[f"{a}__ndcg"] - samples[f"{b}__ndcg"]
            point = float((frame[f"{a}__ndcg"] - frame[f"{b}__ndcg"]).mean())
            report["comparisons"].setdefault(f"{a}_minus_{b}", {})[seg] = {"point": point, **interval(diff)}
    return report
