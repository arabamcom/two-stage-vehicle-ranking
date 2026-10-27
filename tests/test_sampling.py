from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from vehicle_ranking.config import RankingDatasetConfig
from vehicle_ranking.evaluation.metrics import pool_positive_counts
from vehicle_ranking.gnn.train import masked_infonce
from vehicle_ranking.graph.build import BID
from vehicle_ranking.ranking.baselines import content_affinity_score
from vehicle_ranking.ranking.dataset import evaluation_tasks, sample_negatives


def snapshot(n_users=2, n_items=60):
    return SimpleNamespace(user_emb=np.eye(n_users, 8), item_emb=np.eye(n_items, 8))


def test_negatives_are_distinct_come_from_the_day_roster_and_skip_known_pairs():
    cfg = RankingDatasetConfig(in_day_min_pool_size=5, neg_per_pos=4)
    rosters = {10: np.arange(0, 30), 11: np.arange(30, 60)}
    pos = pd.DataFrame({"u_idx": [0, 0, 0, 1], "i_idx": [3, 4, 35, 4], "label": [5, 5, 5, 2],
                        "epoch_day": [10, 10, 11, 10]})
    known = {0: {3, 4, 35, 7, 8, 9}, 1: {4}}
    neg = sample_negatives(pos, "train", known, rosters, cfg, np.random.default_rng(0), snapshot())

    assert len(neg) == 16
    assert not neg.duplicated(["u_idx", "epoch_day", "i_idx"]).any()
    for row in neg.itertuples():
        assert row.i_idx not in known[row.u_idx]
        assert row.i_idx in rosters[row.epoch_day]


def test_evaluation_tasks_drop_the_users_other_positives():
    rosters = {10: np.arange(0, 10)}
    pos = pd.DataFrame({"u_idx": [0, 0], "i_idx": [2, 5], "epoch_day": [10, 10]})
    tasks = evaluation_tasks(pos, rosters, {0: {7}}, {0}, np.random.default_rng(0), 0, 5, 0.85, "holdout_pool")
    first = tasks[tasks["i_pos"] == 2]
    assert set(first["i_idx"]) == set(range(10)) - {5, 7}
    assert first["label"].sum() == 1


def test_missing_affinity_values_rank_lowest():
    pool = pd.DataFrame({"u_idx": 0, "i_pos": 1, "i_idx": [1, 2],
                         "cat_affinity_brand": [np.nan, 0.5], "cat_affinity_model_family": [0.0, 0.0],
                         "cat_affinity_fuel_type": [0.0, 0.0], "cat_affinity_transmission": [0.0, 0.0],
                         "cat_affinity_vehicle_type": [0.0, 0.0], "price_fit": [np.nan, 0.1], "age_fit": [1.0, 1.0]})
    score = content_affinity_score(pool, ["u_idx", "i_pos"])
    assert np.isfinite(score).all() and score[0] < score[1]


def test_non_finite_scores_are_rejected():
    pool = pd.DataFrame({"u_idx": 0, "i_pos": 1, "i_idx": [1, 2], "s": [np.nan, 0.3]})
    with pytest.raises(ValueError):
        pool_positive_counts(pool, "s", ["u_idx", "i_pos"])


def test_infonce_masks_other_known_positives_of_the_same_user():
    user = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    item = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    batch = {
        BID: SimpleNamespace(edge_label_index=torch.tensor([[0, 0, 1], [0, 1, 2]])),
        "user": SimpleNamespace(n_id=torch.tensor([10, 11])),
        "item": SimpleNamespace(n_id=torch.tensor([20, 21, 22])),
    }
    loss_masked, b = masked_infonce({"user": user, "item": item}, batch, {10: {20, 21}}, torch.tensor(0.15))
    loss_plain, _ = masked_infonce({"user": user, "item": item}, batch, {}, torch.tensor(0.15))
    assert b == 3 and torch.isfinite(loss_masked)
    assert not torch.isclose(loss_masked, loss_plain)
