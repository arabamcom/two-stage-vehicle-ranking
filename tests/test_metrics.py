from itertools import permutations

import numpy as np

from vehicle_ranking.evaluation.metrics import (
    expected_ndcg_at_k,
    ndcg_at_k,
    random_hit_at_k,
    random_ndcg_at_k,
)


def test_expected_ndcg_without_ties_is_reciprocal_log_rank():
    values = expected_ndcg_at_k(np.array([0, 4, 9, 10]), np.zeros(4, dtype=int), 10)
    np.testing.assert_allclose(values, [1.0, 1 / np.log2(6), 1 / np.log2(11), 0.0])


def test_expected_ndcg_averages_over_tied_positions():
    value = expected_ndcg_at_k(np.array([1]), np.array([2]), 10)[0]
    np.testing.assert_allclose(value, np.mean([1 / np.log2(r + 1) for r in (2, 3, 4)]))


def test_random_hit_matches_enumeration():
    rel = np.array([1, 1, 0, 0, 0, 0])
    k = 3
    hits = [float(np.any(rel[list(p)[:k]] > 0)) for p in permutations(range(len(rel)))]
    np.testing.assert_allclose(random_hit_at_k(len(rel), 2, k), np.mean(hits))
    assert random_hit_at_k(8, 1, 10) == 1.0


def test_random_ndcg_matches_enumeration():
    rel = np.array([5.0, 2.0, 0.0, 0.0, 0.0])
    values = []
    for p in permutations(range(len(rel))):
        scores = np.empty(len(rel))
        scores[list(p)] = -np.arange(len(rel))
        values.append(ndcg_at_k(rel, scores, 3))
    np.testing.assert_allclose(random_ndcg_at_k(rel, 3), np.mean(values))
