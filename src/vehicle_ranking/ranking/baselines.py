"""Non-learned baselines for the full-roster evaluation: content affinity and a popularity rate."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from vehicle_ranking.data.export import ExportData, epoch_day

AFFINITY_HIGHER_BETTER = [
    "cat_affinity_brand", "cat_affinity_model_family", "cat_affinity_fuel_type",
    "cat_affinity_transmission", "cat_affinity_vehicle_type",
]
AFFINITY_LOWER_BETTER = ["price_fit", "age_fit"]


def content_affinity_score(pool: pd.DataFrame, keys: List[str]) -> np.ndarray:
    """Mean within-task percentile rank of the affinity and fit features, higher is better.

    na_option="top" gives a missing value the lowest rank number, that is, the worst percentile.
    """
    ranks = [pool.groupby(keys)[c].rank(pct=True, method="average", na_option="top", ascending=True)
             for c in AFFINITY_HIGHER_BETTER]
    ranks += [pool.groupby(keys)[c].rank(pct=True, method="average", na_option="top", ascending=False)
              for c in AFFINITY_LOWER_BETTER]
    return np.mean(np.column_stack([r.to_numpy() for r in ranks]), axis=1)


def _count_before(events: pd.DataFrame, query: pd.DataFrame, window_days: int) -> np.ndarray:
    """For every query row (group, epoch_day): events of the group in [epoch_day - window_days, epoch_day)."""
    daily = events.groupby(["group", "epoch_day"]).size().rename("n").reset_index().sort_values(["group", "epoch_day"])
    daily["epoch_day"] = daily["epoch_day"].astype(np.int64)
    daily["cum"] = daily.groupby("group")["n"].cumsum()

    def cumulative_until(day: pd.Series) -> np.ndarray:
        lookup = query[["group"]].assign(_day=day.to_numpy(np.int64) - 1, _row=np.arange(len(query))).sort_values("_day")
        merged = pd.merge_asof(lookup, daily.sort_values("epoch_day"), left_on="_day", right_on="epoch_day",
                               by="group", direction="backward")
        out = np.zeros(len(query))
        out[merged["_row"].to_numpy()] = merged["cum"].fillna(0.0).to_numpy()
        return out

    return cumulative_until(query["epoch_day"]) - cumulative_until(query["epoch_day"] - window_days)


def popularity_rate_scores(pool: pd.DataFrame, export: ExportData, window_days: int, shrinkage: float) -> np.ndarray:
    """Bids per listed vehicle of the model family in the preceding window, shrunk toward the pooled rate."""
    family = export.property_values("ModelFamily").rename("group")
    bids = export.edges["bid"][["i_idx", "ts"]].assign(epoch_day=lambda d: epoch_day(d["ts"]))
    bids = bids.assign(group=bids["i_idx"].map(family)).dropna(subset=["group"])
    listings = export.roster.assign(group=export.roster["i_idx"].map(family)).dropna(subset=["group"])
    query = pool[["i_idx", "epoch_day"]].assign(group=pool["i_idx"].map(family).fillna("__none"))

    group_bids = _count_before(bids, query, window_days)
    group_listings = _count_before(listings, query, window_days)
    all_query = query.assign(group="_all")
    pooled_bids = _count_before(bids.assign(group="_all"), all_query, window_days)
    pooled_listings = _count_before(listings.assign(group="_all"), all_query, window_days)
    pooled_rate = np.divide(pooled_bids, pooled_listings, out=np.zeros_like(pooled_bids), where=pooled_listings > 0)
    denominator = group_listings + shrinkage
    return np.divide(group_bids + shrinkage * pooled_rate, denominator,
                     out=np.zeros_like(group_bids), where=denominator > 0)
