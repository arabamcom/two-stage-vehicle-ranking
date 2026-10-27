"""User features computed from behavior up to a cutoff.

The same function produces the GNN user node features and the ranking user features, at T_emb for
training rows and at T_train for evaluation rows, so no feature uses an interaction that happens
after the cutoff it is attached to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vehicle_ranking.data.export import ExportData, epoch_day
from vehicle_ranking.utils import BEHAVIOR_EDGES, DAY_SECONDS

PREFERENCE_COLS = [
    "u_pref_price_real_median_log1p", "u_pref_price_real_iqr_log1p",
    "u_pref_age_years_median", "u_pref_age_years_iqr",
    "u_pref_damage_record_share", "u_pref_appraisal_severity_mean", "u_pref_price_index_brand_median",
]


def _hhi(values: pd.Series) -> float:
    return float((values.value_counts(normalize=True) ** 2).sum())


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def user_features(export: ExportData, cutoff: int) -> pd.DataFrame:
    """One row per user; users without behavior before the cutoff get zeros."""
    edges = {name: export.edges_until(name, cutoff) for name in BEHAVIOR_EDGES}
    brand = export.property_values("Brand")
    bid = edges["bid"].assign(brand=lambda d: d["i_idx"].map(brand))
    count = bid["count"] if "count" in bid.columns else pd.Series(1, index=bid.index)

    out = export.users[["u_idx"]].copy()
    static = [c for c in export.users.columns if c != "u_idx"]
    for col in static:
        out[col if col.startswith("u_") else f"u_{col}"] = export.users[col].to_numpy(np.float32)

    def per_user(series: pd.Series) -> pd.Series:
        return out["u_idx"].map(series).fillna(0.0)

    out["u_total_bids_log1p"] = np.log1p(per_user(count.groupby(bid["u_idx"]).sum()))
    for name in ("bid", "fav", "alert"):
        out[f"u_n_items_{name}_log1p"] = np.log1p(per_user(edges[name].groupby("u_idx").size()))
    decay = np.exp(-(cutoff - bid["ts"]) / (30 * DAY_SECONDS))
    out["u_bid_decay30_log1p"] = np.log1p(per_user(decay.groupby(bid["u_idx"]).sum()))
    active_days = pd.Series(epoch_day(bid["ts"]), index=bid.index).groupby(bid["u_idx"]).nunique()
    out["u_hist_days_active_log1p"] = np.log1p(per_user(active_days))
    out["u_bid_freq_per_active_day_log1p"] = np.log1p(per_user(bid.groupby("u_idx").size() / active_days))
    out["u_total_purchases_log1p"] = np.log1p(per_user(edges["purchase"].groupby("u_idx").size()))
    out["u_n_brands_log1p"] = np.log1p(per_user(bid.groupby("u_idx")["brand"].nunique()))
    out["u_brand_concentration_hhi"] = per_user(bid.dropna(subset=["brand"]).groupby("u_idx")["brand"].apply(_hhi))

    items = export.items
    committed = pd.concat([edges["bid"], edges["purchase"]])[["u_idx", "i_idx"]].drop_duplicates().merge(
        items[["i_idx", "item_price_real_log1p", "item_age_years"]], on="i_idx")
    grouped = committed.groupby("u_idx")
    out["u_pref_price_real_median_log1p"] = per_user(grouped["item_price_real_log1p"].median())
    out["u_pref_price_real_iqr_log1p"] = np.log1p(per_user(grouped["item_price_real_log1p"].apply(_iqr)))
    out["u_pref_age_years_median"] = per_user(grouped["item_age_years"].median())
    out["u_pref_age_years_iqr"] = per_user(grouped["item_age_years"].apply(_iqr))
    out["u_has_price_pref"] = out["u_idx"].isin(committed.dropna(subset=["item_price_real_log1p"])["u_idx"])
    out["u_has_age_pref"] = out["u_idx"].isin(committed.dropna(subset=["item_age_years"])["u_idx"])

    any_behavior = pd.concat([e[["u_idx", "i_idx"]] for e in edges.values()]).drop_duplicates().merge(
        items[["i_idx", "damage_record_has_value", "appraisal_severity_mean", "item_price_index_brand_log1p"]],
        on="i_idx")
    grouped = any_behavior.groupby("u_idx")
    out["u_pref_damage_record_share"] = per_user(grouped["damage_record_has_value"].mean())
    out["u_pref_appraisal_severity_mean"] = per_user(grouped["appraisal_severity_mean"].mean())
    out["u_pref_price_index_brand_median"] = per_user(grouped["item_price_index_brand_log1p"].median())
    return out.astype({c: np.float32 for c in out.columns if c != "u_idx"})


def users_with_behavior(export: ExportData, cutoff: int) -> set:
    return set(pd.concat([export.edges_until(n, cutoff)["u_idx"] for n in BEHAVIOR_EDGES]).unique().tolist())
