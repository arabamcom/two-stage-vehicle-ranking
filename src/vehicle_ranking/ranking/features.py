"""Ranking features attached to (user, vehicle) rows.

Rows belong to one of two snapshots. Training rows use everything as of T_emb, evaluation rows
(validation, test, holdout) everything as of T_train: user history and preferences, categorical
affinities and graph embeddings. No row therefore sees its own target interaction, and training
and evaluation rows are computed by the same rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from vehicle_ranking.config import RankingDatasetConfig
from vehicle_ranking.data.export import ExportData
from vehicle_ranking.features.users import user_features, users_with_behavior
from vehicle_ranking.utils import BEHAVIOR_EDGES

CROSS_FEAT = [
    "price_fit", "age_fit", "price_zscore", "age_zscore",
    "damage_record_tolerance_match", "appraisal_severity_fit", "brand_price_fit",
]
POOL_FEAT = [
    "i_price_pct_in_day", "i_age_pct_in_day", "i_price_z_in_day", "i_age_z_in_day",
    "i_brand_share_in_day", "i_n_items_in_day",
]
APPRAISAL_PREFIX = "i_appraisal_"


@dataclass
class Snapshot:
    """Everything a row may use, as of one cutoff."""
    cutoff: int
    users: pd.DataFrame
    affinity: Dict[str, pd.DataFrame]
    user_emb: np.ndarray
    item_emb: np.ndarray
    emb_users: Set[int]


def load_embeddings(emb_dir) -> Tuple[np.ndarray, np.ndarray]:
    user = np.load(emb_dir / "user_embeddings.npy")
    item = np.load(emb_dir / "item_embeddings.npy")
    user = user / (np.linalg.norm(user, axis=1, keepdims=True) + 1e-8)
    item = item / (np.linalg.norm(item, axis=1, keepdims=True) + 1e-8)
    return user, item


def item_categories(export: ExportData, prop_types: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """Label-encoded categorical property per vehicle, one column per kept property type."""
    split = export.props["p_key"].str.split("::", n=1)
    props = export.props.assign(prop_type=split.str[0], prop_value=split.str[1])
    props = props[props["prop_type"].isin(prop_types)]
    wide = export.item_props.merge(props[["p_idx", "prop_type", "prop_value"]], on="p_idx").pivot_table(
        index="i_idx", columns="prop_type", values="prop_value", aggfunc="first")
    wide.columns = [prop_types[c] for c in wide.columns]
    cat_cols = list(wide.columns)
    for col in cat_cols:
        wide[col] = pd.Categorical(wide[col]).codes
        wide[col] = wide[col].replace(-1, np.nan)
    return wide.reset_index(), cat_cols


def category_affinity(export: ExportData, cutoff: int, label_map: Dict[str, int],
                      categories: pd.DataFrame, cat_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Share of a user's label-weighted behavior up to the cutoff that falls on each category value."""
    pairs = pd.concat([export.edges_until(name, cutoff)[["u_idx", "i_idx"]].assign(label=label_map[name])
                       for name in BEHAVIOR_EDGES])
    pairs = pairs.groupby(["u_idx", "i_idx"], as_index=False)["label"].max().merge(categories, on="i_idx")
    total = pairs.groupby("u_idx")["label"].sum().astype(float).rename("total")
    tables = {}
    for col in cat_cols:
        cat = pairs.dropna(subset=[col]).groupby(["u_idx", col])["label"].sum().astype(float).rename("weight")
        table = cat.reset_index().merge(total.reset_index(), on="u_idx")
        tables[col] = table.assign(share=table["weight"] / table["total"])[["u_idx", col, "share"]]
    return tables


def build_snapshot(export: ExportData, cutoff: int, cfg: RankingDatasetConfig, categories: pd.DataFrame,
                   cat_cols: List[str], emb_dir) -> Snapshot:
    user_emb, item_emb = load_embeddings(emb_dir)
    return Snapshot(
        cutoff=cutoff,
        users=user_features(export, cutoff),
        affinity=category_affinity(export, cutoff, cfg.label_map, categories, cat_cols),
        user_emb=user_emb,
        item_emb=item_emb,
        emb_users=users_with_behavior(export, cutoff),
    )


def daily_pool_features(export: ExportData, min_pool_size: int) -> pd.DataFrame:
    """Position of each vehicle within the roster of its listing day, known at decision time."""
    day = export.listing_day().rename("auction_day").reset_index()
    df = day.merge(export.items[["i_idx", "item_price_real_log1p", "item_age_years"]], on="i_idx", how="left")
    df["brand"] = df["i_idx"].map(export.property_values("Brand"))
    df["_n"] = df.groupby("auction_day")["i_idx"].transform("count")
    for col, pct, z in (("item_price_real_log1p", "i_price_pct_in_day", "i_price_z_in_day"),
                        ("item_age_years", "i_age_pct_in_day", "i_age_z_in_day")):
        grouped = df.groupby("auction_day")[col]
        df[pct] = grouped.rank(pct=True).astype(np.float32)
        iqr = (grouped.transform(lambda s: s.quantile(0.75)) - grouped.transform(lambda s: s.quantile(0.25))).clip(lower=1e-4)
        df[z] = ((df[col] - grouped.transform("median")) / iqr).clip(-5, 5).astype(np.float32)
    df["i_brand_share_in_day"] = (df.groupby(["auction_day", "brand"])["i_idx"].transform("count") / df["_n"]).fillna(0.0)
    df["i_n_items_in_day"] = np.log1p(df["_n"]).astype(np.float32)
    small = df["_n"] < min_pool_size
    df.loc[small, ["i_price_pct_in_day", "i_age_pct_in_day"]] = 0.5
    df.loc[small, ["i_price_z_in_day", "i_age_z_in_day", "i_brand_share_in_day"]] = 0.0
    return df[["i_idx"] + POOL_FEAT]


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    df["price_fit"] = (df["u_pref_price_real_median_log1p"] - df["i_item_price_real_log1p"]).abs()
    df["age_fit"] = (df["u_pref_age_years_median"] - df["i_item_age_years"]).abs()
    df["price_zscore"] = ((df["i_item_price_real_log1p"] - df["u_pref_price_real_median_log1p"])
                          / df["u_pref_price_real_iqr_log1p"].clip(lower=1e-4)).clip(-5, 5)
    df["age_zscore"] = ((df["i_item_age_years"] - df["u_pref_age_years_median"])
                        / df["u_pref_age_years_iqr"].clip(lower=1e-4)).clip(-5, 5)
    df["damage_record_tolerance_match"] = (df["i_damage_record_has_value"] - df["u_pref_damage_record_share"]).abs()
    df["appraisal_severity_fit"] = (df["i_appraisal_severity_mean"] - df["u_pref_appraisal_severity_mean"]).abs()
    df["brand_price_fit"] = (df["i_item_price_index_brand_log1p"] - df["u_pref_price_index_brand_median"]).abs()
    return df


def _snapshot_features(rows: pd.DataFrame, snap: Snapshot, cat_cols: List[str], with_embeddings: bool) -> pd.DataFrame:
    rows = rows.merge(snap.users, on="u_idx", how="left")
    for col in cat_cols:
        name = "cat_affinity_" + col.replace("i_", "", 1)
        rows = rows.merge(snap.affinity[col].rename(columns={"share": name}), on=["u_idx", col], how="left")
        rows[name] = rows[name].fillna(0.0).astype(np.float32)
    if with_embeddings:
        u = snap.user_emb[rows["u_idx"].to_numpy()]
        i = snap.item_emb[rows["i_idx"].to_numpy()]
        rows["emb_cosine_sim"] = (u * i).sum(axis=1).astype(np.float32)
        rows["emb_l2_dist"] = np.linalg.norm(u - i, axis=1).astype(np.float32)
        without = ~rows["u_idx"].isin(snap.emb_users)
        rows.loc[without, ["emb_cosine_sim", "emb_l2_dist"]] = 0.0
    return rows


def join_features(rows: pd.DataFrame, export: ExportData, cfg: RankingDatasetConfig, snapshots: Dict[str, Snapshot],
                  categories: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """rows need u_idx, i_idx and a 'snapshot' column naming the entry of snapshots to use."""
    items = export.items.rename(columns={c: c if c.startswith("i_") else f"i_{c}" for c in export.items.columns if c != "i_idx"})
    rows = (rows.reset_index(drop=True).assign(_row=lambda d: np.arange(len(d)))
            .merge(items, on="i_idx", how="left")
            .merge(daily_pool_features(export, cfg.daily_pool_feature_min_size), on="i_idx", how="left")
            .merge(categories, on="i_idx", how="left"))
    for col in POOL_FEAT:
        rows[col] = rows[col].fillna(0.5 if col.endswith("_pct_in_day") else 0.0).astype(np.float32)

    parts = [_snapshot_features(group, snapshots[name], cat_cols, bool(cfg.emb_feat))
             for name, group in rows.groupby("snapshot", sort=False)]
    rows = pd.concat(parts).sort_values("_row").drop(columns="_row").reset_index(drop=True)
    rows = add_cross_features(rows)

    affinity_cols = ["cat_affinity_" + c.replace("i_", "", 1) for c in cat_cols]
    feat_cols = (
        [c for c in cfg.user_feat if c in rows.columns]
        + [c for c in cfg.item_feat if c in rows.columns]
        + [c for c in rows.columns if c.startswith(APPRAISAL_PREFIX)]
        + CROSS_FEAT + POOL_FEAT + affinity_cols
        + [c for c in cfg.emb_feat if c in rows.columns]
    )
    rows[feat_cols] = rows[feat_cols].astype(np.float32)
    return rows, feat_cols
