"""Property nodes: categorical vehicle attributes and discretized price and age.

Price buckets are quintiles of the real (deflated) price, computed only on vehicles listed up to
the training cutoff. Age buckets use fixed cut points. A vehicle without a price or an age is
linked to the dedicated missing-value node of that property.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from vehicle_ranking.config import GraphConfig

CATEGORICAL_COLUMNS = {
    "Brand": "brand",
    "FuelType": "fuel_type",
    "Transmission": "transmission",
    "VehicleType": "vehicle_type",
}


@dataclass
class PropertyGraph:
    nodes: pd.DataFrame  # p_key, p_idx
    edges: pd.DataFrame  # i_idx, p_idx
    bucket_spec: Dict[str, dict]


def _categorical_pairs(vehicles: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    blocks = {}
    for prefix, col in CATEGORICAL_COLUMNS.items():
        values = vehicles[["i_idx", col]].dropna()
        values = values.assign(v=values[col].astype(str).str.strip().str.lower())
        values = values[values["v"] != ""]
        blocks[prefix] = pd.DataFrame({"i_idx": values["i_idx"], "p_key": prefix + "::" + values["v"]})
    family = vehicles[["i_idx", "brand", "model_family"]].dropna()
    blocks["ModelFamily"] = pd.DataFrame({
        "i_idx": family["i_idx"],
        "p_key": "ModelFamily::" + family["brand"].str.lower() + "::" + family["model_family"].str.lower(),
    })
    return blocks


def price_bucket_pairs(vehicles: pd.DataFrame, cutoff_ts: int, quantiles: List[float]) -> tuple[pd.DataFrame, dict]:
    """Quintile buckets of price_real with cut points from vehicles listed up to cutoff_ts."""
    reference = vehicles.loc[vehicles["listed_ts"] <= cutoff_ts, "price_real"].replace([np.inf, -np.inf], np.nan).dropna()
    cuts = sorted(set(float(x) for x in np.quantile(reference, quantiles)))
    bins = [-np.inf] + cuts + [np.inf]
    labels = [f"PriceBucket::Q{i + 1}" for i in range(len(bins) - 1)]
    keys = pd.cut(vehicles["price_real"], bins=bins, labels=labels, include_lowest=True, right=True)
    keys = keys.astype(str).replace("nan", "PriceBucket::__NA")
    spec = {"mode": "quantile_real", "quantiles": list(quantiles), "labels": labels}
    return pd.DataFrame({"i_idx": vehicles["i_idx"].values, "p_key": keys.values}), spec


def age_bucket_pairs(vehicles: pd.DataFrame, cuts: List[float]) -> tuple[pd.DataFrame, dict]:
    bins = [-np.inf] + list(cuts) + [np.inf]
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            labels.append(f"AgeBucket::<= {int(hi)}y")
        elif i == len(bins) - 2:
            labels.append(f"AgeBucket::> {int(lo)}y")
        else:
            labels.append(f"AgeBucket::{int(lo)}-{int(hi)}y")
    keys = pd.cut(vehicles["age_years"], bins=bins, labels=labels, include_lowest=True, right=True)
    keys = keys.astype(str).replace("nan", "AgeBucket::__NA")
    spec = {"mode": "fixed_years", "cutpoints_years": list(cuts), "labels": labels}
    return pd.DataFrame({"i_idx": vehicles["i_idx"].values, "p_key": keys.values}), spec


def coverage_keep(pairs: pd.DataFrame, cfg: GraphConfig, full_keep: bool) -> List[str]:
    """Property values that cover cfg.coverage_target of the pairs, each with at least coverage_min_count vehicles.

    pairs must contain only vehicles listed up to the cutoff, so the vocabulary uses no later inventory.
    """
    counts = pairs["p_key"].value_counts()
    if full_keep or counts.empty:
        return list(counts.index)
    cumulative = counts.cumsum() / counts.sum()
    keep_mask = cumulative <= cfg.coverage_target
    keep_mask.iloc[0] = True
    keep = [k for k in counts.index[keep_mask] if counts[k] >= cfg.coverage_min_count] or [counts.index[0]]
    return keep[: cfg.coverage_max_keep]


def build_property_graph(vehicles: pd.DataFrame, cutoff_ts: int, cfg: GraphConfig) -> PropertyGraph:
    """vehicles columns: i_idx, brand, model_family, fuel_type, transmission, vehicle_type, price_real, age_years, listed_ts."""
    blocks = _categorical_pairs(vehicles)
    blocks["PriceBucket"], price_spec = price_bucket_pairs(vehicles, cutoff_ts, cfg.price_quantiles)
    blocks["AgeBucket"], age_spec = age_bucket_pairs(vehicles, cfg.age_cuts_years)

    listed = set(vehicles.loc[vehicles["listed_ts"] <= cutoff_ts, "i_idx"])
    kept = []
    for prefix, pairs in blocks.items():
        reference = pairs[pairs["i_idx"].isin(listed)]
        keep = set(coverage_keep(reference, cfg, prefix in cfg.full_keep_properties))
        kept.append(pairs[pairs["p_key"].isin(keep)])
    pairs = pd.concat(kept, ignore_index=True).drop_duplicates()

    nodes = pd.DataFrame({"p_key": np.sort(pairs["p_key"].unique())})
    nodes["p_idx"] = np.arange(len(nodes), dtype=np.int64)
    edges = (
        pairs.merge(nodes, on="p_key", how="inner")[["i_idx", "p_idx"]]
        .drop_duplicates()
        .astype({"i_idx": np.int64, "p_idx": np.int64})
        .sort_values(["i_idx", "p_idx"])
        .reset_index(drop=True)
    )
    return PropertyGraph(nodes=nodes, edges=edges, bucket_spec={"PriceBucket": price_spec, "AgeBucket": age_spec})
