"""Synthetic auction data in the export format consumed by the pipeline.

The generator simulates daily auction rosters of unique vehicles and users with latent price, age,
brand and damage-record preferences who bid on, buy, favorite and set alerts for vehicles. It writes
the same node tables, edge tables, property graph, cutoff specification and decision-time rosters
that the proprietary pipeline produces. The statistical properties of the proprietary data are not
reproduced; the data exists to run and inspect the implementation end to end.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vehicle_ranking.config import GraphConfig
from vehicle_ranking.graph.properties import build_property_graph
from vehicle_ranking.utils import (
    DAY_SECONDS,
    EDGES_ITEM_PROP,
    NODES_ITEMS,
    NODES_PROPS,
    NODES_USERS,
    ROSTER,
    Cutoffs,
    edge_file,
    write_cutoffs,
    write_json,
)

logger = logging.getLogger(__name__)

FUEL_TYPES = (["petrol", "diesel", "hybrid", "electric", "lpg"], [0.45, 0.35, 0.10, 0.05, 0.05])
TRANSMISSIONS = (["manual", "automatic", "semi_automatic"], [0.45, 0.45, 0.10])
VEHICLE_TYPES = (["sedan", "hatchback", "suv", "van", "pickup"], [0.35, 0.30, 0.25, 0.05, 0.05])
TYPE_PRICE_OFFSET = {"sedan": 0.0, "hatchback": -0.15, "suv": 0.25, "van": 0.05, "pickup": 0.15}


@dataclass
class SyntheticConfig:
    seed: int = 42
    n_users: int = 800
    n_days: int = 330
    start_epoch_day: int = 18631
    vehicles_per_day: float = 30.0
    n_brands: int = 12
    families_per_brand: int = 4
    user_daily_activity: float = 0.12
    bids_per_active_day: float = 2.5
    fav_share: float = 0.35
    alert_share: float = 0.08
    fav_only_per_active_day: float = 0.5
    purchase_rate: float = 0.7
    late_joiner_share: float = 0.25
    monthly_inflation: float = 0.01
    missing_rate: float = 0.02
    val_days: int = 17
    test_days: int = 15
    # Price buckets, property vocabulary and price-index medians use vehicles listed up to
    # T_train - static_cutoff_days. It must be at least the GNN window W, so that these specs end at
    # or before T_emb.
    static_cutoff_days: int = 180


def _choice(rng, spec, size):
    values, probs = spec
    return rng.choice(values, size=size, p=probs)


def simulate_vehicles(cfg: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    days = range(cfg.start_epoch_day, cfg.start_epoch_day + cfg.n_days)
    listing_day = np.concatenate([np.full(rng.poisson(cfg.vehicles_per_day), d) for d in days])
    n = len(listing_day)
    brand_pop = 1.0 / np.arange(1, cfg.n_brands + 1) ** 0.8
    brand = rng.choice(cfg.n_brands, size=n, p=brand_pop / brand_pop.sum())
    family = rng.integers(0, cfg.families_per_brand, size=n)
    vehicle_type = _choice(rng, VEHICLE_TYPES, n)
    age = np.clip(rng.gamma(2.5, 3.0, size=n), 0.0, 25.0)

    brand_level = rng.normal(10.0, 0.35, size=cfg.n_brands)
    family_offset = rng.normal(0.0, 0.2, size=(cfg.n_brands, cfg.families_per_brand))
    log_real = (brand_level[brand] + family_offset[brand, family]
                + np.vectorize(TYPE_PRICE_OFFSET.get)(vehicle_type) - 0.08 * age + rng.normal(0, 0.15, n))
    months = (listing_day - cfg.start_epoch_day) / 30.0
    price_nominal = np.exp(log_real) * (1.0 + cfg.monthly_inflation) ** months

    has_damage = rng.random(n) < 0.5
    damage = np.where(has_damage, price_nominal * rng.beta(1.2, 8.0, n) * 0.3, 0.0)
    damaged = rng.poisson(1.0 + 0.1 * age)
    replaced = rng.poisson(0.3, n)
    painted = rng.poisson(1.2, n)
    local_painted = rng.poisson(0.8, n)
    severity = rng.beta(2.0, 5.0, n) * (damaged + replaced > 0)

    df = pd.DataFrame({
        "i_idx": np.arange(n, dtype=np.int64),
        "listing_day": listing_day.astype(np.int64),
        "listed_ts": listing_day.astype(np.int64) * DAY_SECONDS + 6 * 3600,
        "brand": [f"brand_{b:02d}" for b in brand],
        "model_family": [f"family_{b:02d}_{f}" for b, f in zip(brand, family)],
        "fuel_type": _choice(rng, FUEL_TYPES, n),
        "transmission": _choice(rng, TRANSMISSIONS, n),
        "vehicle_type": vehicle_type,
        "age_years": age,
        "price_nominal": price_nominal,
        "log_real_true": log_real,
        "damage_record": damage,
        "appraisal_damaged_parts": damaged,
        "appraisal_replaced_parts": replaced,
        "appraisal_painted_parts": painted,
        "appraisal_local_painted_parts": local_painted,
        "appraisal_severity_mean": severity,
        "appraisal_severity_max": np.minimum(1.0, severity * rng.uniform(1.0, 1.6, n)),
        "appraisal_has_report": (rng.random(n) < 0.9).astype(np.int8),
    })
    df.loc[rng.random(n) < cfg.missing_rate, "age_years"] = np.nan
    df.loc[rng.random(n) < cfg.missing_rate, "price_nominal"] = np.nan
    return df


def simulate_users(cfg: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_users
    late = rng.random(n) < cfg.late_joiner_share
    return pd.DataFrame({
        "u_idx": np.arange(n, dtype=np.int64),
        "has_plan": (rng.random(n) < 0.4).astype(np.int8),
        "pref_log_price": rng.normal(9.6, 0.45, n),
        "price_tolerance": rng.uniform(0.15, 0.5, n),
        "pref_age": rng.gamma(2.0, 3.0, n),
        "age_tolerance": rng.uniform(1.5, 5.0, n),
        "damage_tolerance": rng.uniform(0.0, 1.0, n),
        "activity": np.clip(cfg.user_daily_activity * rng.lognormal(0.0, 0.7, n), 0.0, 0.9),
        "join_day": np.where(late, rng.integers(cfg.start_epoch_day, cfg.start_epoch_day + cfg.n_days, n),
                             cfg.start_epoch_day),
    })


def simulate_interactions(cfg: SyntheticConfig, users: pd.DataFrame, vehicles: pd.DataFrame,
                          rng: np.random.Generator) -> dict:
    """Per auction day, active users bid on the vehicles with the highest utility plus Gumbel noise."""
    brand_codes = vehicles["brand"].str[-2:].astype(int).to_numpy()
    brand_aff = rng.dirichlet(np.full(cfg.n_brands, 0.3), size=len(users)) * cfg.n_brands
    v_log = vehicles["log_real_true"].to_numpy()
    v_age = vehicles["age_years"].fillna(vehicles["age_years"].median()).to_numpy()
    v_damage = (vehicles["damage_record"] > 0).to_numpy(float)

    bids, favs, alerts, purchases = [], [], [], []
    for day, roster in vehicles.groupby("listing_day")["i_idx"]:
        items = roster.to_numpy()
        active = users[(users["join_day"] <= day) & (rng.random(len(users)) < users["activity"])]
        if active.empty:
            continue
        u = active["u_idx"].to_numpy()
        utility = (
            -0.5 * ((v_log[items][None, :] - active["pref_log_price"].to_numpy()[:, None])
                    / active["price_tolerance"].to_numpy()[:, None]) ** 2
            - 0.5 * ((v_age[items][None, :] - active["pref_age"].to_numpy()[:, None])
                     / active["age_tolerance"].to_numpy()[:, None]) ** 2
            + np.log(brand_aff[u][:, brand_codes[items]] + 0.05)
            - 1.5 * v_damage[items][None, :] * (1.0 - active["damage_tolerance"].to_numpy()[:, None])
            + rng.gumbel(size=(len(u), len(items)))
        )
        order = np.argsort(-utility, axis=1)
        day_start = int(day) * DAY_SECONDS
        winners = {}
        for row, user in enumerate(u):
            k = min(1 + rng.poisson(cfg.bids_per_active_day - 1.0), len(items))
            for col in order[row, :k]:
                item = int(items[col])
                ts = day_start + int(rng.uniform(8, 18) * 3600)
                count = 1 + rng.poisson(1.5)
                bids.append((int(user), item, ts, count))
                if rng.random() < cfg.fav_share:
                    favs.append((int(user), item, ts - int(rng.uniform(1, 6) * 3600)))
                if rng.random() < cfg.alert_share:
                    alerts.append((int(user), item, ts - int(rng.uniform(1, 7) * 3600)))
                score = count + rng.random()
                if score > winners.get(item, (None, -1.0))[1]:
                    winners[item] = (int(user), score)
            extra = order[row, k:k + rng.poisson(cfg.fav_only_per_active_day)]
            for col in extra:
                favs.append((int(user), int(items[col]), day_start + int(rng.uniform(7, 18) * 3600)))
        for item, (user, _) in winners.items():
            if rng.random() < cfg.purchase_rate:
                purchases.append((user, item, day_start + 19 * 3600))

    return {
        "bid": pd.DataFrame(bids, columns=["u_idx", "i_idx", "ts", "count"]),
        "purchase": pd.DataFrame(purchases, columns=["u_idx", "i_idx", "ts"]),
        "fav": pd.DataFrame(favs, columns=["u_idx", "i_idx", "ts"]),
        "alert": pd.DataFrame(alerts, columns=["u_idx", "i_idx", "ts"]),
    }


def real_prices(vehicles: pd.DataFrame, base_ts: int) -> pd.Series:
    """Deflate nominal prices to the month of base_ts with the median listing price of each month.

    A month's index is known once its listings are published, so it carries no interaction outcome.
    """
    month = pd.to_datetime(vehicles["listed_ts"], unit="s").dt.to_period("M")
    medians = vehicles["price_nominal"].groupby(month).median()
    base = medians.index[medians.index <= pd.Timestamp(base_ts, unit="s").to_period("M")].max()
    return vehicles["price_nominal"] * month.map(medians[base] / medians).astype(float)


def item_nodes(vehicles: pd.DataFrame, static_cutoff: int) -> tuple[pd.DataFrame, pd.Series]:
    """Vehicle features; reference medians use only vehicles listed up to static_cutoff."""
    v = vehicles.assign(price_real=real_prices(vehicles, static_cutoff))
    reference = v[v["listed_ts"] <= static_cutoff]
    family_median = reference.groupby("model_family")["price_real"].median()
    brand_median = reference.groupby("brand")["price_real"].median()
    deflator = v["price_real"] / v["price_nominal"]
    parts = v[["appraisal_damaged_parts", "appraisal_replaced_parts", "appraisal_painted_parts",
               "appraisal_local_painted_parts"]].sum(axis=1)
    nodes = pd.DataFrame({
        "i_idx": v["i_idx"],
        "item_price_real_log1p": np.log1p(v["price_real"]),
        "item_price_index_log1p": np.log1p(v["price_real"] / v["model_family"].map(family_median)),
        "item_price_index_brand_log1p": np.log1p(v["price_real"] / v["brand"].map(brand_median)),
        "item_age_years": v["age_years"],
        "item_age_log1p": np.log1p(v["age_years"]),
        "damage_record_log1p": np.log1p(v["damage_record"]),
        "damage_record_real_log1p": np.log1p(v["damage_record"] * deflator.fillna(1.0)),
        "damage_record_to_price": v["damage_record"] / v["price_nominal"],
        "damage_record_has_value": (v["damage_record"] > 0).astype(np.int8),
        "appraisal_damaged_parts": v["appraisal_damaged_parts"],
        "appraisal_replaced_parts": v["appraisal_replaced_parts"],
        "appraisal_painted_parts": v["appraisal_painted_parts"],
        "appraisal_local_painted_parts": v["appraisal_local_painted_parts"],
        "appraisal_severity_mean": v["appraisal_severity_mean"],
        "appraisal_severity_max": v["appraisal_severity_max"],
        "appraisal_has_report": v["appraisal_has_report"],
        "appraisal_parts_total_log1p": np.log1p(parts),
    })
    return nodes, v["price_real"]


def edge_tables(events: dict) -> dict:
    """One edge per (user, vehicle) and behavior with the latest ts; bid edges keep the number of bids."""
    out = {"bid": events["bid"].groupby(["u_idx", "i_idx"], as_index=False).agg(ts=("ts", "max"), count=("count", "sum"))}
    for name in ("purchase", "fav", "alert"):
        out[name] = events[name].groupby(["u_idx", "i_idx"], as_index=False).agg(ts=("ts", "max"))
    return {k: v.astype({"u_idx": np.int64, "i_idx": np.int64, "ts": np.int64}) for k, v in out.items()}


def cutoffs_for(cfg: SyntheticConfig) -> Cutoffs:
    test = (cfg.start_epoch_day + cfg.n_days) * DAY_SECONDS
    val = test - cfg.test_days * DAY_SECONDS
    return Cutoffs(train=val - cfg.val_days * DAY_SECONDS, val=val, test=test)


def generate(cfg: SyntheticConfig, export_dir: Path, graph_cfg: GraphConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    cutoffs = cutoffs_for(cfg)
    vehicles = simulate_vehicles(cfg, rng)
    users = simulate_users(cfg, rng)
    events = simulate_interactions(cfg, users, vehicles, rng)

    static_cutoff = cutoffs.train - cfg.static_cutoff_days * DAY_SECONDS
    items, price_real = item_nodes(vehicles, static_cutoff)
    props = build_property_graph(vehicles.assign(price_real=price_real), static_cutoff, graph_cfg)
    edges = edge_tables(events)

    export_dir.mkdir(parents=True, exist_ok=True)
    users[["u_idx", "has_plan"]].to_parquet(export_dir / NODES_USERS, index=False)
    items.to_parquet(export_dir / NODES_ITEMS, index=False)
    props.nodes.to_parquet(export_dir / NODES_PROPS, index=False)
    props.edges.to_parquet(export_dir / EDGES_ITEM_PROP, index=False)
    for name, df in edges.items():
        df.to_parquet(export_dir / edge_file(name), index=False)
    vehicles[["listing_day", "i_idx"]].rename(columns={"listing_day": "epoch_day"}).to_parquet(
        export_dir / ROSTER, index=False)
    write_cutoffs(export_dir, cutoffs)
    write_json(export_dir / "bucket_spec.json", props.bucket_spec)

    summary = {
        "n_users": int(len(users)),
        "n_vehicles": int(len(vehicles)),
        "n_property_nodes": int(len(props.nodes)),
        "edges": {k: int(len(v)) for k, v in edges.items()},
        "n_vehicle_property_edges": int(len(props.edges)),
    }
    logger.info("Synthetic export written to %s: %s", export_dir, summary)
    return summary
