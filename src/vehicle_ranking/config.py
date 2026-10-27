"""Configuration dataclasses and YAML loading.

The defaults reproduce the configuration reported in the manuscript (Tables 5 and 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GraphConfig:
    """Edge weights, reverse relations and property-node construction."""
    weight_bid: float = 3.0
    weight_purchase: float = 10.0
    weight_fav: float = 2.0
    weight_alert: float = 1.0
    # Repeated bids on the same pair: weight_bid * (1 + amplification * ln(1 + count)).
    bid_count_amplification: float = 0.2
    reverse_relations: List[str] = field(default_factory=lambda: ["BID", "PURCHASE", "FAVORITE", "ALERT", "HAS_PROPERTY"])
    # Multipliers applied to edge_weight during message passing; a reverse relation uses its base value.
    mp_rel_weights: Dict[str, float] = field(default_factory=lambda: {
        "BID": 1.0, "PURCHASE": 1.0, "FAVORITE": 1.0, "ALERT": 1.0, "HAS_PROPERTY": 0.8,
    })
    price_quantiles: List[float] = field(default_factory=lambda: [0.20, 0.40, 0.60, 0.80])
    age_cuts_years: List[float] = field(default_factory=lambda: [3.0, 6.0, 10.0])
    coverage_target: float = 0.99
    coverage_min_count: int = 20
    coverage_max_keep: int = 5000
    full_keep_properties: List[str] = field(default_factory=lambda: [
        "FuelType", "Transmission", "VehicleType", "PriceBucket", "AgeBucket",
    ])


@dataclass
class ModelConfig:
    """Heterogeneous GNN hyperparameters (Table 5)."""
    hidden_channels: int = 256
    out_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    prop_embedding_dim: int = 32
    residual: bool = True
    temperature: float = 0.15


@dataclass
class GnnTrainingConfig:
    """GNN training loop (Table 5)."""
    device: str = "auto"
    num_epochs: int = 200
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    num_neighbors: List[int] = field(default_factory=lambda: [25, 20])
    patience: int = 10
    min_delta: float = 0.0
    eval_neg_ratio: float = 3.0
    # Share of the vehicles with training bids that become targets in an epoch. Their bids are the
    # supervision edges and all their behavior edges leave the message-passing graph; the split is
    # redrawn every epoch, so targets and message edges never overlap.
    supervision_ratio: float = 0.3
    # Sliding window W. The GNN is trained on data up to T_emb = T_train - W days and selected and
    # tested on the following val_days and test_days.
    emb_cutoff_days: int = 180
    val_days: int = 17
    test_days: int = 15
    seed: int = 42


@dataclass
class RankingDatasetConfig:
    """LambdaRank training-data construction."""
    max_pos_per_user: int = 150
    neg_per_pos: int = 4
    min_neg_per_query: int = 0
    in_day_min_pool_size: int = 20
    # Share of negatives taken as embedding-nearest pool vehicles. The reported configuration
    # draws all negatives at random.
    hard_neg_ratio: float = 0.0
    recency_half_life_days: int = 180
    holdout_days: int = 7
    max_test_pool_positives: int = 5000
    max_holdout_pool_positives: int = 0
    pool_eval_min_size: int = 5
    test_pool_warm_fraction: float = 0.85
    daily_pool_feature_min_size: int = 20
    seed: int = 42
    label_map: Dict[str, int] = field(default_factory=lambda: {
        "alert": 1, "fav": 2, "bid": 5, "purchase": 6,
    })
    label_priority: Dict[int, float] = field(default_factory=lambda: {6: 6, 5: 5, 2: 2, 1: 2})
    prop_types_to_keep: Dict[str, str] = field(default_factory=lambda: {
        "Transmission": "i_transmission",
        "FuelType": "i_fuel_type",
        "Brand": "i_brand",
        "VehicleType": "i_vehicle_type",
        "ModelFamily": "i_model_family",
    })
    user_feat: List[str] = field(default_factory=lambda: [
        "u_has_plan",
        "u_total_bids_log1p",
        "u_n_items_bid_log1p",
        "u_n_items_fav_log1p",
        "u_n_items_alert_log1p",
        "u_bid_decay30_log1p",
        "u_hist_days_active_log1p",
        "u_bid_freq_per_active_day_log1p",
        "u_total_purchases_log1p",
        "u_n_brands_log1p",
        "u_brand_concentration_hhi",
        "u_has_price_pref",
        "u_has_age_pref",
    ])
    item_feat: List[str] = field(default_factory=lambda: [
        "i_item_price_real_log1p",
        "i_item_price_index_log1p",
        "i_item_price_index_brand_log1p",
        "i_item_age_years",
        "i_item_age_log1p",
        "i_damage_record_log1p",
        "i_damage_record_real_log1p",
        "i_damage_record_to_price",
        "i_damage_record_has_value",
    ])
    emb_feat: List[str] = field(default_factory=lambda: ["emb_cosine_sim", "emb_l2_dist"])


@dataclass
class RankingTrainingConfig:
    """LightGBM LambdaRank training (Table 6)."""
    num_boost_round: int = 5000
    early_stopping: int = 50
    log_period: int = 10
    query_group: str = "user_day"
    max_query_rows: int = 9000
    seed: int = 42
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10, 20, 50, 100, 200],
        "num_leaves": 31,
        "learning_rate": 0.07,
        "min_data_in_leaf": 700,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.5,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 10.0,
        "max_depth": 6,
        "verbose": -1,
        "seed": 42,
        "deterministic": True,
        "force_row_wise": True,
    })


@dataclass
class EvaluationConfig:
    """Full-pool evaluation, baselines and bootstrap."""
    k: int = 10
    n_resamples: int = 2000
    seed: int = 42
    popularity_window_days: int = 90
    popularity_shrinkage: float = 0.0


def _build(cls, values: Optional[Dict[str, Any]]):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_yaml(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class GnnConfig:
    graph: GraphConfig
    model: ModelConfig
    training: GnnTrainingConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> "GnnConfig":
        raw = load_yaml(path)
        return cls(
            graph=_build(GraphConfig, raw.get("graph")),
            model=_build(ModelConfig, raw.get("model")),
            training=_build(GnnTrainingConfig, raw.get("training")),
        )


@dataclass
class RankingConfig:
    dataset: RankingDatasetConfig
    training: RankingTrainingConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RankingConfig":
        raw = load_yaml(path)
        return cls(
            dataset=_build(RankingDatasetConfig, raw.get("dataset")),
            training=_build(RankingTrainingConfig, raw.get("training")),
            evaluation=_build(EvaluationConfig, raw.get("evaluation")),
        )
