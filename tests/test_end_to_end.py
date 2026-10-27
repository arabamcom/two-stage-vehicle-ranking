"""Small end-to-end run of both stages on synthetic data."""
import numpy as np
import pandas as pd
import torch

from vehicle_ranking.config import (
    GnnTrainingConfig,
    GraphConfig,
    ModelConfig,
    RankingDatasetConfig,
    RankingTrainingConfig,
)
from vehicle_ranking.data.export import load_export
from vehicle_ranking.data.synthetic import SyntheticConfig, generate
from vehicle_ranking.evaluation.full_pool import KEYS, evaluate_systems
from vehicle_ranking.gnn.export import EMBEDDING_SNAPSHOTS, export_embeddings, gnn_cutoffs, graph_at
from vehicle_ranking.gnn.model import HeteroGNN
from vehicle_ranking.gnn.train import GnnTrainer
from vehicle_ranking.ranking.baselines import content_affinity_score, popularity_rate_scores
from vehicle_ranking.ranking.dataset import build_ranking_dataset
from vehicle_ranking.ranking.train import load_splits, train_lambdarank
from vehicle_ranking.utils import set_seed


def test_pipeline_runs_end_to_end(tmp_path):
    set_seed(42)
    graph_cfg = GraphConfig()
    generate(SyntheticConfig(n_users=150, n_days=90, vehicles_per_day=25, val_days=7, test_days=6,
                             static_cutoff_days=30),
             tmp_path / "export", graph_cfg)
    export = load_export(tmp_path / "export")

    gnn_cfg = GnnTrainingConfig(num_epochs=2, batch_size=256, num_neighbors=[5, 5], emb_cutoff_days=30,
                                val_days=7, test_days=7, device="cpu")
    cutoffs = gnn_cutoffs(export.cutoffs, gnn_cfg)
    data, scalers = graph_at(export, cutoffs.train, graph_cfg)
    model = HeteroGNN(data, ModelConfig(hidden_channels=16, out_channels=8, prop_embedding_dim=4))
    results = GnnTrainer(model, gnn_cfg, tmp_path).fit(data, cutoffs, export.daily_rosters(), export.listing_day())
    assert 0.0 <= results["test_auc"] <= 1.0
    bid_weight = model.convs[0].convs[("user", "BID", "item")].lin_rel.weight
    assert float(bid_weight.abs().mean()) > 1e-4, "bid messages must be trained"

    export_embeddings(model, data, cutoffs.train, tmp_path / "gnn" / EMBEDDING_SNAPSHOTS["train_rows"])
    data_t_train, _ = graph_at(export, export.cutoffs.train, graph_cfg, scalers)
    export_embeddings(model, data_t_train, export.cutoffs.train, tmp_path / "gnn" / EMBEDDING_SNAPSHOTS["eval_rows"])

    data_dir = tmp_path / "ranking"
    summary = build_ranking_dataset(export, tmp_path / "gnn", 30, RankingDatasetConfig(holdout_days=3), data_dir)
    assert summary["n_features"] == 50

    train_cfg = RankingTrainingConfig(num_boost_round=20, early_stopping=5)
    train_cfg.lgb_params["min_data_in_leaf"] = 20
    feat_cols = summary["feature_cols"]
    booster = train_lambdarank(load_splits(data_dir / "ranking_dataset.parquet", feat_cols, train_cfg), feat_cols, train_cfg)

    pool = pd.read_parquet(data_dir / "holdout_pool.parquet")
    scores = {
        "full": booster.predict(pool[feat_cols].to_numpy(np.float32)),
        "content_affinity": content_affinity_score(pool, KEYS),
        "popularity_rate": popularity_rate_scores(pool, export, 90, 0.0),
    }
    report = evaluate_systems(pool, scores, [("full", "content_affinity"), ("full", "random")], 10, 50, 42)
    assert report["n_tasks"] > 0
    assert 0.0 <= report["systems"]["full"]["aggregate"]["ndcg"] <= 1.0
    assert torch.isfinite(torch.tensor(scores["content_affinity"])).all()
