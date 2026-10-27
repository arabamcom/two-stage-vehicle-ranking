"""Stage 2: train the LambdaRank model and report the sampled-list diagnostic on the test split."""
import argparse
import logging
from pathlib import Path

from vehicle_ranking.config import RankingConfig
from vehicle_ranking.evaluation.metrics import sampled_list_metrics
from vehicle_ranking.ranking.train import load_splits, train_lambdarank
from vehicle_ranking.utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="runs/ranking_data")
    parser.add_argument("--run-dir", default="runs/ranker_full")
    parser.add_argument("--config", default="configs/ranking.yaml")
    parser.add_argument("--no-gnn", action="store_true", help="Drop the graph-embedding features (ablation)")
    parser.add_argument("--query-group", choices=["user", "user_day"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = RankingConfig.from_yaml(args.config)
    if args.query_group:
        cfg.training.query_group = args.query_group
    if args.seed is not None:
        cfg.training.seed = args.seed

    dataset_dir, run_dir = Path(args.dataset_dir), Path(args.run_dir)
    feat_cols = read_json(dataset_dir / "dataset_summary.json")["feature_cols"]
    if args.no_gnn:
        feat_cols = [c for c in feat_cols if c not in cfg.dataset.emb_feat]

    splits = load_splits(dataset_dir / "ranking_dataset.parquet", feat_cols, cfg.training)
    model = train_lambdarank(splits, feat_cols, cfg.training)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(run_dir / "model.txt"), num_iteration=model.best_iteration)

    test = splits["test"]
    diagnostic = sampled_list_metrics(test["y"], model.predict(test["X"], num_iteration=model.best_iteration),
                                      test["groups"], k=10)
    write_json(run_dir / "config.json", {
        "feature_cols": feat_cols,
        "query_group": cfg.training.query_group,
        "seed": cfg.training.seed,
        "no_gnn": args.no_gnn,
        "best_iteration": int(model.best_iteration),
    })
    write_json(run_dir / "metrics.json", {"sampled_test_diagnostic": diagnostic})


if __name__ == "__main__":
    main()
