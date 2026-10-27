"""Full-roster evaluation of the learned rankers and the baselines on the locked holdout."""
import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vehicle_ranking.config import RankingConfig
from vehicle_ranking.data.export import load_export
from vehicle_ranking.evaluation.full_pool import KEYS, evaluate_systems
from vehicle_ranking.ranking.baselines import content_affinity_score, popularity_rate_scores
from vehicle_ranking.utils import read_json, write_json

COMPARISONS = [
    ("full", "content_affinity"), ("full", "popularity_rate"), ("full", "no_gnn"), ("full", "random"),
    ("content_affinity", "popularity_rate"),
]


def model_scores(run_dir: Path, pool: pd.DataFrame) -> np.ndarray:
    cols = read_json(run_dir / "config.json")["feature_cols"]
    return lgb.Booster(model_file=str(run_dir / "model.txt")).predict(pool[cols].to_numpy(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", default="data/synthetic")
    parser.add_argument("--dataset-dir", default="runs/ranking_data")
    parser.add_argument("--full-run", default="runs/ranker_full")
    parser.add_argument("--no-gnn-run", default="runs/ranker_no_gnn")
    parser.add_argument("--pool", default="holdout_pool", choices=["holdout_pool", "test_pool"])
    parser.add_argument("--config", default="configs/ranking.yaml")
    parser.add_argument("--out", default="runs/evaluation.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = RankingConfig.from_yaml(args.config).evaluation
    export = load_export(Path(args.export_dir))
    pool = pd.read_parquet(Path(args.dataset_dir) / f"{args.pool}.parquet")

    scores = {
        "full": model_scores(Path(args.full_run), pool),
        "content_affinity": content_affinity_score(pool, KEYS),
        "popularity_rate": popularity_rate_scores(pool, export, cfg.popularity_window_days, cfg.popularity_shrinkage),
    }
    comparisons = list(COMPARISONS)
    if Path(args.no_gnn_run).exists():
        scores["no_gnn"] = model_scores(Path(args.no_gnn_run), pool)
    else:
        comparisons = [c for c in comparisons if "no_gnn" not in c]

    report = evaluate_systems(pool, scores, comparisons, cfg.k, cfg.n_resamples, cfg.seed)
    write_json(Path(args.out), report)
    print(f"{report['n_tasks']} tasks, {report['mean_candidates']:.1f} candidates per task")
    for system, segments in report["systems"].items():
        print(f"  {system:18s} " + "  ".join(f"{seg} {v['ndcg']:.4f}" for seg, v in segments.items()))
    for name, segments in report["comparisons"].items():
        print(f"  {name:36s} " + "  ".join(
            f"{seg} {v['point']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}]" for seg, v in segments.items()))


if __name__ == "__main__":
    main()
