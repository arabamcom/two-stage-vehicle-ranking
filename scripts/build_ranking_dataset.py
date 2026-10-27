"""Build the LambdaRank dataset and the full-roster evaluation tasks from a GNN run."""
import argparse
import logging
from pathlib import Path

from vehicle_ranking.config import RankingConfig
from vehicle_ranking.data.export import load_export
from vehicle_ranking.ranking.dataset import build_ranking_dataset
from vehicle_ranking.utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", default="data/synthetic")
    parser.add_argument("--gnn-run", default="runs/gnn")
    parser.add_argument("--out-dir", default="runs/ranking_data")
    parser.add_argument("--config", default="configs/ranking.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = RankingConfig.from_yaml(args.config)
    gnn_run, out_dir = Path(args.gnn_run), Path(args.out_dir)
    window_days = int(read_json(gnn_run / "results.json")["emb_cutoff_days"])
    summary = build_ranking_dataset(load_export(Path(args.export_dir)), gnn_run, window_days, cfg.dataset, out_dir)
    write_json(out_dir / "dataset_summary.json", summary)


if __name__ == "__main__":
    main()
