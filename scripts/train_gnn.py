"""Stage 1: train the heterogeneous GNN on data up to T_emb and export the ranking embeddings."""
import argparse
import logging
from pathlib import Path

from vehicle_ranking.config import GnnConfig
from vehicle_ranking.data.export import load_export
from vehicle_ranking.gnn.export import EMBEDDING_SNAPSHOTS, export_embeddings, gnn_cutoffs, graph_at
from vehicle_ranking.gnn.model import HeteroGNN
from vehicle_ranking.gnn.train import GnnTrainer
from vehicle_ranking.utils import set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", default="data/synthetic")
    parser.add_argument("--run-dir", default="runs/gnn")
    parser.add_argument("--config", default="configs/gnn.yaml")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override num_epochs, e.g. for a quick run")
    parser.add_argument("--emb-window-days", type=int, default=None, help="Override the sliding window W")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = GnnConfig.from_yaml(args.config)
    if args.max_epochs is not None:
        cfg.training.num_epochs = args.max_epochs
    if args.emb_window_days is not None:
        cfg.training.emb_cutoff_days = args.emb_window_days
    set_seed(cfg.training.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    export = load_export(Path(args.export_dir))
    cutoffs = gnn_cutoffs(export.cutoffs, cfg.training)

    data, scalers = graph_at(export, cutoffs.train, cfg.graph)
    model = HeteroGNN(data, cfg.model)
    results = GnnTrainer(model, cfg.training, run_dir).fit(data, cutoffs, export.daily_rosters(), export.listing_day())

    export_embeddings(model, data, cutoffs.train, run_dir / EMBEDDING_SNAPSHOTS["train_rows"])
    data_t_train, _ = graph_at(export, export.cutoffs.train, cfg.graph, scalers)
    export_embeddings(model, data_t_train, export.cutoffs.train, run_dir / EMBEDDING_SNAPSHOTS["eval_rows"])

    results.update({"emb_cutoff_days": cfg.training.emb_cutoff_days, "gnn_cutoffs": cutoffs.__dict__})
    write_json(run_dir / "results.json", results)


if __name__ == "__main__":
    main()
