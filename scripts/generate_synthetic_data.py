"""Write a synthetic export directory: node and edge tables, property graph, cutoffs and daily rosters."""
import argparse
import logging
from pathlib import Path

from vehicle_ranking.config import GnnConfig, load_yaml
from vehicle_ranking.data.synthetic import SyntheticConfig, generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--gnn-config", default="configs/gnn.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = SyntheticConfig(**load_yaml(args.config))
    generate(cfg, Path(args.out), GnnConfig.from_yaml(args.gnn_config).graph)


if __name__ == "__main__":
    main()
