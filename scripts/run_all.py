"""Run the complete pipeline on synthetic data: generate, train the GNN, build the ranking data,
train the full and no-GNN rankers, and evaluate."""
import argparse
import subprocess
import sys


def run(*args: str) -> None:
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=None, help="Cap GNN epochs for a quick run")
    args = parser.parse_args()

    run("scripts/generate_synthetic_data.py")
    gnn = ["scripts/train_gnn.py"] + (["--max-epochs", str(args.max_epochs)] if args.max_epochs else [])
    run(*gnn)
    run("scripts/build_ranking_dataset.py")
    run("scripts/train_ranker.py", "--run-dir", "runs/ranker_full")
    run("scripts/train_ranker.py", "--run-dir", "runs/ranker_no_gnn", "--no-gnn")
    run("scripts/evaluate.py")


if __name__ == "__main__":
    main()
