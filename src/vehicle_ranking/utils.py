"""Shared helpers: seeding, cutoff specification and export-directory file names."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DAY_SECONDS = 86400

NODES_USERS = "nodes_users.parquet"
NODES_ITEMS = "nodes_item.parquet"
NODES_PROPS = "nodes_item_props.parquet"
EDGES_ITEM_PROP = "edges_item_prop.parquet"
CUTOFF_SPEC = "cutoff_spec.json"
ROSTER = "roster.parquet"
BEHAVIOR_EDGES = ("bid", "purchase", "fav", "alert")


def edge_file(name: str) -> str:
    return f"edges_{name}.parquet"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Cutoffs:
    """UTC epoch seconds: GNN training ends at train, validation at val, test at test."""
    train: int
    val: int
    test: int

    def __post_init__(self):
        if not self.train < self.val < self.test:
            raise ValueError(f"Cutoff order violated: {self}")


def write_cutoffs(export_dir: Path, cutoffs: Cutoffs) -> None:
    spec = {
        "train_cutoff_epoch": cutoffs.train,
        "val_cutoff_epoch": cutoffs.val,
        "test_cutoff_epoch": cutoffs.test,
    }
    (export_dir / CUTOFF_SPEC).write_text(json.dumps(spec, indent=2), encoding="utf-8")


def read_cutoffs(export_dir: Path) -> Cutoffs:
    spec = json.loads((export_dir / CUTOFF_SPEC).read_text(encoding="utf-8"))
    return Cutoffs(
        train=int(spec["train_cutoff_epoch"]),
        val=int(spec["val_cutoff_epoch"]),
        test=int(spec["test_cutoff_epoch"]),
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
