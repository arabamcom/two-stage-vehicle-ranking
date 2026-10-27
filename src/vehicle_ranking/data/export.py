"""The export directory read by every pipeline stage, loaded once."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from vehicle_ranking.utils import (
    BEHAVIOR_EDGES,
    DAY_SECONDS,
    EDGES_ITEM_PROP,
    NODES_ITEMS,
    NODES_PROPS,
    NODES_USERS,
    ROSTER,
    Cutoffs,
    edge_file,
    read_cutoffs,
)


@dataclass
class ExportData:
    users: pd.DataFrame          # u_idx and static user attributes
    items: pd.DataFrame          # i_idx and vehicle features
    props: pd.DataFrame          # p_key, p_idx
    item_props: pd.DataFrame     # i_idx, p_idx
    edges: Dict[str, pd.DataFrame]  # behavior -> u_idx, i_idx, ts (bid also has count)
    roster: pd.DataFrame         # epoch_day, i_idx: vehicles listed in each day's auction
    cutoffs: Cutoffs

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def edges_until(self, name: str, cutoff: int) -> pd.DataFrame:
        df = self.edges[name]
        return df[df["ts"] <= cutoff]

    def edges_between(self, name: str, start: int, end: int) -> pd.DataFrame:
        df = self.edges[name]
        return df[(df["ts"] > start) & (df["ts"] <= end)]

    def listing_day(self) -> pd.Series:
        """First roster day of every vehicle, indexed by i_idx."""
        return self.roster.groupby("i_idx")["epoch_day"].min()

    def daily_rosters(self) -> Dict[int, np.ndarray]:
        return {int(d): np.sort(g.to_numpy()) for d, g in self.roster.groupby("epoch_day")["i_idx"]}

    def property_values(self, prefix: str) -> pd.Series:
        """Value of one property type per vehicle (e.g. 'Brand', 'ModelFamily'), indexed by i_idx."""
        props = self.props[self.props["p_key"].str.startswith(prefix + "::")]
        pairs = self.item_props.merge(props, on="p_idx").drop_duplicates("i_idx")
        return pairs.set_index("i_idx")["p_key"]


def load_export(export_dir: Path) -> ExportData:
    edges = {}
    for name in BEHAVIOR_EDGES:
        df = pd.read_parquet(export_dir / edge_file(name))
        edges[name] = df.astype({"u_idx": np.int64, "i_idx": np.int64, "ts": np.int64})
    roster = pd.read_parquet(export_dir / ROSTER).astype({"epoch_day": np.int64, "i_idx": np.int64})
    return ExportData(
        users=pd.read_parquet(export_dir / NODES_USERS).sort_values("u_idx").reset_index(drop=True),
        items=pd.read_parquet(export_dir / NODES_ITEMS).sort_values("i_idx").reset_index(drop=True),
        props=pd.read_parquet(export_dir / NODES_PROPS).sort_values("p_idx").reset_index(drop=True),
        item_props=pd.read_parquet(export_dir / EDGES_ITEM_PROP),
        edges=edges,
        roster=roster,
        cutoffs=read_cutoffs(export_dir),
    )


def epoch_day(ts) -> np.ndarray:
    return (np.asarray(ts) // DAY_SECONDS).astype(np.int64)
