import torch
from torch_geometric.data import HeteroData

from vehicle_ranking.gnn.train import split_supervision
from vehicle_ranking.graph.build import BID, PROPERTY_RELATION, add_reverse_edges
from vehicle_ranking.graph.temporal import create_time_sliced_graph, drop_item_behavior

FAV = ("user", "FAVORITE", "item")


def toy_graph() -> HeteroData:
    """Vehicle 0 has an early bid, vehicle 1 a late bid and an early favorite, vehicle 2 nothing."""
    data = HeteroData()
    data["user"].x = torch.zeros(2, 3)
    data["item"].x = torch.zeros(3, 3)
    data["property"].num_nodes = 2
    data[BID].edge_index = torch.tensor([[0, 1], [0, 1]])
    data[BID].ts = torch.tensor([100, 300])
    data[FAV].edge_index = torch.tensor([[0], [1]])
    data[FAV].ts = torch.tensor([150])
    data[PROPERTY_RELATION].edge_index = torch.tensor([[0, 1, 2], [0, 1, 1]])
    return add_reverse_edges(data, ["BID", "FAVORITE", "HAS_PROPERTY"])


def test_future_behavior_edges_are_removed():
    sliced = create_time_sliced_graph(toy_graph(), cutoff_ts=200)
    assert sliced[BID].edge_index.tolist() == [[0], [0]]
    assert sliced[("item", "BID_rev", "user")].edge_index.tolist() == [[0], [0]]


def test_property_edges_follow_source_visibility():
    sliced = create_time_sliced_graph(toy_graph(), cutoff_ts=200)
    assert sliced[PROPERTY_RELATION].edge_index.tolist() == [[0, 1], [0, 1]]
    rev = sliced[("property", "HAS_PROPERTY_rev", "item")].edge_index
    assert sorted(rev[1].tolist()) == [0, 1, 2]


def test_target_vehicles_lose_every_behavior_edge():
    graph = toy_graph()
    drop_item_behavior(graph, torch.tensor([1]))
    for edge_type in graph.edge_types:
        if edge_type[2] == "item" and edge_type[0] == "user":
            assert 1 not in graph[edge_type].edge_index[1].tolist()
        if edge_type[0] == "item" and edge_type[2] == "user":
            assert 1 not in graph[edge_type].edge_index[0].tolist()
    assert graph[BID].edge_index.tolist() == [[0], [0]]
    assert graph[BID].ts.tolist() == [100]
    assert graph[PROPERTY_RELATION].num_edges == 3


def test_supervision_targets_are_all_bids_of_the_chosen_vehicles():
    bids = torch.tensor([[0, 1, 2, 0], [5, 5, 6, 7]])
    targets, items = split_supervision(bids, 0.5, torch.Generator().manual_seed(0))
    assert len(items) == 2
    assert set(targets[1].tolist()) == set(items.tolist())
    assert targets.size(1) == int(torch.isin(bids[1], items).sum())
