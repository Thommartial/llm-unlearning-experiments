"""Unit tests for the federated helpers (no GPU, no downloads)."""

from __future__ import annotations

import torch

from unlearning import data
from unlearning.config import load_config
from unlearning.federated import _weighted_average, partition_clients


def test_weighted_average_equal():
    a = {"w": torch.tensor([0.0, 0.0])}
    b = {"w": torch.tensor([2.0, 4.0])}
    avg = _weighted_average([a, b], [1, 1])
    assert torch.allclose(avg["w"], torch.tensor([1.0, 2.0]))


def test_weighted_average_weighted():
    a = {"w": torch.tensor([0.0, 0.0])}
    b = {"w": torch.tensor([4.0, 8.0])}
    avg = _weighted_average([a, b], [3, 1])  # 3:1 toward a
    assert torch.allclose(avg["w"], torch.tensor([1.0, 2.0]))


def test_partition_covers_retain_disjointly():
    cfg = load_config("configs/smoke_fed_synthetic.yaml")
    splits = data.load_splits(cfg)  # synthetic needs no tokenizer
    clients = partition_clients(splits, cfg)
    assert len(clients) == cfg.federated.num_clients
    # client 0 is exactly the forget set
    assert len(clients[0]) == len(splits.forget)
    # retain shards partition the retain set (cover all, no overlap)
    total = sum(len(c) for c in clients[1:])
    assert total == len(splits.retain)
