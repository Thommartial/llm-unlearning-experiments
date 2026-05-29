"""Smoke tests for the reproducibility infrastructure (no torch/GPU required)."""

import random

from unlearning.config import load_config
from unlearning.seed import set_seed


def test_set_seed_is_deterministic():
    set_seed(123)
    first = [random.random() for _ in range(5)]
    set_seed(123)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_load_smoke_config():
    cfg = load_config("configs/smoke_synthetic.yaml")
    assert cfg.seed == 42
    assert cfg.model.name == "tiny"
    assert 0.0 < cfg.data.forget_fraction < 1.0
