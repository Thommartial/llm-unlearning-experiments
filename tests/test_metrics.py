"""Torch-free unit tests for the MIA metric math."""

import numpy as np

from unlearning.evaluate import _auc, mia_metrics


def test_auc_perfect_separation():
    member = np.array([0.9, 0.8, 0.95])
    nonmember = np.array([0.1, 0.2, 0.05])
    assert _auc(member, nonmember) == 1.0


def test_auc_chance_is_half():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    b = rng.normal(size=2000)
    assert abs(_auc(a, b) - 0.5) < 0.05


def test_mia_metrics_includes_slices():
    import torch

    scores = {
        "forget": torch.tensor([0.9, 0.85, 0.2, 0.25]),
        "holdout": torch.tensor([0.1, 0.15, 0.2]),
        "forget_is_outlier": [True, True, False, False],
    }
    m = mia_metrics(scores)
    assert "mia_auc" in m and "mia_auc_outlier" in m and "mia_auc_typical" in m
