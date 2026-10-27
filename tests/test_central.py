"""Tests for fine-tune-once centralized runs and the token-weighted loss."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from unlearning import attacks, central, data, evaluate, unlearn
from unlearning.config import load_config
from unlearning.seed import set_seed


def _cfg(dataset: str = "synthetic_official"):
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.data.dataset = dataset
    cfg.data.n_synthetic = 240
    cfg.data.forget_fraction = 0.2
    cfg.data.holdout_fraction = 0.2
    cfg.unlearn.epochs = 1
    return cfg


def _trained(dataset: str):
    cfg = _cfg(dataset)
    set_seed(0)
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg)
    unlearn.finetune_base(cfg, splits, "cpu", model=model)
    return cfg, model.eval(), splits


def test_official_loss_is_token_weighted_and_batch_invariant():
    cfg, model, splits = _trained("synthetic_official")
    ds = splits.retain
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(len(ds)):
            tok, m = attacks._token_logprobs(model, data.collate([ds[i]]), "cpu")
            total -= float((tok * m).sum())
            n += int(m.sum())
    losses = []
    for bs in (1, 7, 64):
        cfg.unlearn.eval_batch_size = bs
        losses.append(evaluate.mean_loss(model, ds, cfg, "cpu"))
    assert np.allclose(losses, total / n, rtol=0, atol=1e-6)


def test_legacy_loss_unchanged():
    cfg, model, splits = _trained("synthetic")
    ds = splits.retain
    with torch.no_grad():
        ref = np.mean([float(model(**b).loss)
                       for b in DataLoader(ds, batch_size=cfg.unlearn.batch_size, collate_fn=data.collate)])
    cfg.unlearn.eval_batch_size = 1  # must not be used on the legacy path
    assert evaluate.mean_loss(model, ds, cfg, "cpu") == ref


def _fresh(cfg, method: str):
    """The per-method pipeline that run_central replaces: build, fine-tune (or not), train."""
    set_seed(cfg.seed)
    model, tok = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg, tok)
    if method == "retrain":
        set_seed(cfg.seed)
        unlearn.retrain_base(cfg, splits, "cpu", model=model)
    else:
        unlearn.finetune_base(cfg, splits, "cpu", model=model)
        set_seed(cfg.seed)
        unlearn.apply_unlearning(model, splits, replace(cfg, unlearn=replace(cfg.unlearn, method=method)), "cpu")
    return {a: attacks.run_attack(model, splits, replace(cfg, attack=replace(cfg.attack, method=a)), "cpu")
            for a in central.ATTACKS}


@pytest.fixture(scope="module")
def full_run():
    cfg = _cfg()
    return cfg, central.run_central(cfg, "cpu")


@pytest.mark.parametrize("method", central.METHODS)
def test_restored_runs_match_fresh_pipeline(full_run, method):
    cfg, res = full_run
    fresh = _fresh(cfg, method)
    for a in central.ATTACKS:
        for split in ("forget", "holdout"):
            assert torch.equal(res["post"][method]["scores"][a][split], fresh[a][split])


def test_method_result_independent_of_other_methods(full_run):
    cfg, res = full_run
    alone = central.run_central(cfg, "cpu", methods=["npo"])
    assert list(alone["post"]) == ["npo"]
    for a in central.ATTACKS:
        assert torch.equal(alone["post"]["npo"]["scores"][a]["forget"], res["post"]["npo"]["scores"][a]["forget"])
    assert alone["pre"]["losses"] == res["pre"]["losses"]


def test_finetune_and_pre_scoring_run_once(monkeypatch):
    calls = {"train": 0, "attack": 0}
    train, attack = unlearn.train_lm, attacks.run_attack

    def count_train(*a, **k):
        calls["train"] += 1
        return train(*a, **k)

    def count_attack(*a, **k):
        calls["attack"] += 1
        return attack(*a, **k)

    monkeypatch.setattr(unlearn, "train_lm", count_train)
    monkeypatch.setattr(attacks, "run_attack", count_attack)
    res = central.run_central(_cfg(), "cpu")
    assert calls["train"] == 2  # one fine-tune + one retrain
    assert calls["attack"] == len(central.ATTACKS) * (1 + len(central.METHODS))
    assert set(res["post"]) == set(central.METHODS)


def test_rows_and_unknown_method(full_run):
    cfg, res = full_run
    rows = central.central_rows(cfg, res)
    assert len(rows) == len(central.METHODS) * len(central.ATTACKS)
    assert len({r["forget_loss_pre"] for r in rows}) == 1  # pre computed once, shared
    assert all(np.isfinite(r["auc_post"]) and np.isfinite(r["retain_loss_post"]) for r in rows)
    with pytest.raises(ValueError):
        central.run_central(cfg, "cpu", methods=["finetune"])
