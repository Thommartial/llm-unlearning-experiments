"""Tests for the update-signal attacks: score change and gradient alignment."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unlearning import attacks, data, unlearn, update_attacks
from unlearning.config import load_config
from unlearning.evaluate import _auc
from unlearning.seed import set_seed


def _trained():
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.data.dataset = "synthetic_official"
    cfg.data.n_synthetic = 240
    cfg.data.forget_fraction = 0.2
    cfg.data.holdout_fraction = 0.2
    set_seed(0)
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg)
    unlearn.finetune_base(cfg, splits, "cpu", model=model)
    return cfg, model.eval(), splits


@pytest.fixture(scope="module")
def trained():
    return _trained()


def _loss_grads(model, batch, params):
    loss = -attacks._mean_of(*attacks._token_logprobs(model, batch, "cpu")).mean()
    return torch.autograd.grad(loss, list(params.values()))


def _subset(ds, n):
    return torch.utils.data.Subset(ds, range(n))


def test_score_change_on_hand_made_numbers():
    pre = {"forget": [1.0, 1.0, 1.0], "primary": [1.0, 1.0], "holdout10": [0.5, 0.5]}
    post = {"forget": [0.0, 0.5, 0.8], "primary": [1.0, 0.9], "holdout10": [0.5, 0.0]}
    change = update_attacks.score_change(pre, post)
    assert np.allclose(change["forget"], [1.0, 0.5, 0.2]) and np.allclose(change["primary"], [0.0, 0.1])
    auc = update_attacks.score_change_auc(pre, post)
    assert auc == {"primary": 1.0, "holdout10": _auc([1.0, 0.5, 0.2], [0.0, 0.5])}


def test_alignment_matches_flat_cosine(trained):
    _, model, splits = trained
    params = update_attacks._params(model, None)
    delta = update_attacks.random_direction({n: p.detach() for n, p in params.items()}, seed=3)
    ds = _subset(splits.forget, 6)
    got = update_attacks.grad_alignment_scores(model, delta, ds, "cpu")
    flat_d = torch.cat([v.flatten() for v in delta.values()])
    for i, batch in enumerate(DataLoader(ds, batch_size=1, collate_fn=data.collate)):
        g = torch.cat([x.flatten() for x in _loss_grads(model, batch, params)])
        assert abs(float(got[i]) - float(F.cosine_similarity(g, flat_d, dim=0))) < 1e-5


def test_param_filter_uses_only_those_parameters(trained):
    _, model, splits = trained
    keep = lambda n: ".h.0." in n  # noqa: E731
    params = update_attacks._params(model, keep)
    assert params and all(".h.0." in n for n in params)
    delta = update_attacks.random_direction({n: p.detach() for n, p in params.items()}, seed=4)
    ds = _subset(splits.forget, 4)
    got = update_attacks.grad_alignment_scores(model, delta, ds, "cpu", param_filter=keep)
    flat_d = torch.cat([v.flatten() for v in delta.values()])
    for i, batch in enumerate(DataLoader(ds, batch_size=1, collate_fn=data.collate)):
        g = torch.cat([x.flatten() for x in _loss_grads(model, batch, params)])
        assert abs(float(got[i]) - float(F.cosine_similarity(g, flat_d, dim=0))) < 1e-5
    with pytest.raises(ValueError):  # a direction over all parameters does not match the filter
        update_attacks.grad_alignment_scores(model, update_attacks.parameter_delta(model, model.state_dict()), ds,
                                             "cpu", param_filter=keep)


def test_sign_convention(trained):
    _, model, splits = trained
    params = update_attacks._params(model, None)
    ds = _subset(splits.forget, 3)
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=data.collate)))
    ascent = {n: g.detach() for n, g in zip(params, _loss_grads(model, batch, params), strict=True)}
    descent = {n: -v for n, v in ascent.items()}
    got = update_attacks.grad_alignment_scores(model, {"ascent": ascent, "descent": descent}, ds, "cpu")
    assert abs(float(got["ascent"][0]) - 1) < 1e-6 and abs(float(got["descent"][0]) + 1) < 1e-6
    assert torch.allclose(got["ascent"], -got["descent"])


def test_one_ascent_step_separates_forget_from_nonmembers():
    _, model, splits = _trained()
    pre = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.train()
    loader = DataLoader(splits.forget, batch_size=len(splits.forget), collate_fn=data.collate)
    params = update_attacks._params(model, None)
    grads = _loss_grads(model, next(iter(loader)), params)
    with torch.no_grad():
        for p, g in zip(params.values(), grads, strict=True):
            p.add_(1e-3 * g)  # one gradient-ascent step on the forget loss
    delta = update_attacks.parameter_delta(model, pre)
    dirs = {"delta": delta, "random": update_attacks.random_direction(delta, seed=0)}
    model.load_state_dict(pre)
    scores = {k: update_attacks.grad_alignment_scores(model, dirs, ds, "cpu")
              for k, ds in {"forget": splits.forget, **splits.nonmembers}.items()}
    for nm in splits.nonmembers:
        assert _auc(scores["forget"]["delta"].numpy(), scores[nm]["delta"].numpy()) > 0.9
        assert abs(_auc(scores["forget"]["random"].numpy(), scores[nm]["random"].numpy()) - 0.5) < 0.25


def test_random_direction_norm_and_rng():
    delta = {"a": torch.ones(3, 4), "b": torch.full((5,), 2.0)}
    state = torch.random.get_rng_state()
    r = update_attacks.random_direction(delta, seed=7)
    assert torch.equal(torch.random.get_rng_state(), state)  # global RNG untouched
    assert abs(update_attacks._norm(r) / update_attacks._norm(delta) - 1) < 1e-6  # fp32 rounding
    assert {n: v.shape for n, v in r.items()} == {n: v.shape for n, v in delta.items()}
    assert all(torch.equal(r[n], v) for n, v in update_attacks.random_direction(delta, seed=7).items())
    assert not torch.equal(r["a"], update_attacks.random_direction(delta, seed=8)["a"])
