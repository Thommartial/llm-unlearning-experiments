"""Tests for the LoRA path (tiny Llama, CPU, no download)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from unlearning import attacks, central, costs, data, unlearn, update_attacks
from unlearning.config import load_config
from unlearning.seed import set_seed


def _cfg(lora: dict | None = None):
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.model.name = "tiny_llama"
    cfg.model.lora = {} if lora is None else lora
    cfg.data.dataset = "synthetic_official"
    cfg.data.max_new_tokens = 8
    cfg.unlearn.epochs = 1
    return cfg


def _build(cfg):
    set_seed(cfg.seed)
    return unlearn.build_model_and_tokenizer(cfg)[0]


def test_lora_model_shape_and_dtypes():
    cfg = _cfg()
    model = _build(cfg)
    assert unlearn.is_peft(model) and model.is_gradient_checkpointing
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert all(p.dtype == torch.float32 for p in trainable.values())  # adapters fp32
    assert all(p.dtype == torch.float16 for n, p in model.named_parameters() if "lora_" not in n)  # base fp16
    s = unlearn.lora_settings(cfg)
    assert s == unlearn.LORA_DEFAULTS
    targets = {n.split(".")[-4] for n in trainable}  # ...self_attn.q_proj.lora_A.default.weight
    assert targets == set(s["target_modules"])
    a = next(p for n, p in trainable.items() if "lora_A" in n)
    assert a.shape[0] == 16
    assert costs.trainable_params(model) == sum(p.numel() for p in trainable.values())
    small = _build(_cfg({"r": 4, "base_dtype": "float32", "gradient_checkpointing": False}))
    assert next(p for n, p in small.named_parameters() if "lora_A" in n).shape[0] == 4
    assert not small.is_gradient_checkpointing
    assert all(p.dtype == torch.float32 for p in small.parameters())
    assert unlearn.lora_filter(cfg)("x.lora_A.default.weight") and not unlearn.lora_filter(cfg)("x.q_proj.weight")
    full = _cfg()
    full.model.lora = None
    assert unlearn.lora_filter(full) is None and not unlearn.is_peft(_build(full))


def test_finetune_changes_adapters_only():
    cfg = _cfg()
    model = _build(cfg)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    unlearn.finetune_base(cfg, data.load_splits(cfg), "cpu", model=model)
    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        assert torch.equal(p, before[n]), n
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if "lora_B" in n)


def test_npo_reference_is_a_frozen_second_adapter(monkeypatch):
    cfg = _cfg()
    model = _build(cfg)
    splits = data.load_splits(cfg)
    unlearn.finetune_base(cfg, splits, "cpu", model=model)
    keys = set(model.state_dict())
    batch = next(iter(DataLoader(splits.forget, batch_size=4, collate_fn=data.collate_for(splits.forget))))
    model.eval()
    with torch.no_grad():
        lp_pre = attacks.sequence_avg_logprob(model, batch, "cpu")
    real = copy.deepcopy

    def no_model_copy(obj, *a, **k):
        if isinstance(obj, torch.nn.Module):
            pytest.fail("no full model copy under LoRA")
        return real(obj, *a, **k)
    monkeypatch.setattr(copy, "deepcopy", no_model_copy)
    logprob, release = unlearn._npo_reference(model, "cpu")
    assert unlearn.NPO_REFERENCE in model.peft_config
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and not any(unlearn.NPO_REFERENCE in n for n in trainable)
    assert torch.equal(logprob(batch), lp_pre)  # the reference is the trained adapter
    model.train()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        (-attacks.sequence_avg_logprob(model, batch, "cpu").mean()).backward()
        opt.step()
        assert torch.equal(logprob(batch), lp_pre)  # frozen
        assert [n for n, p in model.named_parameters() if p.requires_grad] == trainable
    model.eval()
    with torch.no_grad():
        assert not torch.equal(attacks.sequence_avg_logprob(model, batch, "cpu"), lp_pre)
    release()
    assert unlearn.NPO_REFERENCE not in model.peft_config and set(model.state_dict()) == keys


def test_npo_under_lora_runs_and_cleans_up():
    cfg = _cfg()
    cfg.unlearn.method = "npo"
    model = _build(cfg)
    splits = data.load_splits(cfg)
    unlearn.finetune_base(cfg, splits, "cpu", model=model)
    keys = set(model.state_dict())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    unlearn.apply_unlearning(model, splits, cfg, "cpu")
    assert set(model.state_dict()) == keys
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters() if "lora_" not in n)
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if "lora_" in n)


def test_fp32_weights_restores_dtypes_and_values():
    model = _build(_cfg())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with update_attacks.fp32_weights(model):
        assert all(p.dtype == torch.float32 for p in model.parameters())
    for k, v in model.state_dict().items():
        assert v.dtype == state[k].dtype and torch.equal(v, state[k])


def test_alignment_under_lora_is_fp32_over_adapters():
    cfg = _cfg()
    model = _build(cfg)
    splits = data.load_splits(cfg)
    unlearn.finetune_base(cfg, splits, "cpu", model=model)
    keep = unlearn.lora_filter(cfg)
    params = update_attacks._params(model, keep)
    delta = update_attacks.random_direction({n: p.detach().float() for n, p in params.items()}, seed=0)
    ds = torch.utils.data.Subset(splits.forget, range(6))
    got = update_attacks.grad_alignment_scores(model, delta, ds, "cpu", keep)
    assert all(p.dtype == torch.float16 for n, p in model.named_parameters() if "lora_" not in n)  # restored
    up = copy.deepcopy(model).float()
    ref = update_attacks.grad_alignment_scores(up, delta, ds, "cpu", keep)
    assert torch.equal(got, ref) and torch.isfinite(got).all()


@pytest.fixture(scope="module")
def lora_run():
    cfg = _cfg()
    return cfg, central.run_central(cfg, "cpu")


def test_central_lora_rows(lora_run):
    cfg, res = lora_run
    rows = central.central_rows(cfg, res)
    assert len(rows) == len(central.METHODS) * len(central.ATTACKS) * 2
    for r in rows:
        assert all(np.isfinite(r[k]) for k in ("auc_post", "gap_post", "auc_score_change", "auc_grad_align",
                                               "auc_grad_align_random", "model_utility_post"))
        assert r["delta_norm"] > 0


def test_lora_retrain_is_fresh_adapters_on_retain(lora_run):
    cfg, res = lora_run
    set_seed(cfg.seed)
    model, tok = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg, tok)
    with torch.random.fork_rng(devices=[]):
        central._scores(model, splits, cfg, "cpu")  # as in run_central
    set_seed(cfg.seed)
    unlearn.retrain_base(cfg, splits, "cpu", model=model)
    fresh = attacks.score_all(model, splits.forget, cfg, "cpu")["loss_threshold"]
    assert torch.equal(fresh, res["post"]["retrain"]["scores"]["loss_threshold"]["forget"])


def test_lora_delta_covers_adapters_only(monkeypatch):
    seen = []
    original = update_attacks.parameter_delta

    def wrapper(model, pre_state, *a, **k):
        delta = original(model, pre_state, *a, **k)
        seen.append(set(delta))
        return delta
    monkeypatch.setattr(update_attacks, "parameter_delta", wrapper)
    cfg = _cfg()
    central.run_central(cfg, "cpu", methods=["npo"])
    model = _build(cfg)
    adapters = {n for n, _ in model.named_parameters() if "lora_" in n}
    assert seen == [adapters]


def _cuda_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


@pytest.mark.skipif(not _cuda_supported(), reason="needs a GPU this PyTorch build supports")
def test_lora_fp16_amp_on_cuda_is_finite():
    cfg = _cfg()
    cfg.model.precision = "fp16_amp"
    rows = central.central_rows(cfg, central.run_central(cfg, "cuda"))
    assert all(np.isfinite(r["auc_post"]) and np.isfinite(r["auc_grad_align"]) for r in rows)
    assert all(r["peak_mem_mib_alignment"] > 0 for r in rows)


def test_snapshot_under_lora_holds_adapters_only():
    cfg = _cfg()
    model = _build(cfg)
    snap = central._snapshot(model)
    assert snap and set(snap) == {k for k in model.state_dict() if "lora_" in k}
    unlearn.finetune_base(cfg, data.load_splits(cfg), "cpu", model=model)
    central._restore(model, snap)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in snap.items())
    with pytest.raises(RuntimeError):
        central._restore(model, {k: v for k, v in list(snap.items())[1:]})
