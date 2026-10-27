"""Tests for the fp32 log-prob path, batch-size knobs and CUDA-only mixed precision."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from unlearning import attacks, data, unlearn
from unlearning.config import load_config


def _old_token_logprobs(model, batch):
    """The log_softmax + gather implementation used before WP2 (reference)."""
    ids, mask = batch["input_ids"], batch["attention_mask"]
    logp = F.log_softmax(model(input_ids=ids, attention_mask=mask).logits[:, :-1, :], dim=-1)
    tok = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok, (batch["labels"][:, 1:] != -100).to(mask.dtype)


def _tiny(dataset: str):
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.data.dataset = dataset
    torch.manual_seed(0)
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    return cfg, model.eval(), data.load_splits(cfg)


@pytest.mark.parametrize("dataset", ["synthetic", "synthetic_official"])
def test_new_logprobs_match_old(dataset):
    _, model, splits = _tiny(dataset)
    ds = splits.forget
    batch = data.collate_for(ds)([ds[i] for i in range(8)])
    with torch.no_grad():
        new_tok, new_mask = attacks._token_logprobs(model, batch, "cpu")
        old_tok, old_mask = _old_token_logprobs(model, batch)
    assert torch.equal(new_mask, old_mask)
    assert new_tok.dtype == torch.float32
    assert torch.allclose(new_tok * new_mask, old_tok * old_mask, atol=1e-5)
    with torch.no_grad():
        new_avg = attacks.sequence_avg_logprob(model, batch, "cpu")
    old_avg = (old_tok * old_mask).sum(-1) / old_mask.sum(-1).clamp(min=1)
    assert torch.allclose(new_avg, old_avg, atol=1e-5)


def _cached_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        return AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True).eval(), tok
    except OSError:
        return None, None


def test_new_logprobs_match_old_on_pretrained_gpt2():
    """Real GPT-2 logits (about -320..120) expose precision loss that the tiny
    model does not, e.g. from feeding cross_entropy a transposed [B, V, T] input."""
    model, tok = _cached_gpt2()
    if model is None:
        pytest.skip("gpt2 not in the local Hugging Face cache")
    tok.pad_token = tok.eos_token
    rows = [data.encode_qa(tok, "Who wrote this novel about the sea?", "It was written by an "
                           "author from Lisbon, Portugal, known for maritime fiction.", 128),
            data.encode_qa(tok, "Where was the author born?", "In Santiago, Chile.", 128)]
    ds = data.QADataset([r[0] for r in rows], [r[1] for r in rows], [0, 1], [0, 0],
                        ["", ""], ["", ""], tok.pad_token_id)
    batch = data.collate_for(ds)([ds[0], ds[1]])
    with torch.no_grad():
        new_tok, mask = attacks._token_logprobs(model, batch, "cpu")
        old_tok, _ = _old_token_logprobs(model, batch)
    assert torch.allclose(new_tok * mask, old_tok * mask, atol=1e-5, rtol=0)


def test_amp_is_cuda_only():
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.model.precision = "fp16_amp"
    assert not unlearn._use_amp(cfg, "cpu")
    assert unlearn._use_amp(cfg, "cuda") and unlearn._use_amp(cfg, "cuda:1")
    cfg.model.precision = "fp32"
    assert not unlearn._use_amp(cfg, "cuda")


def test_batch_size_defaults():
    cfg = load_config("configs/smoke_synthetic.yaml")
    assert cfg.unlearn.train_batch_size is None  # falls back to batch_size
    assert cfg.unlearn.eval_batch_size == 64


def _train_and_unlearn(precision: str, device: str, method: str = "npo"):
    cfg, model, splits = _tiny("synthetic_official")
    cfg.model.precision = precision
    cfg.unlearn.method = method
    cfg.unlearn.train_batch_size = 16
    model = copy.deepcopy(model)
    torch.manual_seed(1)
    unlearn.train_lm(model, splits.retain, cfg, device, epochs=1, lr=1e-3)
    unlearn.apply_unlearning(model, splits, cfg, device)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def test_fp16_amp_leaves_cpu_path_unchanged():
    a = _train_and_unlearn("fp32", "cpu")
    b = _train_and_unlearn("fp16_amp", "cpu")
    assert all(torch.equal(a[k], b[k]) for k in a)


def _cuda_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


@pytest.mark.skipif(not _cuda_supported(), reason="needs a GPU this PyTorch build supports")
@pytest.mark.parametrize("method", ["gradient_difference", "npo"])
def test_fp16_amp_on_cuda_is_finite_and_close(method):
    a = _train_and_unlearn("fp32", "cuda", method)
    b = _train_and_unlearn("fp16_amp", "cuda", method)
    assert all(torch.isfinite(v).all() for v in b.values() if v.is_floating_point())
    diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a if a[k].is_floating_point())
    assert diff < 0.05
