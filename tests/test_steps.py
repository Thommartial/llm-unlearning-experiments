"""Tests for gradient accumulation, the step counters, the memory probe and the LoRA diagnostic."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from unlearning import campaign, data, memprobe, unlearn
from unlearning.config import load_config


def _tiny(name: str = "tiny"):
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.model.name = name
    cfg.data.dataset = "synthetic_official"
    cfg.unlearn.train_batch_size = 16
    torch.manual_seed(0)
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0  # the two runs below must see the same forward pass
    return cfg, model, data.load_splits(cfg)


def _trained(cfg, model, ds, accum: int):
    cfg = copy.deepcopy(cfg)
    cfg.unlearn.grad_accum_steps = accum
    model = copy.deepcopy(model)
    torch.manual_seed(1)
    stats = unlearn.train_lm(model, ds, cfg, "cpu", epochs=1, lr=1e-3)
    return {k: v.detach() for k, v in model.state_dict().items()}, stats


@pytest.mark.parametrize("accum", [2, 3, 4])
def test_grad_accumulation_gives_the_whole_batch_gradient(accum):
    cfg, model, splits = _tiny()
    ds = splits.retain
    batch = data.collate_for(ds)([ds[i] for i in range(16)])
    grads = []
    for n in (1, accum):
        model.zero_grad()
        if n == 1:
            model(**batch).loss.backward()
        else:
            assert unlearn._accumulate(model, batch, n, cfg, "cpu", torch.amp.GradScaler("cuda", enabled=False))
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-7)
    # whole runs: the same steps; Adam turns rounding into differences of about 1e-5
    a, sa = _trained(cfg, model, ds, 1)
    b, sb = _trained(cfg, model, ds, accum)
    assert sa.batches == sb.batches == math.ceil(len(ds) / 16) and sa.applied == sb.applied == sa.batches
    assert max(float((a[k] - b[k]).abs().max()) for k in a if a[k].is_floating_point()) < 1e-4


def test_micro_batches_are_cut_to_their_longest_record():
    _, _, splits = _tiny()
    ds = splits.retain
    batch = data.collate_for(ds)([ds[i] for i in range(16)])
    micro = unlearn._micro_batches(batch, 2)
    assert [len(m["input_ids"]) for m in micro] == [8, 8]
    for m, rows in zip(micro, (range(8), range(8, 16)), strict=True):
        assert m["input_ids"].shape[1] == max(len(ds[i]["input_ids"]) for i in rows)
    assert sum(unlearn._scored(m["labels"]) for m in micro) == unlearn._scored(batch["labels"])


def test_grad_accum_steps_must_be_positive():
    cfg, model, splits = _tiny()
    cfg.unlearn.grad_accum_steps = 0
    with pytest.raises(ValueError, match="grad_accum_steps"):
        unlearn.train_lm(model, splits.retain, cfg, "cpu", epochs=1, lr=1e-3)


def test_step_stats_record():
    s = unlearn.StepStats(batches=4)
    s.record(2.0, 65536.0, 65536.0)
    s.record(float("inf"), 65536.0, 32768.0)  # the scaler skipped this step and halved its scale
    s.record(0.0, 32768.0, 32768.0)
    assert (s.skipped_scaler, s.nonfinite_grad, s.zero_grad, s.applied) == (1, 1, 1, 3)
    assert (s.scale_first, s.scale_min, s.scale_last, s.grad_norm_mean) == (65536.0, 32768.0, 32768.0, 1.0)
    assert s.summary()["applied"] == 3


def test_counters_in_training_and_unlearning(capsys):
    cfg, model, splits = _tiny()
    stats = unlearn.train_lm(model, splits.retain, cfg, "cpu", epochs=1, lr=1e-3)
    assert stats.skipped_loss == stats.skipped_scaler == 0 and stats.scale_last == 1.0  # no scaler on CPU
    assert "[steps] train_lm:" in capsys.readouterr().out
    cfg.unlearn.method = "gradient_ascent"
    s = unlearn.StepStats()
    unlearn.apply_unlearning(model, splits, cfg, "cpu", s)
    assert s.batches == cfg.unlearn.epochs * math.ceil(len(splits.forget) / cfg.unlearn.batch_size) == s.applied
    assert "[steps] apply_unlearning gradient_ascent:" in capsys.readouterr().out
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))  # every loss is now NaN
    s = unlearn.train_lm(model, splits.retain, cfg, "cpu", epochs=1, lr=1e-3)
    assert s.skipped_loss == s.batches and s.applied == 0


def test_campaign_configs_split_gpt2_medium_into_micro_batches():
    for name in ("central", "fed_iid"):
        path = f"configs/revision/{name}.yaml"
        units = campaign.units(campaign.read_campaign(path))
        assert len({u.slug for u in units}) == len(units) == 15
        for u in units:
            cfg = campaign.resolve(path, u)
            assert cfg.unlearn.train_batch_size == 16  # effective batch and rate unchanged
            assert cfg.unlearn.grad_accum_steps == (2 if u.model == "gpt2-medium" else 1)


def test_memory_probe_runs_on_cpu(capsys):
    out = memprobe.run_probe("configs/revision/smoke_cpu.yaml", "tiny", "cpu")
    assert list(out) == ["finetune", "npo", "fedavg", "client_ga", "eval"]
    assert all(c["peak_mem_reserved_mib"] is None for c in out.values())
    text = capsys.readouterr().out
    assert "[steps] apply_unlearning npo:" in text and "[steps] client_gradient_ascent:" in text
    assert list(memprobe.run_probe("configs/revision/smoke_cpu.yaml", "tiny_llama", "cpu")) == ["finetune", "npo", "eval"]


def test_lora_diag_runs_on_cpu(tmp_path):
    spec = importlib.util.spec_from_file_location("lora_diag", "scripts/lora_diag.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    assert diag.main(["--config", "configs/revision/smoke_cpu.yaml", "--model", "tiny_llama", "--device", "cpu",
                      "--steps", "3", "--arms", "amp", "fp32_adapters", "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "lora_diag.json").read_text())
    assert [s["arm"] for s in summary] == ["amp", "fp32_adapters"]
    for s in summary:
        assert s["steps"]["applied"] == 3 and s["adapters_with_grad_step1"] == s["trainable_tensors"] > 0
        assert s["lora_B_nonzero_grad_step1"] == s["trainable_tensors"] // 2
        assert s["adapter_delta"] > 0 and s["lora_B_norm"] > 0
    assert len((tmp_path / "lora_diag_steps.csv").read_text().splitlines()) == 1 + 2 * 3


def test_diag_notebook_is_generated():
    import ast

    spec = importlib.util.spec_from_file_location("make_revision_notebooks", "scripts/make_revision_notebooks.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    for name, d in gen.DIAGNOSTICS.items():
        nb = json.loads((Path("notebooks/revision") / name / f"{name}.ipynb").read_text())
        assert nb == gen.diag_notebook(name, d), f"{name}: rerun scripts/make_revision_notebooks.py"
        assert json.loads((Path("notebooks/revision") / name / "kernel-metadata.json").read_text()) == gen.metadata(name)
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for cell in code:
            ast.parse(cell)
        assert 'REPO_COMMIT = "SET_BY_KAGGLE_RUN"' in code[0] and Path(d["script"]).exists()
    # every notebook sets the allocator before torch starts CUDA; 01 and 02 probe GPT-2-medium first
    for name in [*gen.NOTEBOOKS, *gen.DIAGNOSTICS]:
        text = (Path("notebooks/revision") / name / f"{name}.ipynb").read_text()
        assert text.index("expandable_segments:True") < text.index("import torch")
        assert ("--memory-probe" in text) == (name in ("01_central", "02_fed_iid"))


def _cuda_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


@pytest.mark.skipif(not _cuda_supported(), reason="needs a GPU this PyTorch build supports")
def test_lora_fp16_amp_steps_are_applied_on_cuda():
    cfg, _, splits = _tiny("tiny_llama")
    cfg.model.lora = {}
    cfg.model.precision = "fp16_amp"
    torch.manual_seed(0)
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    s = unlearn.train_lm(model, splits.retain, cfg, "cuda", epochs=2, lr=1e-3)
    assert s.scale_first == 65536.0 and s.applied > s.batches // 2 and s.zero_grad == 0
