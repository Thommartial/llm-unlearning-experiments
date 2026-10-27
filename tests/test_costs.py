"""Tests for the cost instrumentation: phases, environment, communication."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unlearning import central, costs, unlearn
from unlearning.config import load_config


def test_phase_on_cpu():
    with costs.phase("cpu") as c:
        sum(range(10_000))
    assert c["wall_s"] > 0 and c["peak_mem_mib"] is None and c["peak_mem_reserved_mib"] is None
    with pytest.raises(RuntimeError), costs.phase("cpu"), costs.phase("cpu"):
        pass
    with pytest.raises(ValueError), costs.phase("cpu"):
        raise ValueError
    with costs.phase("cpu") as c:  # the failed phases left nothing active
        pass
    assert "wall_s" in c


def test_combine_and_columns():
    a = {"wall_s": 1.0, "peak_mem_mib": 10.0, "peak_mem_reserved_mib": 40.0}
    b = {"wall_s": 2.5, "peak_mem_mib": 30.0, "peak_mem_reserved_mib": 32.0}
    assert costs.combine(a, b) == {"wall_s": 3.5, "peak_mem_mib": 30.0, "peak_mem_reserved_mib": 40.0}
    none = {"wall_s": 1.0, "peak_mem_mib": None, "peak_mem_reserved_mib": None}
    assert costs.combine(a, none)["peak_mem_mib"] == 10.0
    assert costs.columns({"finetune": a, "alignment": None}) == {
        "time_s_finetune": 1.0, "peak_mem_mib_finetune": 10.0, "peak_mem_reserved_mib_finetune": 40.0,
        "time_s_alignment": None, "peak_mem_mib_alignment": None, "peak_mem_reserved_mib_alignment": None}


def test_communication():
    assert costs.fedavg_transfers(5, 3) == (15, 15)
    assert costs.communication(1000, 15, 15) == {"comm_uploads": 15, "comm_downloads": 15,
                                                 "comm_bytes": 30 * 1000 * 4}


def test_trainable_params_counts_tied_weights_once():
    cfg = load_config("configs/smoke_synthetic.yaml")
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    assert model.lm_head.weight is model.transformer.wte.weight
    n = sum(p.numel() for _, p in model.named_parameters())
    assert costs.trainable_params(model) == n
    model.transformer.wte.weight.requires_grad_(False)  # e.g. frozen base weights under LoRA
    assert costs.trainable_params(model) == n - model.transformer.wte.weight.numel()


def test_environment_on_cpu():
    env = costs.environment("cpu")
    assert env == {"gpu_name": "cpu", "torch_version": torch.__version__, "cuda_version": torch.version.cuda}


def _cuda_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


@pytest.mark.skipif(not _cuda_supported(), reason="needs a GPU this PyTorch build supports")
def test_run_central_on_cuda_records_gpu_costs():
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.data.dataset = "synthetic_official"
    cfg.data.max_new_tokens = 8
    cfg.unlearn.epochs = 1
    with costs.phase("cuda") as c:
        x = torch.ones(1024, 1024, device="cuda")
    assert c["peak_mem_mib"] >= x.numel() * 4 / 2**20 and c["peak_mem_reserved_mib"] >= c["peak_mem_mib"]
    rows = central.central_rows(cfg, central.run_central(cfg, "cuda", methods=["retrain", "gradient_ascent"]))
    for r in rows:
        assert r["gpu_name"] != "cpu" and r["cuda_version"]
        assert all(r[f"peak_mem_mib_{p}"] > 0 and r[f"peak_mem_reserved_mib_{p}"] >= r[f"peak_mem_mib_{p}"]
                   and np.isfinite(r[f"time_s_{p}"])
                   for p in ("base_scoring", "finetune", "pre_eval", "method", "post_eval", "alignment"))
