"""Unit tests for the federated helpers (no GPU, no downloads)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unlearning import data, federated, unlearn
from unlearning.config import load_config
from unlearning.federated import _weighted_average, partition_clients
from unlearning.seed import set_seed


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


# Revision path (run_fed)

def _fed_cfg(sharding: str = "iid"):
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.data.dataset = "synthetic_official"
    cfg.data.n_synthetic = 240
    cfg.data.forget_fraction = 0.2
    cfg.data.holdout_fraction = 0.2
    cfg.data.max_new_tokens = 8
    cfg.federated.num_clients = 4
    cfg.federated.rounds = 2
    cfg.federated.client_unlearn_epochs = 1
    cfg.federated.sharding = sharding
    return cfg


@pytest.mark.parametrize("sharding", ["iid", "by_author"])
def test_shards_are_disjoint_and_cover_retain(sharding):
    cfg = _fed_cfg(sharding)
    splits = data.load_splits(cfg)
    clients = federated.shard_clients(splits, cfg)
    assert len(clients) == cfg.federated.num_clients and clients[0] is splits.forget
    idx = [i for c in clients[1:] for i in c.indices]
    assert sorted(idx) == list(range(len(splits.retain)))  # disjoint and complete
    if sharding == "by_author":
        owners = {}
        for k, c in enumerate(clients[1:]):
            for i in c.indices:
                owners.setdefault(splits.retain.author_ids[i], set()).add(k)
        assert all(len(v) == 1 for v in owners.values())  # every author on one client
        assert set(owners) == set(splits.retain.author_ids)
    else:
        sizes = [len(c) for c in clients[1:]]
        assert max(sizes) - min(sizes) <= 1
    again = federated.shard_clients(splits, cfg)
    assert all(a.indices == b.indices for a, b in zip(clients[1:], again[1:], strict=True))
    cfg.seed += 1
    assert any(a.indices != b.indices for a, b in zip(clients[1:], federated.shard_clients(splits, cfg)[1:],
                                                      strict=True))


def test_by_author_needs_author_ids():
    cfg = load_config("configs/smoke_fed_synthetic.yaml")
    cfg.federated.sharding = "by_author"
    with pytest.raises(ValueError):
        federated.shard_clients(data.load_splits(cfg), cfg)


def _tiny(cfg):
    set_seed(0)
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    return model


def test_project_to_ball():
    cfg = _fed_cfg()
    model = _tiny(cfg)
    center = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    before = federated._distance(model, center)
    assert federated.project_to_ball(model, center, 2 * before) == before
    assert federated._distance(model, center) == before  # inside the ball: unchanged
    federated.project_to_ball(model, center, 0.5 * before)
    assert abs(federated._distance(model, center) / (0.5 * before) - 1) < 1e-5


def test_client_ga_stays_in_ball(monkeypatch):
    cfg = _fed_cfg()
    cfg.unlearn.learning_rate = 1e-2  # large steps, so the projection must act
    model = _tiny(cfg)
    splits = data.load_splits(cfg)
    center = {n: p.detach().clone() for n, p in model.state_dict().items()}
    rho = 0.05
    after = []
    original = federated.project_to_ball

    def recorded(m, c, r):
        before = original(m, c, r)
        after.append((before, federated._distance(m, c)))
        return before
    monkeypatch.setattr(federated, "project_to_ball", recorded)
    federated.client_gradient_ascent(model, splits.forget, cfg, "cpu", center, rho)
    assert after and all(d <= rho * (1 + 1e-5) for _, d in after)
    assert any(b > rho for b, _ in after)  # the projection was active


def test_fedavg_round_is_weighted_average():
    cfg = _fed_cfg()
    splits = data.load_splits(cfg)
    clients = federated.shard_clients(splits, cfg)
    model = _tiny(cfg)
    start = {k: v.clone() for k, v in model.state_dict().items()}
    set_seed(1)
    norms = federated.fedavg_rounds(model, clients, cfg, "cpu", 1)
    states, sizes = [], []
    worker = _tiny(cfg)
    set_seed(1)
    for c in clients:
        worker.load_state_dict(start)
        unlearn.train_lm(worker, c, cfg, "cpu", cfg.federated.local_epochs,
                         cfg.unlearn.finetune_lr or cfg.unlearn.learning_rate)
        states.append({k: v.clone() for k, v in worker.state_dict().items()})
        sizes.append(len(c))
    ref = _weighted_average(states, sizes)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, ref[k], rtol=0, atol=1e-6)
    assert abs(norms[0] - federated._distance(model, start)) < 1e-6


@pytest.fixture(scope="module")
def fed_run():
    cfg = _fed_cfg()
    calls = []
    original = unlearn.train_lm

    def recorded(model, dataset, *a, **k):
        calls.append(dataset)
        return original(model, dataset, *a, **k)
    unlearn.train_lm = recorded
    try:
        res = federated.run_fed(cfg, "cpu")
    finally:
        unlearn.train_lm = original
    return cfg, res, calls


def test_run_fed_trains_each_stage_on_the_right_clients(fed_run):
    cfg, res, calls = fed_run
    k, r = cfg.federated.num_clients, cfg.federated.rounds
    assert len(calls) == r * k + r * (k - 1) + cfg.federated.recovery_rounds * (k - 1)
    forget = calls[0]  # client 0 trains first in every FedAvg round of θ_pre
    assert sum(c is forget for c in calls) == r  # never in retrain or recovery
    assert res["federated"]["client_sizes"][0] == len(forget)


def test_run_fed_rows(fed_run):
    cfg, res, _ = fed_run
    rows = federated.fed_rows(cfg, res)
    assert set(res["post"]) == {"retrain", "client_ga_upload", "client_ga"}
    assert len(rows) == 3 * 3 * 2 and len({tuple(r) for r in rows}) == 1  # same columns in every row
    fed = res["federated"]
    assert fed["rho"] == cfg.federated.rho_factor * fed["round_update_norms"][-1]
    assert len(fed["round_update_norms"]) == cfg.federated.rounds
    for r in rows:
        assert r["view"] == federated.VIEWS[r["method"]]
        assert all(np.isfinite(r[k]) for k in ("auc_post", "gap_post", "auc_score_change", "auc_grad_align",
                                               "auc_grad_align_random", "auc_post_outlier_basediff"))
        if r["method"] == "client_ga_upload":
            assert r["model_utility_post"] is None and r["delta_norm"] <= fed["rho"] * (1 + 1e-5)
        else:
            assert 0 <= r["forget_quality_post"] <= 1 and np.isfinite(r["model_utility_post"])
    assert {r["forget_quality_post"] for r in rows if r["method"] == "retrain"} == {1.0}
    assert res["strata"]["primary"]["basediff"]["forget"]  # strata from the base model


def test_run_fed_scores_roundtrip(fed_run, tmp_path):
    from unlearning import central

    _, res, _ = fed_run
    central.save_scores(res, tmp_path / "fed.npz")
    z = np.load(tmp_path / "fed.npz")
    assert np.array_equal(z["post_client_ga_upload__grad_align__forget"],
                          res["post"]["client_ga_upload"]["alignment"]["scores"]["delta"]["forget"].numpy())
    assert "post_client_ga__truth_ratio__forget" in z and "post_client_ga_upload__truth_ratio__forget" not in z


def _cuda_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


@pytest.mark.skipif(not _cuda_supported(), reason="needs a GPU this PyTorch build supports")
def test_run_fed_fp16_amp_on_cuda_is_finite():
    cfg = _fed_cfg()
    cfg.model.precision = "fp16_amp"
    cfg.unlearn.train_batch_size = 16
    rows = federated.fed_rows(cfg, federated.run_fed(cfg, "cuda"))
    assert all(np.isfinite(r["auc_post"]) and np.isfinite(r["auc_grad_align"]) for r in rows)
    assert all(r["delta_norm"] <= r["rho"] * (1 + 1e-4) for r in rows if r["method"] == "client_ga_upload")
