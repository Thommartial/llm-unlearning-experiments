"""Federated fine-tuning and federated unlearning.

This extends the single-node pipeline to a federated setting while reusing the
existing training, attack, and evaluation code. The forget set is treated as
one client's data: client 0 holds the forget records, and clients 1..K-1 hold
disjoint shards of the retain set. A global model is trained by FedAvg over all
clients, then client 0's contribution is unlearned. The holdout stays as the
membership-inference non-members, so the existing attacks and metrics apply
unchanged to the post-unlearning global model.

Beyond the standard residual-leakage measurement, we also test the second-order
vulnerability directly: whether the unlearning update itself (the pre- to
post-unlearning change in per-record scores) reveals which records were
removed. In a distributed setting that update is observable to the server and,
in some protocols, to other clients, so its leakage matters.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from . import attacks, evaluate, unlearn
from .config import ExperimentConfig
from .data import Splits
from .seed import set_seed


def partition_clients(splits: Splits, cfg: ExperimentConfig) -> list:
    """Client 0 = the forget set; clients 1..K-1 = disjoint retain shards."""
    k = max(2, cfg.federated.num_clients)
    n_retain_clients = k - 1
    idx = list(range(len(splits.retain)))
    rng = np.random.RandomState(cfg.seed + 7)
    rng.shuffle(idx)
    shards = [idx[i::n_retain_clients] for i in range(n_retain_clients)]
    return [splits.forget] + [Subset(splits.retain, sorted(s)) for s in shards]


def _weighted_average(states: list[dict], weights: list[float]) -> dict:
    """Weighted average of a list of (CPU) state_dicts."""
    total = float(sum(weights))
    avg = copy.deepcopy(states[0])
    for key in avg:
        ref = states[0][key]
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for st, w in zip(states, weights, strict=False):
            acc += st[key].float() * (w / total)
        avg[key] = acc.to(ref.dtype)
    return avg


def fedavg_train(cfg: ExperimentConfig, clients: list, device: str,
                 rounds: int, local_epochs: int):
    """Train a global model by FedAvg over the client datasets."""
    global_model, _ = unlearn.build_model_and_tokenizer(cfg)
    worker, _ = unlearn.build_model_and_tokenizer(cfg)
    global_model.to(device)
    worker.to(device)
    lr = cfg.unlearn.finetune_lr or cfg.unlearn.learning_rate
    active = [(c, len(c)) for c in clients if len(c) > 0]

    dtypes = {k: v.dtype for k, v in global_model.state_dict().items()}
    for _ in range(rounds):
        g_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        acc, total = None, 0.0  # incremental weighted sum on CPU (memory-safe for large models)
        for dataset, size in active:
            worker.load_state_dict(g_state)
            unlearn.train_lm(worker, dataset, cfg, device, local_epochs, lr)
            total += size
            w_state = worker.state_dict()
            if acc is None:
                acc = {k: v.detach().float().cpu() * size for k, v in w_state.items()}
            else:
                for k, v in w_state.items():
                    acc[k] += v.detach().float().cpu() * size
        global_model.load_state_dict({k: (v / total).to(dtypes[k]) for k, v in acc.items()})

    del worker
    if device == "cuda":
        torch.cuda.empty_cache()
    return global_model


def federated_unlearn(global_model, splits: Splits, clients: list,
                      cfg: ExperimentConfig, device: str):
    """Remove client 0's contribution from the global model.

    ``approx``  : server-side approximate unlearning that reuses the configured
                  unlearn method (gradient ascent / difference / NPO).
    ``retrain`` : the gold reference, FedAvg re-run over the retain clients only.
    """
    method = cfg.federated.unlearn_method
    if method == "approx":
        return unlearn.apply_unlearning(global_model, splits, cfg, device)
    if method == "retrain":
        return fedavg_train(cfg, clients[1:], device,
                            cfg.federated.rounds, cfg.federated.local_epochs)
    raise ValueError(f"Unknown federated unlearn_method '{method}'.")


def update_signal_metrics(pre_scores: dict, post_scores: dict) -> dict:
    """Can the unlearning update reveal which records were removed?

    Uses the per-record change in membership score (pre minus post) as an
    attacker's signal: forgotten records lose the most score, holdout records
    barely move. A high AUC means the update itself leaks the forget set.
    """
    pre_f = np.asarray(pre_scores["forget"], dtype=float)
    post_f = np.asarray(post_scores["forget"], dtype=float)
    pre_h = np.asarray(pre_scores["holdout"], dtype=float)
    post_h = np.asarray(post_scores["holdout"], dtype=float)
    delta_forget = pre_f - post_f
    delta_holdout = pre_h - post_h
    auc = evaluate._auc(delta_forget, delta_holdout)
    return {"update_signal_auc": auc, "update_signal_advantage": abs(auc - 0.5)}


def run_federated(cfg: ExperimentConfig, out_dir: Path) -> dict:
    """Federated counterpart of cli.run: FedAvg train, attack, unlearn, attack."""
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = unlearn.resolve_device(cfg)
    print(f"[fed] device={device}  clients={cfg.federated.num_clients}  "
          f"rounds={cfg.federated.rounds}  unlearn={cfg.federated.unlearn_method}")

    from . import data

    probe, tokenizer = unlearn.build_model_and_tokenizer(cfg)
    del probe  # only the tokenizer is needed here; free the probe model
    splits = data.load_splits(cfg, tokenizer)
    clients = partition_clients(splits, cfg)
    print(f"[fed] forget(client 0)={len(splits.forget)}  "
          f"retain shards={[len(c) for c in clients[1:]]}  holdout={len(splits.holdout)}")

    global_model = fedavg_train(cfg, clients, device,
                                cfg.federated.rounds, cfg.federated.local_epochs)

    if not any(splits.forget_is_outlier):
        attacks.tag_outliers(global_model, splits, cfg, device)
    pre = attacks.run_attack(global_model, splits, cfg, device)
    pre_losses = {
        "forget": evaluate.mean_loss(global_model, splits.forget, cfg, device),
        "retain": evaluate.mean_loss(global_model, splits.retain, cfg, device),
    }
    print(f"[fed] pre-unlearning  MIA AUC = {evaluate.mia_metrics(pre)['mia_auc']:.3f}")

    global_model = federated_unlearn(global_model, splits, clients, cfg, device)
    post = attacks.run_attack(global_model, splits, cfg, device)

    metrics = evaluate.summarise(cfg, global_model, splits, pre, post, pre_losses, device, out_dir)
    metrics["federated"] = {
        "num_clients": cfg.federated.num_clients,
        "rounds": cfg.federated.rounds,
        "local_epochs": cfg.federated.local_epochs,
        "unlearn_method": cfg.federated.unlearn_method,
    }
    metrics.update(update_signal_metrics(pre, post))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[fed] post-unlearning MIA AUC = {metrics['post_unlearning']['mia_auc']:.3f}  "
          f"update-signal AUC = {metrics['update_signal_auc']:.3f}")
    return metrics
