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

`run_fed` is the revision path on the official splits: client 0 holds
forget10 and runs projected gradient ascent itself (`client_ga`); the
reference is federated retraining over clients 1..K-1. It reports the same
attacks and TOFU metrics as `central.run_central`, and the update-signal
attacks for client 0's upload (server view) and the global model (any
client). `run_federated` with `approx` is kept unchanged as the legacy path.
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


# Revision path (`run_fed`): client-level unlearning on the official splits,
# with every attack of the central runs and the update-signal attacks for the
# server's and the clients' view. The legacy functions above are unchanged.

FED_METHODS = ("retrain", "client_ga")
# stage -> who observes the change from θ_pre: the server sees client 0's upload,
# every client receives the global models
VIEWS = {"retrain": "any_client", "client_ga_upload": "server", "client_ga": "any_client"}
FED_ALIGN_OFFSET = 5  # random-control seeds: central.ALIGN_SEED + 10 * seed + 5 + stage index
SHARD_SEED = 30_000   # retain sharding: default_rng(SHARD_SEED + seed)


def shard_clients(splits: Splits, cfg: ExperimentConfig) -> list:
    """Client 0 = the forget set; clients 1..K-1 split the retain set.

    `iid`: a seeded shuffle of the retain QAs, cut into K-1 near-equal shards.
    `by_author`: a seeded shuffle of the retain authors, cut into K-1 groups
    of near-equal author counts (non-IID). The shuffle depends on the seed.
    """
    k = max(2, cfg.federated.num_clients)
    rng = np.random.default_rng(SHARD_SEED + cfg.seed)
    retain = splits.retain
    if cfg.federated.sharding == "iid":
        groups = np.array_split(rng.permutation(len(retain)), k - 1)
    elif cfg.federated.sharding == "by_author":
        authors = getattr(retain, "author_ids", None)
        if authors is None:
            raise ValueError("by_author sharding needs author ids (tofu_official or synthetic_official).")
        order = rng.permutation(sorted(set(authors)))
        if len(order) < k - 1:
            raise ValueError(f"by_author sharding: {len(order)} retain authors for {k - 1} clients.")
        owner = {int(a): i for i, grp in enumerate(np.array_split(order, k - 1)) for a in grp}
        groups = [[j for j, a in enumerate(authors) if owner[a] == i] for i in range(k - 1)]
    else:
        raise ValueError(f"Unknown sharding '{cfg.federated.sharding}'; expected iid or by_author.")
    return [splits.forget] + [Subset(retain, sorted(int(j) for j in g)) for g in groups]


def _distance(model, ref: dict) -> float:
    """‖θ − ref‖ over the model's parameters (fp32 per parameter)."""
    return float(sum(float((p.detach().float() - ref[n].to(p.device, torch.float32)).pow(2).sum())
                     for n, p in model.named_parameters())) ** 0.5


def fedavg_rounds(model, clients: list, cfg: ExperimentConfig, device: str, rounds: int) -> list[float]:
    """FedAvg in place on `model`; returns the norm of each round's update.

    Each round, every client starts from the global weights and trains for
    `local_epochs` with a fresh optimizer (`train_lm`: `train_batch_size`
    and fp16 AMP as configured); the server averages the states weighted by
    client size. The global state and the weighted sum stay on the training
    device when `accumulate_on_device` is set, on CPU otherwise.
    """
    lr = cfg.unlearn.finetune_lr or cfg.unlearn.learning_rate
    where = device if cfg.federated.accumulate_on_device else "cpu"
    active = [(c, len(c)) for c in clients if len(c) > 0]
    model.to(device)
    dtypes = {k: v.dtype for k, v in model.state_dict().items()}
    norms = []
    for _ in range(rounds):
        g_state = {k: v.detach().to(where, copy=True) for k, v in model.state_dict().items()}
        acc, total = None, 0.0
        for dataset, size in active:
            model.load_state_dict(g_state)
            unlearn.train_lm(model, dataset, cfg, device, cfg.federated.local_epochs, lr)
            total += size
            w_state = model.state_dict()
            if acc is None:
                acc = {k: v.detach().to(where, torch.float32) * size for k, v in w_state.items()}
            else:
                for k, v in w_state.items():
                    acc[k] += v.detach().to(where, torch.float32) * size
        model.load_state_dict({k: (v / total).to(dtypes[k]) for k, v in acc.items()})
        norms.append(_distance(model, g_state))
    return norms


@torch.no_grad()
def project_to_ball(model, center: dict, rho: float) -> float:
    """Project the parameters onto {θ : ‖θ − center‖ ≤ rho} in place (the full
    parameter vector, one scale for all); returns the distance before."""
    dist = _distance(model, center)
    if dist > rho:
        scale = rho / dist
        for n, p in model.named_parameters():
            c = center[n].to(p.device, torch.float32)
            p.copy_((c + (p.float() - c) * scale).to(p.dtype))
    return dist


def client_gradient_ascent(model, dataset, cfg: ExperimentConfig, device: str, center: dict, rho: float) -> None:
    """Client 0's unlearning: gradient ascent on its own data.

    `client_unlearn_epochs` epochs at `unlearn.learning_rate`, batch
    `unlearn.batch_size`, AdamW, fp16 AMP as configured; after every step
    the parameters are projected onto the L2 ball of radius `rho` around
    `center` (θ_pre), after Halimi et al.
    """
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.unlearn.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=unlearn._use_amp(cfg, device))
    center = {n: center[n].to(device, torch.float32) for n, _ in model.named_parameters()}
    for _ in range(cfg.federated.client_unlearn_epochs):
        for batch in unlearn._loader(dataset, cfg):
            opt.zero_grad()
            with unlearn._autocast(cfg, device):
                loss = -model(input_ids=batch["input_ids"].to(device),
                              attention_mask=batch["attention_mask"].to(device),
                              labels=batch["labels"].to(device)).loss
            if not torch.isfinite(loss):
                continue
            unlearn._step(loss, model, opt, scaler)
            project_to_ball(model, center, rho)


def run_fed(cfg: ExperimentConfig, device: str, methods=FED_METHODS) -> dict:
    """FedAvg once per (model, seed), then federated retrain and client-level GA.

    Same result layout as `central.run_central`, so `central_rows` and
    `save_scores` apply. `pre` is the FedAvg global model θ_pre. Stages
    under "post":
      - "retrain": FedAvg from the base model over clients 1..K-1 for
        `rounds` rounds (the reference for gaps and forget quality);
      - "client_ga_upload": client 0's upload after projected gradient
        ascent (server view, no TOFU metrics);
      - "client_ga": the global model after the server adopts the upload
        and runs `recovery_rounds` FedAvg rounds over clients 1..K-1.
    Each stage gets the score change from θ_pre and, with
    `attack.grad_alignment`, the gradient alignment with its Δθ (θ_retrain,
    the upload or θ_post, minus θ_pre) and a random control. Outlier strata
    come from the base model (`outliers.tag_splits`); the radius is
    rho = `rho_factor` x the norm of the last FedAvg round's update.
    """
    from . import central, data, outliers, tofu_metrics

    unknown = [m for m in methods if m not in FED_METHODS]
    if unknown:
        raise ValueError(f"Unknown federated methods {unknown}; expected a subset of {FED_METHODS}.")
    if cfg.federated.rounds < 1:
        raise ValueError("run_fed needs at least one FedAvg round.")

    set_seed(cfg.seed)
    model, tokenizer = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg, tokenizer)
    eval_sets = tofu_metrics.load_eval_sets(cfg, tokenizer, splits)
    clients = shard_clients(splits, cfg)
    base = central._snapshot(model) if "retrain" in methods else None
    with torch.random.fork_rng(devices=[]):  # scoring must not shift the FedAvg data order
        base_scores = central._scores(model, splits, cfg, device)
        outliers.tag_splits(splits, cfg, model, device, base_logprobs=base_scores["loss_threshold"])
    norms = fedavg_rounds(model, clients, cfg, device, cfg.federated.rounds)
    if not any(splits.forget_is_outlier):
        attacks.tag_outliers(model, splits, cfg, device)
    pre_state = central._snapshot(model)
    base_loss = base_scores["loss_threshold"]
    pre = central._evaluate(model, splits, cfg, device, base_loss, eval_sets)
    outliers.add_legacy(splits, cfg, model, device, finetuned_logprobs=pre["scores"]["loss_threshold"])

    sets = {"forget": splits.forget, **data.nonmember_sets(splits)}
    rho = cfg.federated.rho_factor * norms[-1]
    result = {"base": {"scores": base_scores}, "pre": pre, "post": {},
              "nonmember_sets": list(data.nonmember_sets(splits)),
              "row_ids": {k: central._row_ids(ds) for k, ds in sets.items()},
              "forget_is_outlier": list(splits.forget_is_outlier), "strata": splits.strata,
              "federated": {"client_sizes": [len(c) for c in clients], "round_update_norms": norms, "rho": rho}}

    def evaluate_stage(stage: str, tofu) -> None:
        result["post"][stage] = central._evaluate(model, splits, cfg, device, base_loss, tofu)
        if cfg.attack.grad_alignment:  # leaves the model at θ_pre
            result["post"][stage]["alignment"] = central._alignment(
                model, pre_state, splits, cfg, device,
                central.ALIGN_SEED + 10 * cfg.seed + FED_ALIGN_OFFSET + list(VIEWS).index(stage))

    for method in methods:
        set_seed(cfg.seed)
        if method == "retrain":
            model.load_state_dict(base)
            fedavg_rounds(model, clients[1:], cfg, device, cfg.federated.rounds)
            evaluate_stage("retrain", eval_sets)
        else:
            model.load_state_dict(pre_state)
            client_gradient_ascent(model, clients[0], cfg, device, pre_state, rho)
            upload = central._snapshot(model)
            evaluate_stage("client_ga_upload", None)
            model.load_state_dict(upload)  # the server adopts the upload
            fedavg_rounds(model, clients[1:], cfg, device, cfg.federated.recovery_rounds)
            evaluate_stage("client_ga", eval_sets)
    return result


def fed_rows(cfg: ExperimentConfig, result: dict) -> list[dict]:
    """`central.central_rows` for a `run_fed` result, plus the federated
    settings: `view` (who observes the stage's change), clients, sharding,
    rounds, recovery rounds, client epochs, `rho_factor`, `rho` and the last
    FedAvg round's update norm. Every row has the same columns (the upload
    stage has no TOFU metrics: None)."""
    from . import central

    rows = central.central_rows(cfg, result)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    fed = cfg.federated
    extra = {"num_clients": fed.num_clients, "sharding": fed.sharding, "rounds": fed.rounds,
             "recovery_rounds": fed.recovery_rounds, "client_unlearn_epochs": fed.client_unlearn_epochs,
             "rho_factor": fed.rho_factor, "rho": result["federated"]["rho"],
             "round_update_norm": result["federated"]["round_update_norms"][-1]}
    out = []
    for r in rows:
        row = {k: r.get(k) for k in keys}
        row = {**{k: row.pop(k) for k in ("model", "seed", "method")}, "view": VIEWS[r["method"]], **row, **extra}
        out.append(row)
    return out
