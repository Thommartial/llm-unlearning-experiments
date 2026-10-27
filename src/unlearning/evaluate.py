"""Metrics: MIA AUC (overall + minority slice) and forget/retain utility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(member: np.ndarray, nonmember: np.ndarray) -> float:
    """AUC of a membership score (member=1, non-member=0); NaN-safe."""
    member = np.asarray(member, dtype=float)
    nonmember = np.asarray(nonmember, dtype=float)
    member = member[np.isfinite(member)]
    nonmember = nonmember[np.isfinite(nonmember)]
    if len(member) == 0 or len(nonmember) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    s = np.concatenate([member, nonmember])
    return float(roc_auc_score(y, s))


def mean_loss(model, dataset, cfg, device: str) -> float:
    """Average cross-entropy loss over a dataset (utility / forget quality).

    On the official-split datasets this is the mean over all scored (answer)
    tokens, at `eval_batch_size`, so it does not depend on the batch size. The
    legacy datasets keep the mean of per-batch means at `batch_size`.
    """
    import torch
    from torch.utils.data import DataLoader

    from .data import collate_for

    model.eval()
    if cfg.data.dataset.endswith("_official"):
        return _token_mean_loss(model, dataset, cfg, device)
    losses = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=cfg.unlearn.batch_size,
                                collate_fn=collate_for(dataset)):
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            losses.append(float(out.loss))
    return float(np.mean(losses)) if losses else float("nan")


def _token_mean_loss(model, dataset, cfg, device: str) -> float:
    """Token-weighted mean NLL over the scored tokens of `dataset`."""
    import torch
    from torch.utils.data import DataLoader

    from .attacks import _token_logprobs
    from .data import collate_for

    total, n = 0.0, 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=cfg.unlearn.eval_batch_size,
                                collate_fn=collate_for(dataset)):
            tok, m = _token_logprobs(model, batch, device)
            total -= float((tok * m).double().sum())
            n += int(m.sum())
    return total / n if n else float("nan")


def mia_metrics(scores: dict) -> dict:
    """Overall and minority-sliced MIA AUC, with attack advantage |AUC - 0.5|."""
    forget = np.asarray(scores["forget"])
    holdout = np.asarray(scores["holdout"])
    outlier = np.array(scores["forget_is_outlier"], dtype=bool)

    def entry(member, non):
        auc = _auc(member, non)
        return auc, abs(auc - 0.5)

    auc, adv = entry(forget, holdout)
    metrics = {"mia_auc": auc, "mia_advantage": adv}
    if outlier.any():
        metrics["mia_auc_outlier"], metrics["mia_advantage_outlier"] = entry(forget[outlier], holdout)
    if (~outlier).any():
        metrics["mia_auc_typical"], metrics["mia_advantage_typical"] = entry(forget[~outlier], holdout)
    # Stratum-matched: outlier-forget vs outlier-holdout, typical vs typical.
    for name, tags in scores.get("strata", {}).items():
        f_out = np.array(tags["forget"], dtype=bool)
        h_out = None if tags["holdout"] is None else np.array(tags["holdout"], dtype=bool)
        for stratum, f_sel, h_sel in (("outlier", f_out, h_out),
                                      ("typical", ~f_out, None if h_out is None else ~h_out)):
            h = holdout if h_sel is None else holdout[h_sel]
            metrics[f"mia_auc_{stratum}_{name}"], metrics[f"mia_advantage_{stratum}_{name}"] = \
                entry(forget[f_sel], h)
            metrics[f"n_forget_{stratum}_{name}"], metrics[f"n_holdout_{stratum}_{name}"] = int(f_sel.sum()), len(h)
    return metrics


def summarise(cfg, model, splits, pre_scores, post_scores, pre_losses, device, out_dir: Path) -> dict:
    """Assemble metrics (MIA + utility trade-off), persist them, and return them."""
    metrics = {
        "experiment": cfg.name,
        "method": cfg.unlearn.method,
        "attack": cfg.attack.method,
        "pre_unlearning": mia_metrics(pre_scores),
        "post_unlearning": mia_metrics(post_scores),
        "forget_loss_pre": pre_losses["forget"],
        "retain_loss_pre": pre_losses["retain"],
        "forget_loss_post": mean_loss(model, splits.forget, cfg, device),
        "retain_loss_post": mean_loss(model, splits.retain, cfg, device),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(
        out_dir / "scores.npz",
        pre_forget=np.asarray(pre_scores["forget"]), pre_holdout=np.asarray(pre_scores["holdout"]),
        post_forget=np.asarray(post_scores["forget"]), post_holdout=np.asarray(post_scores["holdout"]),
        forget_is_outlier=np.array(post_scores["forget_is_outlier"], dtype=bool),
    )
    return metrics
