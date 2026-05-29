"""Metrics: MIA AUC (overall + minority slice) and forget/retain utility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(member: np.ndarray, nonmember: np.ndarray) -> float:
    """AUC of a membership score (member=1, non-member=0)."""
    if len(member) == 0 or len(nonmember) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    s = np.concatenate([member, nonmember])
    return float(roc_auc_score(y, s))


def mean_loss(model, dataset, cfg, device: str) -> float:
    """Average cross-entropy loss over a dataset (utility / forget quality)."""
    import torch
    from torch.utils.data import DataLoader

    from .data import collate

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=cfg.unlearn.batch_size, collate_fn=collate):
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            losses.append(float(out.loss))
    return float(np.mean(losses)) if losses else float("nan")


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
