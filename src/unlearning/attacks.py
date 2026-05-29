"""Membership-inference attacks and the token-level log-probability helpers
shared with the NPO unlearning objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import Splits, collate


def _token_logprobs(model, batch: dict[str, torch.Tensor], device: str):
    """Return per-token log-probs [B, T-1] of the realised next tokens, and a mask."""
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    logits = model(input_ids=ids, attention_mask=mask).logits
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = ids[:, 1:]
    tok = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return tok, mask[:, 1:]


def sequence_avg_logprob(model, batch, device: str) -> torch.Tensor:
    """Mean per-token log-prob for each sequence [B] (higher = more memorised)."""
    tok, m = _token_logprobs(model, batch, device)
    return (tok * m).sum(-1) / m.sum(-1).clamp(min=1)


def min_k_score(model, batch, device: str, k: float) -> torch.Tensor:
    """Min-K%% Prob score: mean of the lowest-k fraction of token log-probs [B]."""
    tok, m = _token_logprobs(model, batch, device)
    scores = []
    for row, rmask in zip(tok, m, strict=False):
        vals = row[rmask.bool()]
        n = max(1, int(k * vals.numel()))
        scores.append(torch.topk(vals, n, largest=False).values.mean())
    return torch.stack(scores)


@torch.no_grad()
def score_dataset(model, dataset, cfg: ExperimentConfig, device: str) -> torch.Tensor:
    """Per-record membership score (higher = more likely a member)."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.unlearn.batch_size, collate_fn=collate
    )
    out = []
    for batch in loader:
        if cfg.attack.method == "min_k_prob":
            out.append(min_k_score(model, batch, device, cfg.attack.min_k))
        else:  # loss_threshold: higher avg log-prob (lower loss) => member
            out.append(sequence_avg_logprob(model, batch, device))
    return torch.cat(out).cpu()


def run_attack(model, splits: Splits, cfg: ExperimentConfig, device: str) -> dict:
    """Return membership scores for forget (members) and holdout (non-members)."""
    return {
        "forget": score_dataset(model, splits.forget, cfg, device),
        "holdout": score_dataset(model, splits.holdout, cfg, device),
        "forget_is_outlier": list(splits.forget_is_outlier),
    }


@torch.no_grad()
def memorisation_scores(model, dataset, cfg: ExperimentConfig, device: str) -> torch.Tensor:
    """Per-record memorisation (mean log-prob); higher = more memorised."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.unlearn.batch_size, collate_fn=collate
    )
    out = [sequence_avg_logprob(model, batch, device) for batch in loader]
    return torch.cat(out).cpu()


def tag_outliers(model, splits: Splits, cfg: ExperimentConfig, device: str) -> None:
    """Flag the most-memorised forget records as the high-risk minority.

    Outliers are defined on the fine-tuned (pre-unlearning) model as the
    top `outlier_fraction` of forget records by memorisation; detectability is
    later evaluated on the post-unlearning model, so the tagging is not
    circular with the attack it informs.
    """
    if cfg.data.outlier_fraction <= 0 or len(splits.forget) == 0:
        return
    scores = memorisation_scores(model, splits.forget, cfg, device)
    n_out = max(1, int(cfg.data.outlier_fraction * len(scores)))
    top = set(torch.topk(scores, n_out, largest=True).indices.tolist())
    splits.forget_is_outlier = [i in top for i in range(len(scores))]
