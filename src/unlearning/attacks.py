"""Membership-inference attacks and the token-level log-probability helpers
shared with the NPO unlearning objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import Splits, collate_for


def _token_logprobs(model, batch: dict[str, torch.Tensor], device: str):
    """Return per-token log-probs [B, T-1] of the realised next tokens, and a mask.

    The mask comes from the labels (-100 = not scored): padding only on the
    legacy paths, padding and question tokens on the answer-only path.
    Log-probs are computed in fp32 with cross_entropy (0 at masked positions).
    """
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    logits = model(input_ids=ids, attention_mask=mask).logits
    # Targets shifted left and padded with -100 so the full (contiguous) logits can be
    # flattened without a copy; a transposed [B, V, T] input loses precision on GPT-2.
    target = F.pad(batch["labels"][:, 1:], (0, 1), value=-100).to(device)
    tok = -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                           reduction="none").view(target.shape)[:, :-1]
    return tok, (target[:, :-1] != -100).to(mask.dtype)


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
        dataset, batch_size=cfg.unlearn.eval_batch_size, collate_fn=collate_for(dataset)
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
        "strata": splits.strata,
    }


@torch.no_grad()
def memorisation_scores(model, dataset, cfg: ExperimentConfig, device: str) -> torch.Tensor:
    """Per-record memorisation (mean log-prob); higher = more memorised."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.unlearn.eval_batch_size, collate_fn=collate_for(dataset)
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
