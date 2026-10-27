"""Update-signal attacks: does the unlearning update reveal the forget set?

Two attackers who see the model before (θ_pre) and after (θ_post) unlearning:

- black-box: the per-record change of a membership score, pre − post
  (`score_change_auc`). Scores are higher for members, and unlearning lowers
  them on the forget set, so forget records get the larger change.
- white-box: the cosine between a record's loss gradient at θ_pre and the
  update Δθ = θ_post − θ_pre (`grad_alignment_scores`).

The scores of both attackers are higher for records that are more likely to
have been forgotten, so AUC > 0.5 means the update leaks the forget set.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import numpy as np
import torch

from .data import collate_for
from .evaluate import _auc


def score_change(pre_scores: dict, post_scores: dict) -> dict:
    """Per-record score change pre − post for every set in `pre_scores`."""
    return {k: np.asarray(pre_scores[k], dtype=float) - np.asarray(post_scores[k], dtype=float)
            for k in pre_scores}


def score_change_auc(pre_scores: dict, post_scores: dict) -> dict:
    """AUC of the score change (pre − post), forget against each non-member set.

    `pre_scores` and `post_scores` map "forget" and the non-member set names
    to per-record membership scores (higher = member). Returns
    {non-member set: AUC}.
    """
    change = score_change(pre_scores, post_scores)
    return {k: _auc(change["forget"], v) for k, v in change.items() if k != "forget"}


def _params(model, param_filter: Callable[[str], bool] | None) -> dict:
    return {n: p for n, p in model.named_parameters()
            if p.requires_grad and (param_filter is None or param_filter(n))}


def parameter_delta(model_at_post, pre_state: dict, param_filter: Callable[[str], bool] | None = None) -> dict:
    """Δθ = θ_post − θ_pre (fp32, on the model's device) for the parameters
    `grad_alignment_scores` uses. `pre_state` is a state dict of θ_pre (CPU is fine)."""
    return {n: p.detach().float() - pre_state[n].to(p.device, torch.float32)
            for n, p in _params(model_at_post, param_filter).items()}


def random_direction(delta: dict, seed: int) -> dict:
    """Gaussian direction over the same parameters as `delta`, scaled to ‖Δθ‖.

    Negative control for the white-box attack: a direction that carries no
    information about the forget set, so its alignment AUC should be ≈ 0.5.
    Drawn on CPU from its own generator, so it does not move the global RNG.
    """
    gen = torch.Generator().manual_seed(seed)
    r = {n: torch.randn(d.shape, generator=gen).to(d.device) for n, d in delta.items()}
    scale = _norm(delta) / _norm(r)
    return {n: v * scale for n, v in r.items()}


def _norm(direction: dict) -> float:
    return float(sum(v.double().pow(2).sum() for v in direction.values()) ** 0.5)


@contextlib.contextmanager
def fp32_weights(model):
    """Run with every floating parameter and buffer in fp32, then restore the
    original dtypes (e.g. the fp16 base weights under LoRA). fp16 -> fp32 ->
    fp16 is exact, so the weights come back unchanged. No-op for fp32 models."""
    tensors = [t for t in (*model.parameters(), *model.buffers()) if t.is_floating_point()]
    dtypes = [t.dtype for t in tensors]
    if all(d == torch.float32 for d in dtypes):
        yield model
        return
    for t in tensors:
        t.data = t.data.float()
    try:
        yield model
    finally:
        for t, d in zip(tensors, dtypes, strict=True):
            t.data = t.data.to(d)


def grad_alignment_scores(model_at_pre, delta, dataset, device: str,
                          param_filter: Callable[[str], bool] | None = None):
    """Cosine between each record's loss gradient at θ_pre and Δθ.

    Sign convention: the loss of a record is its mean answer-token NLL, i.e.
    minus the `loss_threshold` score. Gradient ascent on a record moves θ
    along +∇L, so a record that was unlearned has a positive cosine with Δθ;
    the score is cos(∇_θ L(x; θ_pre), Δθ), higher = more likely forgotten.
    GD and NPO also descend the retain loss, so retain-like records can get a
    negative cosine.

    Batch size 1, fp32 (base weights in another dtype are upcast for the
    pass, `fp32_weights`), model in eval mode (no dropout). The dot product
    and the norms are accumulated parameter by parameter (fp32 per
    parameter, summed as Python floats), so no flat gradient is built.
    `delta` is a dict {parameter name: fp32 tensor} (see `parameter_delta`)
    or a dict of such dicts {label: direction}, which returns {label: scores}
    from one backward pass per record. Only parameters that require grad and
    pass `param_filter` (e.g. the LoRA weights) are used; every direction
    must cover exactly those. A record without scored tokens gets NaN.
    """
    multi = all(isinstance(v, dict) for v in delta.values())
    directions = delta if multi else {None: delta}
    params = _params(model_at_pre, param_filter)
    for label, d in directions.items():
        if set(d) != set(params):
            raise ValueError(f"Direction {label!r} does not cover the same parameters as the model.")
    d_norm = {label: _norm(d) for label, d in directions.items()}
    names, tensors = list(params), list(params.values())
    model_at_pre.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_for(dataset))
    out = {label: [] for label in directions}
    with fp32_weights(model_at_pre):
        _accumulate(model_at_pre, loader, names, tensors, directions, d_norm, device, out)
    scores = {label: torch.tensor(v, dtype=torch.float64) for label, v in out.items()}
    return scores if multi else scores[None]


def _accumulate(model_at_pre, loader, names, tensors, directions, d_norm, device, out) -> None:
    from .attacks import _mean_of, _token_logprobs

    for batch in loader:
        with torch.enable_grad():
            loss = -_mean_of(*_token_logprobs(model_at_pre, batch, device)).sum()
            grads = torch.autograd.grad(loss, tensors, allow_unused=True)
        g_sq = 0.0
        dots = dict.fromkeys(directions, 0.0)
        for n, g in zip(names, grads, strict=True):
            if g is None:
                continue
            g = g.float().flatten()
            g_sq += float(torch.dot(g, g))
            for label, d in directions.items():
                dots[label] += float(torch.dot(g, d[n].flatten()))
        for label in directions:
            den = g_sq ** 0.5 * d_norm[label]
            out[label].append(dots[label] / den if den > 0 else float("nan"))
