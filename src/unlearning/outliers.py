"""Outlier definitions that do not depend on the target (fine-tuned) model.

* `basediff`: base-model difficulty, the mean answer-token NLL under the
  pre-trained model before fine-tuning.
* `rarity`: mean -log unigram frequency of the answer tokens, with frequencies
  counted over the TOFU corpus (all records of every split, as encoded for
  training; for `tofu_official` that is full + holdout10).
* `legacy`: the submitted definition, the most memorized forget records on the
  fine-tuned model (forget only; compared against all non-members).
* `legacy_pooled`: memorization on the fine-tuned model, with forget and
  non-members tagged by one pooled threshold (ablation: definition vs matching).

`basediff`, `rarity` and `legacy_pooled` tag forget and one non-member set
with one pooled threshold (the top `outlier_fraction` of the two together),
so outlier-forget can be compared with outlier-non-members and typical with
typical (stratum-matched AUC). This is done separately for each non-member
set, so `splits.strata` is keyed by set name, then by definition.
"""

from __future__ import annotations

import numpy as np
import torch

DEFINITIONS = ("basediff", "rarity", "legacy", "legacy_pooled")


def base_difficulty(base_model, datasets, cfg, device: str) -> list[np.ndarray]:
    """Mean scored-token NLL of each record under `base_model` (higher = harder)."""
    from .attacks import memorisation_scores

    return [-memorisation_scores(base_model, ds, cfg, device).double().numpy() for ds in datasets]


def _scored_tokens(dataset):
    for i in range(len(dataset)):
        lab = dataset[i]["labels"]
        yield lab[lab != -100]


def token_rarity(corpus_ids, datasets) -> list[np.ndarray]:
    """Mean -log unigram frequency of each record's scored tokens.

    Frequencies are counted over `corpus_ids` (token-id sequences) with add-one
    smoothing, so a token missing from the corpus still gets a finite value.
    """
    seqs = [torch.as_tensor(s).long() for s in corpus_ids]
    vocab = 1 + max(int(s.max()) for s in seqs if len(s))
    vocab = max([vocab] + [1 + int(t.max()) for ds in datasets for t in _scored_tokens(ds) if len(t)])
    counts = torch.bincount(torch.cat(seqs), minlength=vocab).double()
    neglog = -torch.log((counts + 1) / (counts.sum() + vocab))
    return [np.array([float(neglog[t].mean()) if len(t) else float("nan") for t in _scored_tokens(ds)])
            for ds in datasets]


def tag_pooled(forget: np.ndarray, holdout: np.ndarray, fraction: float) -> tuple[list[bool], list[bool]]:
    """Top `fraction` of the pooled forget + holdout values; one threshold for both sets."""
    pooled = np.concatenate([np.asarray(forget, dtype=float), np.asarray(holdout, dtype=float)])
    k = int(fraction * len(pooled))
    top = np.zeros(len(pooled), dtype=bool)
    if k > 0:
        top[np.argsort(-pooled, kind="stable")[:k]] = True
    return top[:len(forget)].tolist(), top[len(forget):].tolist()


def _pooled(f, h, fraction: float) -> dict:
    f_tag, h_tag = tag_pooled(f, h, fraction)
    return {"forget": f_tag, "holdout": h_tag, "forget_value": f, "holdout_value": h}


def _mean_logprobs(model, splits, cfg, device: str, given: dict | None) -> dict:
    """Mean scored-token log-prob per record for forget and every non-member set."""
    from .attacks import memorisation_scores
    from .data import nonmember_sets

    sets = {"forget": splits.forget, **nonmember_sets(splits)}
    if given is not None:
        return {k: np.asarray(given[k], dtype=float) for k in sets}
    return {k: memorisation_scores(model, ds, cfg, device).double().numpy() for k, ds in sets.items()}


def tag_splits(splits, cfg, base_model, device: str, base_logprobs: dict | None = None) -> None:
    """Add the `basediff` and `rarity` strata for every non-member set (call before fine-tuning).

    `base_logprobs` (set name -> base-model mean log-prob per record, including
    "forget") avoids scoring the base model again when the caller already has it.
    """
    from .data import nonmember_sets

    nm = nonmember_sets(splits)
    diff = {k: -v for k, v in _mean_logprobs(base_model, splits, cfg, device, base_logprobs).items()}
    corpus = [ds[i]["input_ids"] for ds in (splits.forget, splits.retain, *nm.values()) for i in range(len(ds))]
    rare = dict(zip(["forget", *nm], token_rarity(corpus, [splits.forget, *nm.values()]), strict=True))
    for name in nm:
        splits.strata[name] = {
            "basediff": _pooled(diff["forget"], diff[name], cfg.data.outlier_fraction),
            "rarity": _pooled(rare["forget"], rare[name], cfg.data.outlier_fraction),
        }


def add_legacy(splits, cfg, finetuned_model, device: str, finetuned_logprobs: dict | None = None) -> None:
    """Add the `legacy` stratum (the `forget_is_outlier` tag, against all
    non-members) and the `legacy_pooled` stratum (memorization on the
    fine-tuned model, pooled) for every non-member set."""
    from .data import nonmember_sets

    mem = _mean_logprobs(finetuned_model, splits, cfg, device, finetuned_logprobs)
    for name in nonmember_sets(splits):
        strata = splits.strata.setdefault(name, {})
        strata["legacy"] = {"forget": list(splits.forget_is_outlier), "holdout": None}
        strata["legacy_pooled"] = _pooled(mem["forget"], mem[name], cfg.data.outlier_fraction)
