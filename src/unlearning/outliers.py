"""Outlier definitions that do not depend on the target (fine-tuned) model.

* `basediff`: base-model difficulty, the mean answer-token NLL under the
  pre-trained model before fine-tuning.
* `rarity`: mean -log unigram frequency of the answer tokens, with frequencies
  counted over the TOFU corpus (members + holdout, as encoded for training).
* `legacy`: the submitted definition, the most memorized forget records on the
  fine-tuned model (forget only; compared against all holdout records).
* `legacy_pooled`: memorization on the fine-tuned model, with forget and
  holdout tagged by one pooled threshold (ablation: definition vs matching).

`basediff` and `rarity` tag forget and holdout with one pooled threshold (the
top `outlier_fraction` of forget + holdout), so outlier-forget can be compared
with outlier-holdout and typical with typical (stratum-matched AUC).
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


def tag_splits(splits, cfg, base_model, device: str) -> None:
    """Add the `basediff` and `rarity` strata to `splits` (call before fine-tuning)."""
    f_diff, h_diff = base_difficulty(base_model, [splits.forget, splits.holdout], cfg, device)
    corpus = [ds[i]["input_ids"] for ds in (splits.forget, splits.retain, splits.holdout)
              for i in range(len(ds))]
    f_rare, h_rare = token_rarity(corpus, [splits.forget, splits.holdout])
    for name, f, h in (("basediff", f_diff, h_diff), ("rarity", f_rare, h_rare)):
        f_tag, h_tag = tag_pooled(f, h, cfg.data.outlier_fraction)
        splits.strata[name] = {"forget": f_tag, "holdout": h_tag, "forget_value": f, "holdout_value": h}


def add_legacy(splits, cfg, finetuned_model, device: str) -> None:
    """Add the `legacy` stratum (the `forget_is_outlier` tag) and the
    `legacy_pooled` stratum (memorization on the fine-tuned model, pooled)."""
    from .attacks import memorisation_scores

    splits.strata["legacy"] = {"forget": list(splits.forget_is_outlier), "holdout": None}
    f, h = (memorisation_scores(finetuned_model, ds, cfg, device).double().numpy()
            for ds in (splits.forget, splits.holdout))
    f_tag, h_tag = tag_pooled(f, h, cfg.data.outlier_fraction)
    splits.strata["legacy_pooled"] = {"forget": f_tag, "holdout": h_tag, "forget_value": f, "holdout_value": h}
