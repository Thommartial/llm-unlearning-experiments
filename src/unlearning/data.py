"""Data loading: an offline synthetic source and the TOFU benchmark.

Setup (standard MIA-on-unlearning protocol):
  * partition the corpus into MEMBERS (used to fine-tune) and a never-seen
    HOLDOUT (MIA negatives);
  * within members, carve a FORGET set (fraction `forget_fraction`); the rest
    is the RETAIN set;
  * tag a subset of forget records as outliers to support the minority slice.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig


class TokenDataset(Dataset):
    """Holds tokenised sequences as input_ids/attention_mask/labels tensors."""

    def __init__(self, input_ids: list[torch.Tensor]):
        self.input_ids = input_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


@dataclass
class Splits:
    forget: TokenDataset
    retain: TokenDataset
    holdout: TokenDataset
    forget_is_outlier: list[bool] = field(default_factory=list)


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def collate(batch):  # exposed for DataLoader(collate_fn=...)
    return _collate(batch)


def _synthetic(cfg: ExperimentConfig) -> Splits:
    rng = random.Random(cfg.seed)
    n, length, vocab = cfg.data.n_synthetic, cfg.data.seq_len, cfg.data.vocab_size
    seqs = [torch.tensor([rng.randrange(vocab) for _ in range(length)]) for _ in range(n)]
    # Outliers contain a rare marker token (vocab-1) repeated, raising memorisation.
    n_out = int(cfg.data.outlier_fraction * n)
    for i in range(n_out):
        seqs[i][: length // 4] = vocab - 1
    rng.shuffle(seqs)  # note: outlier flags re-derived below after split
    return _partition(cfg, seqs, marker=vocab - 1)


def _partition(cfg: ExperimentConfig, seqs: list[torch.Tensor], marker: int | None) -> Splits:
    rng = random.Random(cfg.seed + 1)
    rng.shuffle(seqs)
    n = len(seqs)
    n_hold = max(1, int(cfg.data.holdout_fraction * n))
    holdout, members = seqs[:n_hold], seqs[n_hold:]
    n_forget = max(1, int(cfg.data.forget_fraction * len(members)))
    forget, retain = members[:n_forget], members[n_forget:]
    if marker is not None:
        outlier = [bool((s == marker).any().item()) for s in forget]
    else:
        outlier = [False] * len(forget)
    return Splits(TokenDataset(forget), TokenDataset(retain), TokenDataset(holdout), outlier)


def _tofu(cfg: ExperimentConfig, tokenizer) -> Splits:
    from datasets import load_dataset

    ds = load_dataset("locuslab/TOFU", "full")["train"]
    rows = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]
    if cfg.data.max_samples:
        rows = rows[: cfg.data.max_samples]
    enc = tokenizer(
        rows, truncation=True, max_length=cfg.data.max_length,
        padding="max_length", return_tensors="pt",
    )["input_ids"]
    seqs = [enc[i] for i in range(enc.shape[0])]
    return _partition(cfg, seqs, marker=None)


def load_splits(cfg: ExperimentConfig, tokenizer=None) -> Splits:
    """Return forget/retain/holdout splits for the configured dataset."""
    if cfg.data.dataset == "synthetic":
        return _synthetic(cfg)
    if cfg.data.dataset == "tofu":
        if tokenizer is None:
            raise ValueError("TOFU requires a tokenizer.")
        return _tofu(cfg, tokenizer)
    raise ValueError(f"Unknown dataset '{cfg.data.dataset}'.")
