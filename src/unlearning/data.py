"""Data loading: an offline synthetic source and the TOFU benchmark.

Setup (standard MIA-on-unlearning protocol):
  * partition the corpus into MEMBERS (used to fine-tune) and a never-seen
    HOLDOUT (MIA negatives);
  * within members, carve a FORGET set (fraction `forget_fraction`); the rest
    is the RETAIN set;
  * tag a subset of forget records as outliers to support the minority slice.

Padding is handled correctly: real attention masks are kept, and label
positions that correspond to padding are set to -100 so they are ignored by
the loss and by the membership scores.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig


class TokenDataset(Dataset):
    """Tokenised sequences with correct attention masks and -100 label padding."""

    def __init__(self, input_ids: list[torch.Tensor], attention_mask: list[torch.Tensor] | None = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        am = self.attention_mask[idx] if self.attention_mask is not None else torch.ones_like(ids)
        labels = ids.clone()
        labels[am == 0] = -100
        return {"input_ids": ids, "attention_mask": am, "labels": labels}


@dataclass
class Splits:
    forget: TokenDataset
    retain: TokenDataset
    holdout: TokenDataset
    forget_is_outlier: list[bool] = field(default_factory=list)


def collate(batch):  # exposed for DataLoader(collate_fn=...)
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _partition(cfg: ExperimentConfig, examples: list[tuple], marker: int | None) -> Splits:
    """examples: list of (input_ids, attention_mask|None). Splits into f/r/holdout."""
    rng = random.Random(cfg.seed + 1)
    rng.shuffle(examples)
    n = len(examples)
    n_hold = max(1, int(cfg.data.holdout_fraction * n))
    holdout, members = examples[:n_hold], examples[n_hold:]
    n_forget = max(1, int(cfg.data.forget_fraction * len(members)))
    forget, retain = members[:n_forget], members[n_forget:]

    def to_ds(pairs: list[tuple]) -> TokenDataset:
        ids = [p[0] for p in pairs]
        am = [p[1] for p in pairs] if pairs and pairs[0][1] is not None else None
        return TokenDataset(ids, am)

    if marker is not None:
        outlier = [bool((p[0] == marker).any().item()) for p in forget]
    else:
        outlier = [False] * len(forget)
    return Splits(to_ds(forget), to_ds(retain), to_ds(holdout), outlier)


def _synthetic(cfg: ExperimentConfig) -> Splits:
    rng = random.Random(cfg.seed)
    n, length, vocab = cfg.data.n_synthetic, cfg.data.seq_len, cfg.data.vocab_size
    seqs = [torch.tensor([rng.randrange(vocab) for _ in range(length)]) for _ in range(n)]
    # Outliers carry a rare marker token (vocab-1), raising their memorisation.
    n_out = int(cfg.data.outlier_fraction * n)
    for i in range(n_out):
        seqs[i][: length // 4] = vocab - 1
    examples = [(s, None) for s in seqs]
    return _partition(cfg, examples, marker=vocab - 1)


def _tofu(cfg: ExperimentConfig, tokenizer) -> Splits:
    from datasets import load_dataset

    ds = load_dataset("locuslab/TOFU", "full")["train"]
    rows = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]
    if cfg.data.max_samples:
        rows = rows[: cfg.data.max_samples]
    enc = tokenizer(
        rows, truncation=True, max_length=cfg.data.max_length,
        padding="max_length", return_tensors="pt",
    )
    examples = [(enc["input_ids"][i], enc["attention_mask"][i]) for i in range(len(rows))]
    return _partition(cfg, examples, marker=None)


def load_splits(cfg: ExperimentConfig, tokenizer=None) -> Splits:
    """Return forget/retain/holdout splits for the configured dataset."""
    if cfg.data.dataset == "synthetic":
        return _synthetic(cfg)
    if cfg.data.dataset == "tofu":
        if tokenizer is None:
            raise ValueError("TOFU requires a tokenizer.")
        return _tofu(cfg, tokenizer)
    raise ValueError(f"Unknown dataset '{cfg.data.dataset}'.")
