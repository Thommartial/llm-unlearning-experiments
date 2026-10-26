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

The `tofu_official` source instead uses TOFU's own author-level splits
(full / forget10 / retain90 / holdout10). Records are stored unpadded with
answer-only labels (question tokens -100, one EOS appended) and padded per
batch by `collate`. `synthetic_official` mirrors it offline for tests.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset, Subset

from .config import ExperimentConfig

TOFU_AUTHOR_SIZE = 20  # TOFU lists 20 consecutive QAs per author


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


class QADataset(Dataset):
    """Unpadded QA records with answer-only labels and per-record metadata.

    `row_ids` index the source TOFU config; `author_ids` = row // 20.
    """

    def __init__(self, input_ids: list[torch.Tensor], labels: list[torch.Tensor],
                 row_ids: list[int], author_ids: list[int],
                 questions: list[str], answers: list[str], pad_token_id: int):
        self.input_ids = input_ids
        self.labels = labels
        self.row_ids = row_ids
        self.author_ids = author_ids
        self.questions = questions
        self.answers = answers
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "labels": self.labels[idx]}


@dataclass
class Splits:
    forget: TokenDataset | QADataset
    retain: TokenDataset | QADataset
    holdout: TokenDataset | QADataset
    forget_is_outlier: list[bool] = field(default_factory=list)


def collate(batch, pad_token_id: int = 0):  # exposed for DataLoader(collate_fn=...)
    """Stack fixed-length items; right-pad variable-length ones to the batch maximum."""
    if len({b["input_ids"].shape[0] for b in batch}) == 1:
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    pad = {"input_ids": pad_token_id, "attention_mask": 0, "labels": -100}
    return {k: pad_sequence([b[k] for b in batch], batch_first=True, padding_value=pad[k])
            for k in batch[0]}


def _pad_token_id(dataset) -> int | None:
    if isinstance(dataset, QADataset):
        return dataset.pad_token_id
    if isinstance(dataset, Subset):
        return _pad_token_id(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        return _pad_token_id(dataset.datasets[0])
    return None


def collate_for(dataset):
    """`collate` with the dataset's pad token id (plain `collate` for legacy datasets)."""
    pad = _pad_token_id(dataset)
    return collate if pad is None else functools.partial(collate, pad_token_id=pad)


def encode_qa(tokenizer, question: str, answer: str, max_length: int) -> tuple:
    """Token ids and answer-only labels for `Question: {q}\\nAnswer: {a}` + EOS.

    The prompt `Question: {q}\\nAnswer:` (the generation prompt) is labelled -100.
    As in TOFU, a sequence longer than `max_length` is cut and loses its EOS.
    """
    prompt = tokenizer(f"Question: {question}\nAnswer:")["input_ids"]
    reply = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
    ids = (prompt + reply + [tokenizer.eos_token_id])[:max_length]
    labels = ([-100] * len(prompt) + ids[len(prompt):])[:max_length]
    return torch.tensor(ids), torch.tensor(labels)


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


def _qa_dataset(tokenizer, rows: list[dict], row_ids: list[int], max_length: int) -> QADataset:
    ids, labels = [], []
    for r in rows:
        i, lab = encode_qa(tokenizer, r["question"], r["answer"], max_length)
        ids.append(i)
        labels.append(lab)
    return QADataset(ids, labels, row_ids, [i // TOFU_AUTHOR_SIZE for i in row_ids],
                     [r["question"] for r in rows], [r["answer"] for r in rows],
                     tokenizer.pad_token_id)


def _tofu_official(cfg: ExperimentConfig, tokenizer) -> Splits:
    """TOFU's own splits: members = full, forget = forget10, retain = retain90,
    holdout = holdout10. The split is fixed and does not depend on the seed."""
    from datasets import load_dataset

    def rows(name: str) -> list[dict]:
        return list(load_dataset("locuslab/TOFU", name)["train"])

    full, forget, retain, holdout = (rows(n) for n in ("full", "forget10", "retain90", "holdout10"))
    n_retain = len(full) - len(forget)
    # forget10 is the tail of full and retain90 the head; row ids index full.
    if ([r["question"] for r in forget] != [r["question"] for r in full[n_retain:]]
            or [r["question"] for r in retain] != [r["question"] for r in full[:n_retain]]):
        raise ValueError("TOFU forget10/retain90 no longer match the order of full.")
    n = cfg.data.max_length
    return Splits(
        forget=_qa_dataset(tokenizer, forget, list(range(n_retain, len(full))), n),
        retain=_qa_dataset(tokenizer, retain, list(range(n_retain)), n),
        holdout=_qa_dataset(tokenizer, holdout, list(range(len(holdout))), n),
        forget_is_outlier=[False] * len(forget),
    )


def _synthetic_official(cfg: ExperimentConfig) -> Splits:
    """Offline stand-in for `tofu_official`: authors of 20 records each, a
    prompt of 2-5 tokens, an answer of 2-8 tokens, and EOS = vocab - 1.
    The last `forget_fraction` of member authors are forgotten; holdout
    authors are separate. Generated with a fixed seed, like TOFU's split."""
    rng = random.Random(0)
    eos = cfg.data.vocab_size - 1
    n_authors = max(3, cfg.data.n_synthetic // TOFU_AUTHOR_SIZE)
    n_hold = max(1, int(cfg.data.holdout_fraction * n_authors))
    n_members = n_authors - n_hold
    n_forget = max(1, int(cfg.data.forget_fraction * n_members))

    def author_rows(n: int) -> list[tuple]:
        out = []
        for _ in range(n):
            fact = [rng.randrange(1, eos) for _ in range(3)]  # shared by the author's answers
            for _ in range(TOFU_AUTHOR_SIZE):
                q = [rng.randrange(1, eos) for _ in range(rng.randint(2, 5))]
                a = fact[: rng.randint(1, 3)] + [rng.randrange(1, eos) for _ in range(rng.randint(1, 5))]
                out.append((q, a))
        return out

    def to_ds(pairs: list[tuple], row_ids: list[int]) -> QADataset:
        ids = [torch.tensor(q + a + [eos]) for q, a in pairs]
        labels = [torch.tensor([-100] * len(q) + a + [eos]) for q, a in pairs]
        text = [" ".join(map(str, q)) for q, _ in pairs], [" ".join(map(str, a)) for _, a in pairs]
        return QADataset(ids, labels, row_ids, [i // TOFU_AUTHOR_SIZE for i in row_ids],
                         *text, pad_token_id=eos)

    members, holdout = author_rows(n_members), author_rows(n_hold)
    n_retain = (n_members - n_forget) * TOFU_AUTHOR_SIZE
    return Splits(
        forget=to_ds(members[n_retain:], list(range(n_retain, len(members)))),
        retain=to_ds(members[:n_retain], list(range(n_retain))),
        holdout=to_ds(holdout, list(range(len(holdout)))),
        forget_is_outlier=[False] * (len(members) - n_retain),
    )


def load_splits(cfg: ExperimentConfig, tokenizer=None) -> Splits:
    """Return forget/retain/holdout splits for the configured dataset."""
    if cfg.data.dataset == "synthetic":
        return _synthetic(cfg)
    if cfg.data.dataset == "synthetic_official":
        return _synthetic_official(cfg)
    if cfg.data.dataset in ("tofu", "tofu_official"):
        if tokenizer is None:
            raise ValueError("TOFU requires a tokenizer.")
        if cfg.data.dataset == "tofu_official":
            return _tofu_official(cfg, tokenizer)
        return _tofu(cfg, tokenizer)
    raise ValueError(f"Unknown dataset '{cfg.data.dataset}'.")
