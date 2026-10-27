"""TOFU metrics: truth ratio, probability, ROUGE-L, forget quality, model utility.

Formulas follow TOFU's `evaluate_util.py` and `aggregate_eval_stat.py`.
L(x) is the mean NLL of the answer tokens of x (answer + EOS, question
masked), the same scoring as the MIA attacks.

* Raw truth ratio R = exp(mean_j L(perturbed_j) - L(paraphrased)). The
  paraphrased answer is TOFU's `paraphrased_answer` on forget and retain, and
  the answer itself on real authors and world facts.
* Truth-ratio score: mean of min(R, 1/R) on forget; mean of max(0, 1 - 1/R)
  on the other sets.
* Probability: mean of exp(-L(answer)) on forget and retain; on real authors
  and world facts, exp(-L(a)) / (exp(-L(a)) + sum_j exp(-L(perturbed_j))).
* ROUGE-L recall (with stemming) of the greedy generation against the answer.
  Prompt `Question: {q}\\nAnswer:`, left padding, stop at EOS or
  `max_new_tokens`.
* Forget quality: KS-test p-value between the raw R on forget of a model and
  of the retrained model of the same seed.
* Model utility: harmonic mean of probability, ROUGE-L and truth-ratio score
  on retain, real authors and world facts (nine values).

All records of each set are used (400 / 400 / 100 / 117 on TOFU).
"""

from __future__ import annotations

import functools
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .data import QADataset, collate

TOFU_SETS = {"forget": "forget10_perturbed", "retain": "retain_perturbed",
             "real_authors": "real_authors_perturbed", "world_facts": "world_facts_perturbed"}
PARAPHRASED = ("forget", "retain")  # the other sets use the answer itself
UTILITY_SETS = ("retain", "real_authors", "world_facts")
TOFU_FORGET_ROWS = range(3600, 4000)  # forget10_perturbed = full[3600:4000]
TOFU_RETAIN_ROWS = range(0, 400)      # retain_perturbed = full[0:400] (authors 0-19)


@dataclass
class EvalSet:
    """One TOFU evaluation set, encoded with answer-only labels."""
    answer: QADataset
    paraphrased: QADataset | None  # None: the answer itself
    perturbed: list[QADataset]     # one dataset per perturbation index
    eos_token_id: int
    decode: Callable[[list[int]], str]


def truth_ratio(para_nll: np.ndarray, pert_nll: np.ndarray) -> np.ndarray:
    """Raw truth ratio per record; `pert_nll` is [records, perturbations]."""
    return np.exp(np.asarray(pert_nll).mean(-1) - np.asarray(para_nll))


def truth_ratio_score(ratio: np.ndarray, forget: bool) -> float:
    ratio = np.asarray(ratio)
    if forget:
        return float(np.mean(np.minimum(ratio, 1 / ratio)))
    return float(np.mean(np.maximum(0, 1 - 1 / ratio)))


def probability(answer_nll: np.ndarray, pert_nll: np.ndarray | None = None) -> float:
    """Mean answer probability; normalized over the perturbed answers when given."""
    p = np.exp(-np.asarray(answer_nll))
    if pert_nll is None:
        return float(np.mean(p))
    return float(np.mean(p / (p + np.exp(-np.asarray(pert_nll)).sum(-1))))


def rouge_l_recall(generated: list[str], answers: list[str]) -> np.ndarray:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return np.array([scorer.score(gt, gen)["rougeL"].recall for gen, gt in zip(generated, answers, strict=True)])


def forget_quality(ratio: np.ndarray, retrain_ratio: np.ndarray) -> float:
    """KS-test p-value between two raw truth-ratio samples on the forget set."""
    from scipy.stats import ks_2samp

    return float(ks_2samp(ratio, retrain_ratio).pvalue)


def model_utility(values: list[float]) -> float:
    """Harmonic mean; 0 if any value is 0."""
    from scipy.stats import hmean

    values = np.asarray(values, dtype=float)
    return 0.0 if (values <= 0).any() else float(hmean(values))


# --- loading ---------------------------------------------------------------

def load_eval_sets(cfg, tokenizer, splits=None) -> dict[str, EvalSet] | None:
    """The four TOFU evaluation sets, or None when the dataset has none.
    `splits` (synthetic only) avoids building the splits again."""
    if not cfg.data.tofu_metrics:
        return None
    if cfg.data.dataset == "tofu_official":
        return _tofu_sets(cfg, tokenizer)
    if cfg.data.dataset == "synthetic_official":
        return _synthetic_sets(cfg, splits)
    return None


def _tofu_sets(cfg, tokenizer) -> dict[str, EvalSet]:
    from datasets import load_dataset

    from .data import _qa_dataset

    full = [r["question"] for r in load_dataset("locuslab/TOFU", "full")["train"]]
    expected = {"forget": [full[i] for i in TOFU_FORGET_ROWS], "retain": [full[i] for i in TOFU_RETAIN_ROWS]}
    decode = functools.partial(tokenizer.decode, skip_special_tokens=True)
    sets = {}
    for name, config in TOFU_SETS.items():
        rows = list(load_dataset("locuslab/TOFU", config)["train"])
        if name in expected and [r["question"] for r in rows] != expected[name]:
            raise ValueError(f"TOFU {config} no longer matches the rows of full.")
        n_pert = {len(r["perturbed_answer"]) for r in rows}
        if len(n_pert) != 1:
            raise ValueError(f"TOFU {config}: varying number of perturbed answers {n_pert}.")

        def encode(answers, rows=rows):
            return _qa_dataset(tokenizer, [{"question": r["question"], "answer": a} for r, a in zip(rows, answers, strict=True)],
                               list(range(len(rows))), cfg.data.max_length)

        sets[name] = EvalSet(
            answer=encode([r["answer"] for r in rows]),
            paraphrased=encode([r["paraphrased_answer"] for r in rows]) if name in PARAPHRASED else None,
            perturbed=[encode([r["perturbed_answer"][j] for r in rows]) for j in range(n_pert.pop())],
            eos_token_id=tokenizer.eos_token_id, decode=decode,
        )
    return sets


def _synthetic_sets(cfg, splits=None) -> dict[str, EvalSet]:
    """Offline stand-in: forget = the forget records, retain = the first 20
    retain records (author 0, never drawn), real authors and world facts =
    10 fresh records each. Paraphrases reverse the answer; perturbations are
    random answers of the same length."""
    from .data import load_splits

    splits = splits or load_splits(cfg)
    rng = random.Random(1)
    eos = cfg.data.vocab_size - 1

    def noise(n):
        return [rng.randrange(1, eos) for _ in range(n)]

    def encode(qs, answers):
        ids = [torch.tensor(q + a + [eos]) for q, a in zip(qs, answers, strict=True)]
        labels = [torch.tensor([-100] * len(q) + a + [eos]) for q, a in zip(qs, answers, strict=True)]
        return QADataset(ids, labels, list(range(len(qs))), [0] * len(qs),
                         [" ".join(map(str, q)) for q in qs], [" ".join(map(str, a)) for a in answers], eos)

    def ints(texts):
        return [[int(t) for t in s.split()] for s in texts]

    def make(qs, answers, n_pert, paraphrase):
        return EvalSet(answer=encode(qs, answers),
                       paraphrased=encode(qs, [a[::-1] for a in answers]) if paraphrase else None,
                       perturbed=[encode(qs, [noise(len(a)) for a in answers]) for _ in range(n_pert)],
                       eos_token_id=eos, decode=lambda ids: " ".join(map(str, ids)))

    sets = {"forget": make(ints(splits.forget.questions), ints(splits.forget.answers), 5, True),
            "retain": make(ints(splits.retain.questions[:20]), ints(splits.retain.answers[:20]), 5, True)}
    for name in ("real_authors", "world_facts"):
        sets[name] = make([noise(rng.randint(2, 5)) for _ in range(10)],
                          [noise(rng.randint(2, 8)) for _ in range(10)], 3, False)
    return sets


# --- evaluation ------------------------------------------------------------

@torch.no_grad()
def _nll(model, dataset, cfg, device: str) -> np.ndarray:
    from .attacks import sequence_avg_logprob

    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.unlearn.eval_batch_size,
                                         collate_fn=functools.partial(collate, pad_token_id=dataset.pad_token_id))
    return -torch.cat([sequence_avg_logprob(model, b, device).cpu() for b in loader]).double().numpy()


def _max_positions(model) -> int | None:
    return getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)


@torch.no_grad()
def generate(model, dataset: QADataset, eos_token_id: int, max_new_tokens: int,
             batch_size: int, device: str) -> list[list[int]]:
    """Greedy continuations of each record's prompt (left padded), cut at the first EOS."""
    model.eval()
    prompts = [ids[: int((lab == -100).sum())] for ids, lab in zip(dataset.input_ids, dataset.labels, strict=True)]
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        width = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), width), dataset.pad_token_id)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, p in enumerate(chunk):
            ids[j, width - len(p):] = p
            mask[j, width - len(p):] = 1
        n_new = max_new_tokens
        if _max_positions(model):
            n_new = min(n_new, _max_positions(model) - width)
        gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=n_new,
                             do_sample=False, pad_token_id=dataset.pad_token_id, eos_token_id=eos_token_id)
        for row in gen[:, width:].tolist():
            out.append(row[: row.index(eos_token_id)] if eos_token_id in row else row)
    return out


def evaluate_model(model, sets: dict[str, EvalSet], cfg, device: str) -> dict:
    """Per-record raw values for each set: answer / paraphrased / perturbed NLL,
    generations and their ROUGE-L recall."""
    model.eval()
    raw = {}
    for name, es in sets.items():
        answer = _nll(model, es.answer, cfg, device)
        gen = [es.decode(g) for g in generate(model, es.answer, es.eos_token_id, cfg.data.max_new_tokens,
                                              cfg.unlearn.eval_batch_size, device)]
        raw[name] = {
            "answer_nll": answer,
            "para_nll": answer if es.paraphrased is None else _nll(model, es.paraphrased, cfg, device),
            "pert_nll": np.stack([_nll(model, p, cfg, device) for p in es.perturbed], axis=1),
            "generated": gen,
            "rouge": rouge_l_recall(gen, es.answer.answers),
        }
    return raw


def summarise(raw: dict) -> dict:
    """Probability, ROUGE-L and truth-ratio score per set, and model utility."""
    out = {}
    for name, r in raw.items():
        ratio = truth_ratio(r["para_nll"], r["pert_nll"])
        out[f"prob_{name}"] = probability(r["answer_nll"], None if name in PARAPHRASED else r["pert_nll"])
        out[f"rouge_{name}"] = float(np.mean(r["rouge"]))
        out[f"truth_ratio_{name}"] = truth_ratio_score(ratio, forget=name == "forget")
    out["model_utility"] = model_utility([out[f"{k}_{s}"] for s in UTILITY_SETS
                                          for k in ("prob", "rouge", "truth_ratio")])
    return out


def raw_truth_ratio(raw: dict, name: str = "forget") -> np.ndarray:
    return truth_ratio(raw[name]["para_nll"], raw[name]["pert_nll"])
