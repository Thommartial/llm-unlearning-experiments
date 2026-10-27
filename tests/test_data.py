"""Unit tests for the answer-only data path and dynamic padding (no downloads)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from unlearning import attacks, data
from unlearning.config import ExperimentConfig, load_config


def _official_cfg() -> ExperimentConfig:
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.data.dataset = "synthetic_official"
    cfg.data.n_synthetic = 240
    cfg.data.holdout_fraction = 0.2
    cfg.data.forget_fraction = 0.2
    return cfg


class _CharTokenizer:
    """Minimal stand-in for a HF tokenizer: one id per character, EOS = 1."""

    eos_token_id = 1
    pad_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


def test_collate_pads_right_and_masks_labels():
    a = {"input_ids": torch.tensor([5, 6, 7]), "attention_mask": torch.ones(3, dtype=torch.long),
         "labels": torch.tensor([-100, 6, 7])}
    b = {"input_ids": torch.tensor([8]), "attention_mask": torch.ones(1, dtype=torch.long),
         "labels": torch.tensor([8])}
    out = data.collate([a, b], pad_token_id=99)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 99, 99]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]


def test_collate_stacks_fixed_length_items_unchanged():
    ds = data.TokenDataset([torch.tensor([1, 2, 0]), torch.tensor([3, 4, 5])],
                           [torch.tensor([1, 1, 0]), torch.tensor([1, 1, 1])])
    out = data.collate([ds[0], ds[1]])
    assert out["input_ids"].tolist() == [[1, 2, 0], [3, 4, 5]]
    assert out["labels"].tolist() == [[1, 2, -100], [3, 4, 5]]


def test_collate_for_finds_pad_id_through_wrappers():
    splits = data.load_splits(_official_cfg())
    pad = splits.forget.pad_token_id
    for ds in (splits.forget, Subset(splits.retain, [0, 1]),
               ConcatDataset([splits.forget, splits.retain])):
        assert data.collate_for(ds).keywords == {"pad_token_id": pad}
    assert data.collate_for(data.TokenDataset([torch.tensor([1])])) is data.collate


def test_encode_qa_masks_prompt_and_appends_eos():
    tok = _CharTokenizer()
    ids, labels = data.encode_qa(tok, "q?", "ab", max_length=64)
    prompt = "Question: q?\nAnswer:"
    assert ids.tolist() == [ord(c) for c in prompt + " ab"] + [1]
    assert labels.tolist() == [-100] * len(prompt) + [ord(c) for c in " ab"] + [1]
    # truncation drops the EOS, as in TOFU
    ids, labels = data.encode_qa(tok, "q?", "ab", max_length=len(prompt) + 2)
    assert ids.tolist()[-1] == ord("a") and len(labels) == len(ids)


def test_synthetic_official_split_sizes_and_disjointness():
    cfg = _official_cfg()
    splits = data.load_splits(cfg)
    f, r, p, h10 = splits.forget, splits.retain, splits.nonmembers["primary"], splits.nonmembers["holdout10"]
    # 240 records -> 12 authors: 2 holdout10, 10 members-to-be; 2 forgotten, 2 drawn as primary non-members
    assert (len(f), len(r), len(p), len(h10)) == (40, 120, 40, 40)
    assert splits.holdout is p
    assert set(f.author_ids) == {8, 9} and set(h10.author_ids) == {10, 11}
    assert set(p.author_ids) == set(data.draw_nonmember_authors(cfg.seed, range(1, 8), 2))
    assert 0 not in p.author_ids  # like TOFU authors 0-19, never drawn
    authors = [set(ds.author_ids) for ds in (f, r, p, h10)]
    assert all(a.isdisjoint(b) for i, a in enumerate(authors) for b in authors[i + 1:])
    assert sorted(f.row_ids + r.row_ids + p.row_ids) == list(range(200))
    assert len(splits.forget_is_outlier) == len(f)


def test_synthetic_official_only_the_draw_depends_on_the_seed():
    a = _official_cfg()
    b = _official_cfg()
    b.seed = a.seed + 1
    sa, sb = data.load_splits(a), data.load_splits(b)
    assert all(torch.equal(x, y) for x, y in zip(sa.forget.input_ids, sb.forget.input_ids, strict=True))
    h_a, h_b = sa.nonmembers["holdout10"], sb.nonmembers["holdout10"]
    assert all(torch.equal(x, y) for x, y in zip(h_a.input_ids, h_b.input_ids, strict=True))
    draws = {tuple(sorted(set(data.load_splits(_seeded(s)).holdout.author_ids))) for s in range(5)}
    assert len(draws) > 1  # re-drawn per seed


def _seeded(seed):
    cfg = _official_cfg()
    cfg.seed = seed
    return cfg


def test_draw_nonmember_authors_rule():
    pool = data.TOFU_NONMEMBER_POOL
    assert (pool.start, pool.stop) == (20, 180)
    for seed in range(5):
        drawn = data.draw_nonmember_authors(seed, pool, 20)
        assert drawn == sorted(drawn) and len(set(drawn)) == 20
        assert all(20 <= a < 180 for a in drawn)
        assert drawn == data.draw_nonmember_authors(seed, pool, 20)  # deterministic
        expected = np.random.default_rng(10_000 + seed).choice(np.arange(20, 180), size=20, replace=False)
        assert drawn == sorted(expected.tolist())
    assert data.draw_nonmember_authors(0, pool, 20) != data.draw_nonmember_authors(1, pool, 20)


def test_synthetic_official_labels_are_answer_only():
    splits = data.load_splits(_official_cfg())
    eos = splits.forget.pad_token_id
    for ids, labels in zip(splits.forget.input_ids, splits.forget.labels, strict=True):
        scored = labels != -100
        assert not scored[0] and scored[-1] and labels[-1] == eos
        assert torch.equal(labels[scored], ids[scored])


def test_scores_are_answer_only_and_padding_invariant():
    from unlearning.unlearn import build_model_and_tokenizer

    cfg = _official_cfg()
    torch.manual_seed(0)
    model, _ = build_model_and_tokenizer(cfg)
    model.eval()
    ds = Subset(data.load_splits(cfg).forget, list(range(8)))
    with torch.no_grad():
        batched = torch.cat([attacks.sequence_avg_logprob(model, b, "cpu")
                             for b in DataLoader(ds, batch_size=4, collate_fn=data.collate_for(ds))])
        expected = []
        for i in range(len(ds)):
            item = ds[i]
            logp = torch.log_softmax(model(input_ids=item["input_ids"][None]).logits[0, :-1], -1)
            tok = logp.gather(-1, item["input_ids"][1:, None])[:, 0]
            expected.append(tok[item["labels"][1:] != -100].mean())
    assert torch.allclose(batched, torch.stack(expected), atol=1e-5)


def test_label_mask_equals_attention_mask_on_legacy_path():
    ds = data.TokenDataset([torch.tensor([4, 5, 0, 0]), torch.tensor([6, 7, 8, 0])],
                           [torch.tensor([1, 1, 0, 0]), torch.tensor([1, 1, 1, 0])])
    batch = data.collate([ds[0], ds[1]])
    assert torch.equal(batch["labels"] != -100, batch["attention_mask"].bool())
