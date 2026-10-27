"""Tests for the target-independent outlier definitions and stratum-matched AUC."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from unlearning import attacks, central, data, outliers, unlearn
from unlearning.config import load_config
from unlearning.evaluate import _auc, mia_metrics
from unlearning.seed import set_seed


def test_tag_pooled_uses_one_threshold_for_both_sets():
    forget = np.array([0.1, 5.0, 0.3, 4.0, 0.2])
    holdout = np.array([3.0, 0.4, 6.0, 0.5, 0.6])
    f_tag, h_tag = outliers.tag_pooled(forget, holdout, 0.3)  # top 3 of 10
    assert f_tag == [False, True, False, True, False]
    assert h_tag == [False, False, True, False, False]
    tagged = np.concatenate([forget[f_tag], holdout[h_tag]])
    rest = np.concatenate([forget[~np.array(f_tag)], holdout[~np.array(h_tag)]])
    assert tagged.min() > rest.max()


def test_tag_pooled_can_tag_one_set_only():
    f_tag, h_tag = outliers.tag_pooled(np.arange(4.0), np.arange(10.0, 14.0), 0.5)
    assert f_tag == [False] * 4 and h_tag == [True] * 4
    assert outliers.tag_pooled(np.ones(3), np.ones(3), 0.0) == ([False] * 3, [False] * 3)


def test_stratum_matching_selects_the_right_rows():
    forget = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    holdout = np.array([0.95, 0.5, 0.4, 0.05])
    f_out = [True, True, False, False, False]
    h_out = [True, False, False, True]
    scores = {"forget": torch.tensor(forget), "holdout": torch.tensor(holdout),
              "forget_is_outlier": f_out,
              "strata": {"basediff": {"forget": f_out, "holdout": h_out},
                         "legacy": {"forget": f_out, "holdout": None}}}
    m = mia_metrics(scores)
    fo, ho = np.array(f_out), np.array(h_out)
    assert m["mia_auc_outlier_basediff"] == _auc(forget[fo], holdout[ho])
    assert m["mia_auc_typical_basediff"] == _auc(forget[~fo], holdout[~ho])
    assert m["mia_advantage_outlier_basediff"] == abs(_auc(forget[fo], holdout[ho]) - 0.5)
    assert (m["n_forget_outlier_basediff"], m["n_holdout_outlier_basediff"]) == (2, 2)
    assert (m["n_forget_typical_basediff"], m["n_holdout_typical_basediff"]) == (3, 2)
    # legacy compares against all holdout records, as in the submitted version
    assert m["mia_auc_outlier_legacy"] == m["mia_auc_outlier"] == _auc(forget[fo], holdout)
    assert m["mia_auc_typical_legacy"] == m["mia_auc_typical"]
    assert m["n_holdout_outlier_legacy"] == 4


def test_metrics_without_strata_are_unchanged():
    scores = {"forget": torch.tensor([0.9, 0.2]), "holdout": torch.tensor([0.1, 0.5]),
              "forget_is_outlier": [True, False]}
    assert mia_metrics(scores) == mia_metrics({**scores, "strata": {}})
    assert not any(k.endswith(outliers.DEFINITIONS) for k in mia_metrics(scores))


def _qa(ids_labels):
    return data.QADataset([torch.tensor(i) for i, _ in ids_labels], [torch.tensor(lab) for _, lab in ids_labels],
                          list(range(len(ids_labels))), [0] * len(ids_labels), [""] * len(ids_labels),
                          [""] * len(ids_labels), pad_token_id=0)


def test_token_rarity_counts_corpus_and_scores_answer_tokens():
    corpus = [[1, 1, 1, 2], [1, 3]]  # counts: 1 -> 4, 2 -> 1, 3 -> 1; N = 6, V = 4
    ds = _qa([([3, 2], [-100, 2]),        # prompt token 3 is ignored
              ([1, 1, 2], [-100, 1, 2])])
    (vals,) = outliers.token_rarity(corpus, [ds])
    p = lambda c: (c + 1) / (6 + 4)  # noqa: E731
    assert vals[0] == pytest.approx(-math.log(p(1)))
    assert vals[1] == pytest.approx((-math.log(p(4)) - math.log(p(1))) / 2)


def _cfg():
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.data.dataset = "synthetic_official"
    cfg.data.n_synthetic = 240
    cfg.data.forget_fraction = 0.2
    cfg.data.holdout_fraction = 0.2
    cfg.unlearn.epochs = 1
    return cfg


def test_base_difficulty_is_mean_answer_nll():
    cfg = _cfg()
    set_seed(0)
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    model.eval()
    splits = data.load_splits(cfg)
    (vals,) = outliers.base_difficulty(model, [splits.holdout], cfg, "cpu")
    with torch.no_grad():
        tok, m = attacks._token_logprobs(model, data.collate([splits.holdout[3]]), "cpu")
    assert vals[3] == pytest.approx(float(-(tok * m).sum() / m.sum()), abs=1e-5)


def test_central_tags_with_the_base_model_before_fine_tuning():
    cfg = _cfg()
    res = central.run_central(cfg, "cpu", methods=["gradient_ascent"])
    set_seed(cfg.seed)
    base, _ = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg)
    f_diff, h_diff = outliers.base_difficulty(base, [splits.forget, splits.holdout], cfg, "cpu")
    strata = res["strata"]
    assert set(strata) == set(outliers.DEFINITIONS)
    assert np.array_equal(strata["basediff"]["forget_value"], f_diff)
    assert np.array_equal(strata["basediff"]["holdout_value"], h_diff)
    n = len(splits.forget) + len(splits.holdout)
    for d in ("basediff", "rarity"):
        assert sum(strata[d]["forget"]) + sum(strata[d]["holdout"]) == int(cfg.data.outlier_fraction * n)
    assert strata["legacy"]["forget"] == res["forget_is_outlier"] and strata["legacy"]["holdout"] is None
    rows = central.central_rows(cfg, res)
    for d in outliers.DEFINITIONS:
        assert rows[0][f"n_forget_outlier_{d}"] + rows[0][f"n_forget_typical_{d}"] == len(splits.forget)
        assert f"adv_post_outlier_{d}" in rows[0] and f"adv_post_typical_{d}" in rows[0]
