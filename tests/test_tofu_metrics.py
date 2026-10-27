"""Tests for the TOFU metrics, TPR at a fixed FPR and the one-pass attack scores."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from unlearning import attacks, central, data, tofu_metrics, unlearn
from unlearning.config import load_config
from unlearning.evaluate import tpr_at_fpr
from unlearning.seed import set_seed


def _cfg():
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.data.dataset = "synthetic_official"
    cfg.data.forget_fraction = 0.2
    cfg.data.holdout_fraction = 0.2
    cfg.unlearn.epochs = 1
    cfg.data.max_new_tokens = 8
    return cfg


def test_truth_ratio_on_hand_made_numbers():
    para = np.array([1.0, 2.0])
    pert = np.array([[2.0, 4.0], [1.0, 1.0]])  # mean 3 and 1
    ratio = tofu_metrics.truth_ratio(para, pert)
    assert np.allclose(ratio, [math.exp(2), math.exp(-1)])
    # forget: mean of min(R, 1/R); other sets: mean of max(0, 1 - 1/R)
    assert tofu_metrics.truth_ratio_score(ratio, forget=True) == pytest.approx((math.exp(-2) + math.exp(-1)) / 2)
    assert tofu_metrics.truth_ratio_score(ratio, forget=False) == pytest.approx((1 - math.exp(-2) + 0) / 2)


def test_probability_plain_and_normalized():
    nll = np.array([0.0, math.log(2)])  # p = 1, 0.5
    assert tofu_metrics.probability(nll) == pytest.approx(0.75)
    pert = np.array([[math.log(4), math.log(4)], [math.log(2), math.log(2)]])  # 0.25 + 0.25; 0.5 + 0.5
    assert tofu_metrics.probability(nll, pert) == pytest.approx((1 / 1.5 + 0.5 / 1.5) / 2)


def test_model_utility_is_harmonic_mean():
    assert tofu_metrics.model_utility([1.0, 0.5, 0.25]) == pytest.approx(3 / (1 + 2 + 4))
    assert tofu_metrics.model_utility([0.9, 0.0, 0.5]) == 0.0


def test_forget_quality_ks_identical_vs_shifted():
    rng = np.random.default_rng(0)
    a = rng.lognormal(size=400)
    assert tofu_metrics.forget_quality(a, a.copy()) == pytest.approx(1.0)
    assert tofu_metrics.forget_quality(a, a * 3) < 1e-6


def test_rouge_l_recall():
    got = tofu_metrics.rouge_l_recall(["the cats sat on mats", "nothing"], ["The cat sat", "a b"])
    assert got[0] == pytest.approx(1.0) and got[1] == 0.0  # stemming: cats -> cat


def test_tpr_at_fpr():
    member = np.array([0.95, 0.9, 0.5, 0.1])
    nonmember = np.arange(100) / 100  # 0.00 ... 0.99
    # threshold above 0.99: FPR 0, TPR 0; at 0.95 one non-member (0.95..0.99 are 5) -> FPR 0.05
    assert tpr_at_fpr(member, nonmember, 0.01) == 0.0
    assert tpr_at_fpr(member, nonmember, 0.05) == 0.25
    assert tpr_at_fpr(member, nonmember, 0.10) == 0.5
    assert tpr_at_fpr(np.ones(3), np.zeros(3), 0.01) == 1.0
    assert math.isnan(tpr_at_fpr(np.array([]), nonmember, 0.01))


@pytest.mark.parametrize("dataset", ["synthetic_official", "synthetic"])
def test_score_all_matches_score_dataset(dataset):
    cfg = _cfg()
    cfg.data.dataset = dataset
    set_seed(0)
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg)
    unlearn.finetune_base(cfg, splits, "cpu", model=model)
    got = attacks.score_all(model, splits.forget, cfg, "cpu")
    for a in ("loss_threshold", "min_k_prob"):
        assert torch.equal(got[a], attacks.score_dataset(model, splits.forget,
                                                         replace(cfg, attack=replace(cfg.attack, method=a)), "cpu"))


def test_generation_is_left_padding_invariant():
    cfg = _cfg()
    set_seed(0)
    model, _ = unlearn.build_model_and_tokenizer(cfg)
    sets = tofu_metrics.load_eval_sets(cfg, None)
    ds = sets["forget"].answer
    eos = sets["forget"].eos_token_id
    batched = tofu_metrics.generate(model, ds, eos, 8, 64, "cpu")
    single = [g for i in range(8) for g in tofu_metrics.generate(model, _one(ds, i), eos, 8, 1, "cpu")]
    assert batched[:8] == single
    assert len({len(p) for p in ds.questions[:8]}) > 1  # prompts of different lengths were padded


def _one(ds, i):
    return data.QADataset([ds.input_ids[i]], [ds.labels[i]], [0], [0], [ds.questions[i]], [ds.answers[i]],
                          ds.pad_token_id)


def test_synthetic_eval_sets_shape():
    sets = tofu_metrics.load_eval_sets(_cfg(), None)
    assert set(sets) == set(tofu_metrics.TOFU_SETS)
    assert len(sets["forget"].perturbed) == 5 and len(sets["real_authors"].perturbed) == 3
    assert sets["real_authors"].paraphrased is None and sets["forget"].paraphrased is not None
    cfg = _cfg()
    cfg.data.tofu_metrics = False
    assert tofu_metrics.load_eval_sets(cfg, None) is None
    cfg = _cfg()
    cfg.data.dataset = "synthetic"
    assert tofu_metrics.load_eval_sets(cfg, None) is None


@pytest.fixture(scope="module")
def run():
    cfg = _cfg()
    return cfg, central.run_central(cfg, "cpu")


def test_ref_calibrated_is_loss_minus_base(run):
    _, res = run
    for k, base in res["base"]["scores"]["ref_calibrated"].items():
        assert torch.equal(base, torch.zeros_like(base))  # target = base model
        base_loss = res["base"]["scores"]["loss_threshold"][k]
        for stage in [res["pre"], *res["post"].values()]:
            assert torch.equal(stage["scores"]["ref_calibrated"][k], stage["scores"]["loss_threshold"][k] - base_loss)


def test_central_rows_have_tpr_and_tofu_metrics(run):
    cfg, res = run
    rows = central.central_rows(cfg, res)
    assert {r["attack"] for r in rows} == set(central.ATTACKS)
    for r in rows:
        assert all(0 <= r[f"tpr1_{t}"] <= 1 for t in ("base", "pre", "post"))
        assert 0 <= r["model_utility_post"] <= 1 and 0 <= r["forget_quality_post"] <= 1
        for s in tofu_metrics.TOFU_SETS:
            assert all(np.isfinite(r[f"{k}_{s}_{t}"]) for k in ("prob", "rouge", "truth_ratio") for t in ("pre", "post"))
    by_method = {r["method"]: r for r in rows}
    assert by_method["retrain"]["forget_quality_post"] == 1.0  # retrain against itself
    assert len({r["forget_quality_pre"] for r in rows}) == 1  # the fine-tuned model, shared
    # the columns follow from the raw values
    ga = res["post"]["gradient_ascent"]["tofu"]
    assert by_method["gradient_ascent"]["forget_quality_post"] == tofu_metrics.forget_quality(
        tofu_metrics.raw_truth_ratio(ga), tofu_metrics.raw_truth_ratio(res["post"]["retrain"]["tofu"]))
    assert by_method["gradient_ascent"]["model_utility_post"] == tofu_metrics.summarise(ga)["model_utility"]


def test_tofu_nll_matches_attack_scoring(run):
    cfg, res = run
    # the fine-tuned model's TOFU forget answers are the forget records: same NLL as the loss attack
    pre = res["pre"]
    assert np.allclose(pre["tofu"]["forget"]["answer_nll"], -pre["scores"]["loss_threshold"]["forget"].double().numpy(),
                       rtol=0, atol=1e-6)


def test_save_scores_roundtrip(run, tmp_path):
    _, res = run
    path = tmp_path / "scores.npz"
    central.save_scores(res, path)
    z = np.load(path)
    assert np.array_equal(z["post_npo__min_k_prob__holdout10"], res["post"]["npo"]["scores"]["min_k_prob"]["holdout10"].numpy())
    assert np.array_equal(z["row_ids__primary"], res["row_ids"]["primary"])
    assert z["pre__truth_ratio__forget"].shape == (len(res["row_ids"]["forget"]),)
    assert z["post_retrain__generated__world_facts"].dtype.kind == "U"
