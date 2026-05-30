"""Torch-free tests for the sweep aggregation logic."""

from unlearning.sweep import aggregate


def _row(model, seed, attack, adv_post, adv_out, adv_typ):
    return {
        "model": model, "seed": seed, "attack": attack,
        "auc_pre": 0.9, "auc_post": 0.5 + adv_post,
        "adv_pre": 0.4, "adv_post": adv_post,
        "adv_post_outlier": adv_out, "adv_post_typical": adv_typ,
        "forget_loss_post": 1.4, "retain_loss_post": 1.0,
    }


def test_aggregate_mean_and_std():
    rows = [
        _row("gpt2", 0, "min_k_prob", 0.20, 0.30, 0.10),
        _row("gpt2", 1, "min_k_prob", 0.30, 0.40, 0.20),
    ]
    summ = aggregate(rows)
    assert len(summ) == 1
    e = summ[0]
    assert e["model"] == "gpt2" and e["n_seeds"] == 2
    assert abs(e["adv_post_mean"] - 0.25) < 1e-9
    assert e["adv_post_std"] > 0
    assert abs(e["adv_post_outlier_mean"] - 0.35) < 1e-9


def test_aggregate_groups_by_model_and_attack():
    rows = [
        _row("gpt2", 0, "min_k_prob", 0.2, 0.3, 0.1),
        _row("gpt2", 0, "loss_threshold", 0.1, 0.2, 0.05),
        _row("pythia", 0, "min_k_prob", 0.4, 0.5, 0.3),
    ]
    summ = aggregate(rows)
    assert len(summ) == 3
