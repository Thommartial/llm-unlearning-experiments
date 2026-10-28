"""Tests for the revision analysis on synthetic results with known answers."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from unlearning import analysis, campaign
from unlearning.evaluate import tpr_at_fpr

SEEDS = [0, 1, 2, 3, 4]
MODELS = ["distilgpt2", "gpt2", "gpt2-medium"]


def _row(**kw):
    base = {"campaign": "central", "kind": "central", "variant": "central", "attack": "loss_threshold",
            "nonmember": "primary", "auc_base": 0.55, "auc_pre": 0.8, "adv_pre": 0.3, "tpr1_post": 0.05,
            "auc_score_change": 0.9, "auc_grad_align": 0.95, "auc_grad_align_random": 0.5,
            "forget_quality_post": 0.5, "model_utility_post": 0.4, "model_utility_pre": 0.45,
            "time_s_method": 10.0, "gpu_name": "Tesla T4"}
    base.update(kw)
    return base


def _central_rows():
    rows = []
    for m_i, model in enumerate(MODELS):
        for seed in SEEDS:
            for attack in ("loss_threshold", "ref_calibrated"):
                for method in ("retrain", "npo"):
                    gap = 0.0 if method == "retrain" else 0.02 + 0.03 * m_i + 0.001 * seed
                    rows.append(_row(model=model, seed=seed, method=method, attack=attack,
                                     auc_post=0.55 + gap, gap_post=gap, adv_post=0.05 + gap,
                                     gap_post_outlier_basediff=gap + 0.04 + 0.002 * seed,
                                     gap_post_typical_basediff=gap - 0.01,
                                     adv_post_outlier_basediff=0.2, adv_post_typical_basediff=0.05,
                                     auc_score_change=0.95 if method == "retrain" else 0.9 - 0.01 * seed,
                                     auc_grad_align=0.97 if method == "retrain" else 0.92,
                                     auc_grad_align_random=0.5 + 0.01 * (seed - 2) + 0.005 * m_i,
                                     model_utility_post=0.0 if (method == "npo" and seed == 0 and m_i == 0) else 0.4,
                                     time_s_method=100.0 + seed))
    return rows


def _fed_rows():
    rows = []
    for k in (3, 10):
        for seed in SEEDS[:3]:
            for method, t, comm in (("retrain", 200.0, 1000), ("client_ga_upload", 20.0, 10), ("client_ga", 50.0 + k,
                                                                                               300)):
                rows.append(_row(campaign="fed_ext", kind="federated", variant=f"iid_K{k}", model="gpt2", seed=seed,
                                 method=method, view={"retrain": "any_client", "client_ga": "any_client"}.get(
                                     method, "server, and all clients at the first recovery broadcast"),
                                 num_clients=k, sharding="iid", auc_post=0.6, gap_post=0.0 if method == "retrain"
                                 else 0.01, adv_post=0.1, time_s_method=t, comm_bytes_method=comm * k,
                                 peak_mem_mib_method=1000.0, peak_mem_reserved_mib_method=1200.0))
    return rows


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    central = pd.DataFrame(_central_rows())
    fed = pd.DataFrame(_fed_rows())
    central.to_csv(root / "central.csv", index=False)
    fed.to_csv(root / "fed_ext.csv", index=False)
    central.to_csv(root / "central_gpu0.csv", index=False)  # per-GPU files are skipped
    pilot = central[(central["model"] == "gpt2") & (central["seed"] == 0)].assign(campaign="pilot",
                                                                                  time_s_method=50.0)
    pilot.to_csv(root / "pilot.csv", index=False)
    rng = np.random.default_rng(0)
    for model in MODELS:
        for seed in SEEDS:
            z = {"row_ids__forget": np.arange(40), "row_ids__primary": np.arange(40)}
            for stage in ("base", "pre", "post_retrain", "post_npo"):
                z[f"{stage}__loss_threshold__forget"] = rng.normal(1.0, 1, 40)
                z[f"{stage}__loss_threshold__primary"] = rng.normal(0.0, 1, 40)
            z["strata__primary__basediff__forget"] = np.arange(40) < 12
            z["strata__primary__basediff__holdout"] = np.arange(40) >= 28
            d = root / "scores" / "central"
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{model}_seed{seed}_central.npz", **z)
            c = root / "configs" / "central"
            c.mkdir(parents=True, exist_ok=True)
            cfg = {"model": {"name": model, "precision": "fp16_amp", "lora": None}, "seed": seed,
                   "data": {"max_length": 128, "max_samples": None, "outlier_fraction": 0.3, "max_new_tokens": 128},
                   "unlearn": {"finetune_epochs": 3, "finetune_lr": 5e-5, "train_batch_size": 16, "epochs": 2,
                               "learning_rate": 1e-5, "batch_size": 4, "eval_batch_size": 64, "retain_weight": 1.0,
                               "beta": 0.1}, "attack": {"min_k": 0.2}}
            (c / f"{model}_seed{seed}_central.json").write_text(json.dumps(cfg))
    return root


def test_t_ci():
    v = [0.1, 0.2, 0.4, 0.3, 0.25]
    s = analysis.t_ci(v)
    assert s["n"] == 5 and s["mean"] == pytest.approx(np.mean(v))
    assert s["half"] == pytest.approx(stats.t.ppf(0.975, 4) * np.std(v, ddof=1) / math.sqrt(5))
    one = analysis.t_ci([0.3, np.nan])
    assert one["n"] == 1 and one["mean"] == 0.3 and np.isnan(one["half"])
    assert analysis.t_ci([])["n"] == 0


def test_load_drops_and_derived(results):
    a = analysis.load(results)
    df = a["rows"]
    assert set(df["campaign"]) == {"central", "fed_ext"} and len(df) == len(_central_rows()) + len(_fed_rows())
    ref = df[df["attack"] == "ref_calibrated"]
    assert ref["auc_base"].isna().all() and ref["auc_score_change"].isna().all() and ref["adv_base"].isna().all()
    loss = df[df["attack"] == "loss_threshold"]
    assert np.allclose(loss["adv_base"], 0.05) and loss["auc_score_change"].notna().all()
    assert a["pilot"] is not None and len(a["configs"]) == 15


def test_h1_h2_h3(results):
    df = analysis.load(results)["rows"]
    h1 = analysis.h1(df)
    r = h1[(h1["model"] == "gpt2") & (h1["kind"] == "central") & (h1["attack"] == "loss_threshold")].iloc[0]
    want = [0.05 + 0.001 * s for s in SEEDS]
    assert r["method"] == "npo" and r["mean"] == pytest.approx(np.mean(want)) and r["verdict"] == "above retrain"
    assert r["half"] == pytest.approx(analysis.t_ci(want)["half"])
    h2 = analysis.h2(df)
    r = h2[(h2["attack"] == "loss_threshold")].iloc[0]
    x = np.log10([82e6, 124e6, 355e6])  # one fit per seed; here every seed has the same slope
    assert r["slope"] == pytest.approx(stats.linregress(x, [0.02, 0.05, 0.08]).slope) and r["n_seeds"] == 5
    assert r["slope_half"] == pytest.approx(0, abs=1e-12) and r["slope_lo"] > 0
    assert r["gpt2-medium__gap_post"] == pytest.approx(0.08 + 0.002)
    assert r["gpt2__rel_drop"] == pytest.approx(np.mean([(0.3 - (0.1 + 0.001 * s)) / 0.3 for s in SEEDS]))
    h3 = analysis.h3(df)
    r = h3[(h3["model"] == "gpt2") & (h3["definition"] == "basediff") & (h3["attack"] == "loss_threshold")].iloc[0]
    assert r["mean"] == pytest.approx(np.mean([0.05 + 0.002 * s for s in SEEDS])) and r["role"] == "primary"
    assert r["adv_outlier"] == 0.2 and set(h3["definition"]) == {"basediff"}  # only the columns present
    assert r["verdict"] == "outliers higher" and "excludes_zero" not in h1 and "excludes_zero" not in h3


def test_verdicts_read_the_ci_in_both_directions():
    v = analysis._verdict
    assert v(analysis.t_ci([0.1, 0.12, 0.11]), "above retrain", "below retrain") == "above retrain"
    assert v(analysis.t_ci([-0.3, -0.35, -0.4]), "above retrain", "below retrain") == "below retrain"
    assert v(analysis.t_ci([-0.1, 0.1, 0.05]), "above retrain", "below retrain") == "inconclusive"
    assert v(analysis.t_ci([0.2]), "above retrain", "below retrain") is None  # one seed: no CI
    rows = [_row(model="gpt2", seed=s, method=m, gap_post=g, gap_post_outlier_basediff=g - 0.1 - 0.01 * s,
                 gap_post_typical_basediff=g, adv_post_outlier_basediff=0.1, adv_post_typical_basediff=0.1)
            for s in SEEDS for m, g in (("retrain", 0.0), ("npo", -0.3 - 0.01 * s))]
    df = analysis._prepare(pd.DataFrame(rows))
    assert analysis.h1(df)["verdict"].tolist() == ["below retrain"]  # contradicts H1
    assert analysis.h3(df)["verdict"].tolist() == ["typical higher"]


def test_h1_two_sided_distance_per_seed():
    """|AUC - AUC_retrain| per seed: gaps of either sign count as distance; H1 keeps the signed gap."""
    gaps = [-0.30, -0.10, 0.05, -0.20, 0.15]
    rows = []
    for seed, gap in enumerate(gaps):
        for method, g, fq in (("retrain", 0.0, 1.0), ("npo", gap, 0.01 * (seed + 1))):
            rows.append(_row(model="gpt2", seed=seed, method=method, gap_post=g, forget_quality_post=fq,
                             auc_post=0.5 + g, gap_post_typical_basediff=g / 2, auc_post_typical_basediff=0.4 + g / 2,
                             gap_post_outlier_basediff=-abs(g), auc_post_outlier_basediff=0.6 - abs(g)))
        rows.append(_row(campaign="fed_iid", kind="federated", variant="iid_K5", model="gpt2", seed=seed,
                         method="client_ga_upload", gap_post=-gap, auc_post=0.8 - gap, forget_quality_post=None))
    df = analysis._prepare(pd.DataFrame(rows))
    h = analysis.h1_two_sided(df)
    assert "retrain" not in set(h["method"])
    r = h[(h["kind"] == "central") & (h["stratum"] == "all")].iloc[0]
    want = analysis.t_ci(np.abs(gaps))
    assert r["mean"] == pytest.approx(0.16) and r["half"] == pytest.approx(want["half"]) and r["n"] == 5
    assert r["gap_mean"] == pytest.approx(np.mean(gaps)) and r["n_below_retrain"] == 3
    assert r["forget_quality_mean"] == pytest.approx(0.03) and r["role"] == "overall"
    leak = analysis.t_ci(np.array(gaps) / 0.5)  # MUSE's PrivLeak: gap / AUC_retrain, per seed
    assert r["privleak_mean"] == pytest.approx(leak["mean"]) and r["privleak_half"] == pytest.approx(leak["half"])
    assert (r["privleak_lo"], r["privleak_hi"]) == (pytest.approx(leak["lo"]), pytest.approx(leak["hi"]))
    typ = h[(h["kind"] == "central") & (h["stratum"] == "typical_basediff")].iloc[0]
    out = h[(h["kind"] == "central") & (h["stratum"] == "outlier_basediff")].iloc[0]
    assert typ["mean"] == pytest.approx(0.08) and typ["role"] == "primary"
    assert typ["privleak_mean"] == pytest.approx(np.mean(np.array(gaps) / 2 / 0.4))  # the stratum's own retrain AUC
    assert out["mean"] == pytest.approx(0.16) and out["n_below_retrain"] == 5
    assert set(h["stratum"]) == {"all", "typical_basediff", "outlier_basediff"}  # only the columns present
    fed = h[(h["kind"] == "federated") & (h["stratum"] == "all")].iloc[0]
    assert fed["mean"] == pytest.approx(0.16) and fed["n_below_retrain"] == 2 and np.isnan(fed["forget_quality_mean"])
    assert fed["privleak_mean"] == pytest.approx(np.mean(-np.array(gaps) / 0.8))  # against the federated retrain
    h1 = analysis.h1(df)  # the pre-registered one-sided H1 is unchanged: signed mean, CI includes 0
    r = h1[h1["kind"] == "central"].iloc[0]
    assert r["mean"] == pytest.approx(np.mean(gaps)) and r["verdict"] == "inconclusive"


def test_h2_slope_is_fitted_within_each_seed():
    x = np.log10([82e6, 124e6, 355e6])
    rows, want = [], []
    for seed, b in enumerate([0.01, 0.03, 0.02, 0.05]):
        models = MODELS if seed != 3 else MODELS[1:]  # a seed with two models still gives a slope
        for model in models:
            gap = 0.01 * seed + b * x[MODELS.index(model)]
            rows.append(_row(model=model, seed=seed, method="npo", gap_post=gap, adv_post=0.1 + gap))
        want.append(b)  # the exact per-seed slope
    rows.append(_row(model="gpt2", seed=4, method="npo", gap_post=0.05, adv_post=0.15))  # one model: no slope
    r = analysis.h2(analysis._prepare(pd.DataFrame(rows))).iloc[0]
    s = analysis.t_ci(want)
    assert r["n_seeds"] == 4 and r["slope"] == pytest.approx(np.mean(want))
    assert r["slope_half"] == pytest.approx(stats.t.ppf(0.975, 3) * np.std(want, ddof=1) / 2)  # df = seeds - 1
    assert (r["slope_lo"], r["slope_hi"]) == pytest.approx((s["lo"], s["hi"]))
    assert r["gpt2__gap_post"] == pytest.approx(np.mean([0.01 * k + b * x[1] for k, b in
                                                         enumerate([0.01, 0.03, 0.02, 0.05])] + [0.05]))


def test_pooled_low_fpr(results):
    a = analysis.load(results)
    lf = analysis.pooled_low_fpr(a["rows"], a["scores"])
    r = lf[(lf["model"] == "gpt2") & (lf["stage"] == "post_npo") & (lf["attack"] == "loss_threshold")].iloc[0]
    zs = [np.load(results / "scores" / "central" / f"gpt2_seed{s}_central.npz") for s in SEEDS]
    f = np.concatenate([z["post_npo__loss_threshold__forget"] for z in zs])
    h = np.concatenate([z["post_npo__loss_threshold__primary"] for z in zs])
    assert (r["n_seeds"], r["n_forget"], r["n_nonmember"]) == (5, 200, 200)
    assert r["tpr_at_0.1"] == tpr_at_fpr(f, h, 0.001) and r["tpr_at_1"] == tpr_at_fpr(f, h, 0.01)
    tf, th = np.tile(np.arange(40) < 12, 5), np.tile(np.arange(40) >= 28, 5)
    assert r["tpr_at_1_outlier_basediff"] == tpr_at_fpr(f[tf], h[th], 0.01)
    assert r["tpr_at_1_typical_basediff"] == tpr_at_fpr(f[~tf], h[~th], 0.01)
    assert lf[lf["attack"] == "min_k_prob"].empty  # not in the files
    # per seed: 40 non-members, so 0.1% FPR allows no false positive (TPR at zero false positives)
    per_seed = [tpr_at_fpr(z["post_npo__loss_threshold__forget"], z["post_npo__loss_threshold__primary"], 0.001)
                for z in zs]
    zero_fp = [np.mean(z["post_npo__loss_threshold__forget"] > z["post_npo__loss_threshold__primary"].max())
               for z in zs]
    assert per_seed == pytest.approx(zero_fp) and r["n_nonmember_per_seed"] == 40
    assert r["tpr_at_0.1_seed_mean"] == pytest.approx(np.mean(per_seed))
    assert r["tpr_at_0.1_seed_half"] == pytest.approx(analysis.t_ci(per_seed)["half"])


def test_update_attacks_and_control(results):
    a = analysis.load(results)
    u = analysis.update_attacks(a["rows"])
    r = u[(u["model"] == "gpt2") & (u["method"] == "npo") & (u["kind"] == "central")].iloc[0]
    assert r["gap_auc_score_change"] == pytest.approx(np.mean([0.9 - 0.01 * s - 0.95 for s in SEEDS]))
    assert r["gap_auc_grad_align"] == pytest.approx(0.92 - 0.97) and bool(r["gap_auc_grad_align_excludes_zero"])
    assert (u["attack"] != "ref_calibrated").all()
    c = analysis.random_control(a["rows"], a["scores"])
    r = c[(c["kind"] == "central") & (c["model"] == "gpt2")].iloc[0]
    draws = [0.5 + 0.01 * (s - 2) + 0.005 for s in SEEDS for _ in ("retrain", "npo")]
    assert r["n_draws"] == 10 and r["mean"] == pytest.approx(np.mean(draws))
    assert r["sd"] == pytest.approx(np.std(draws, ddof=1)) and r["null_sd"] == pytest.approx(math.sqrt(81 / 19200))


def test_costs_slowdown_tofu_hyper(results):
    a = analysis.load(results)
    df = a["rows"]
    c = analysis.costs(df).set_index("num_clients")
    assert c.loc[3, "time_s_method__ratio"] == pytest.approx(53 / 200)
    assert c.loc[10, "comm_bytes_method__ratio"] == pytest.approx(0.3) and c.loc[10, "n"] == 3
    assert c.loc[3, "peak_mem_reserved_mib_method__ratio"] == pytest.approx(1.0)
    sl = analysis.pilot_slowdown(df, a["pilot"])
    assert set(sl["method"]) == {"retrain", "npo"} and np.allclose(sl["slowdown"], 100 / 50) and sl["same_gpu"].all()
    summary, zero = analysis.tofu(df)
    z = zero[(zero["model"] == "distilgpt2") & (zero["column"] == "model_utility_post")].iloc[0]
    assert (z["rows"], z["zero"]) == (10, 1)
    mu = summary[(summary["model"] == "distilgpt2") & (summary["method"] == "npo")].set_index("column")
    assert (mu.loc["model_utility_post", "n"], mu.loc["model_utility_post", "mean"]) == (5, pytest.approx(0.32))
    assert (mu.loc["model_utility_post_nonzero", "n"], mu.loc["model_utility_post_nonzero", "mean"]) == (4, 0.4)
    hyper = analysis.hyperparameters(a["configs"], {"central": "central"})
    assert hyper.loc["seeds", "central"] == "0, 1, 2, 3, 4"
    assert hyper.loc["models", "central"] == "distilgpt2, gpt2, gpt2-medium"
    assert hyper.loc["unlearn.finetune_lr", "central"] == "5e-05" and "federated.rounds" not in hyper.index


def test_the_lora_pilot_rows_are_pilot_rows(tmp_path):
    lora = [_row(campaign="lora", model="EleutherAI/pythia-1b", variant="lora", seed=0, method=m, auc_post=0.6,
                 gap_post=0.0, adv_post=0.1, time_s_method=t) for m, t in (("retrain", 30.0), ("npo", 12.0))]
    pd.DataFrame(lora).to_csv(tmp_path / "lora.csv", index=False)
    pd.DataFrame(lora).assign(campaign="pilot", time_s_method=lambda d: d["time_s_method"] / 2).to_csv(
        tmp_path / "pilot_lora.csv", index=False)
    a = analysis.load(tmp_path)
    assert set(a["rows"]["campaign"]) == {"lora"} and set(a["pilot"]["model"]) == {"EleutherAI/pythia-1b"}
    sl = analysis.pilot_slowdown(a["rows"], a["pilot"])
    assert set(sl["method"]) == {"retrain", "npo"} and np.allclose(sl["slowdown"], 2.0)
    assert analysis._model("EleutherAI/pythia-1b") == "Pythia-1B (LoRA)"


def test_analyse_writes_tables_and_figures(results, tmp_path):
    a = analysis.analyse(results, tmp_path / "fig", tmp_path / "tab")
    names = {p.name for p in a["tables"]}
    assert {"tab_central_loss_threshold_primary.tex", "tab_federated_loss_threshold_primary.tex", "tab_h1.tex",
            "tab_h2.tex", "tab_h3.tex", "tab_low_fpr.tex", "tab_update.tex", "tab_control.tex", "tab_costs.tex",
            "tab_tofu.tex", "tab_hyperparameters.tex", "tab_pilot_slowdown.tex"} <= names
    for p in a["tables"]:
        text = p.read_text()
        assert "\\toprule" in text and "\\bottomrule" in text and text.count("\\begin{table") == 1
    fed = (tmp_path / "tab" / "tab_federated_loss_threshold_primary.tex").read_text()
    assert "\\textbf{All}" in fed and "\\textbf{Typical}" in fed and "\\textbf{Outlier}" in fed
    central = (tmp_path / "tab" / "tab_central_loss_threshold_primary.tex").read_text()
    assert "\\pm" in central and "GPT-2-medium" in central and "\\textbf{All}" in central
    tofu = (tmp_path / "tab" / "tab_tofu.tex").read_text()
    assert "Rows with model utility 0: 1 of" in tofu and "MU excl. 0" in tofu
    low = (tmp_path / "tab" / "tab_low_fpr.tex").read_text()
    assert "TPR at zero false positives (40 non-members)" in low and "\\shortstack{TPR@0.1\\%\\\\pooled}" in low
    figs = {p.name for p in a["figures"]}
    assert {"fig_rev_central.pdf", "fig_rev_strata.pdf", "fig_rev_update.pdf", "fig_rev_fed_update.pdf",
            "fig_rev_control.pdf", "fig_rev_costs.pdf"} <= figs
    assert all(p.exists() and p.with_suffix(".png").exists() for p in a["figures"])
    assert (tmp_path / "tab" / "h1.csv").exists()
    two = pd.read_csv(tmp_path / "tab" / "h1_two_sided.csv")
    assert set(two["kind"]) == {"central", "federated"} and (two["mean"] >= 0).all()
    tex = (tmp_path / "tab" / "tab_h1_two_sided.tex").read_text()
    assert "Exploratory (post hoc)" in tex and "\\label{tab:h1-two-sided}" in tex and "Forget quality" in tex
    assert "MUSE's PrivLeak~\\cite{shi2024muse}" in tex and "PrivLeak (\\%)" in tex
    assert two["privleak_mean"].notna().any()
    assert {"tab_h1_two_sided.tex", "tab_h1_paper.tex", "tab_h1_two_sided_paper.tex"} <= names
    full, paper = ((tmp_path / "tab" / f"tab_h1{s}.tex").read_text() for s in ("", "_paper"))
    assert "\\textbf{Verdict}" in full and "above retrain" in full and "CI excludes" not in full
    assert "Ref.-calib." in full and "Ref.-calib." not in paper and "Non-members" not in paper
    assert "\\label{tab:h1-paper}" in paper and "Table~\\ref{tab:h1} (appendix)" in paper
    # rows: 3 central models (2 attacks in the full table) + 2 federated runs x 2 stages; plus the note
    assert paper.count("above retrain") == 3 + 4 + 1 and full.count("above retrain") == 6 + 4 + 1
    two = (tmp_path / "tab" / "tab_h1_two_sided_paper.tex").read_text()
    assert "Exploratory (post hoc)" in two and "\\ref{tab:h1-paper}" in two and "Ref.-calib." not in two
    assert "PrivLeak (\\%)" in two and "\\cite{shi2024muse}" in two
    update = (tmp_path / "tab" / "tab_update.tex").read_text()
    assert "nan" not in update and "release" in update  # central rows have no view column
    h3 = (tmp_path / "tab" / "tab_h3.tex").read_text()
    assert "outliers higher" in h3 and "CI excl." not in h3


def test_collapsed_models_are_marked_in_every_table(tmp_path):
    """Retain loss above 3 x its pre-unlearning value in every seed marks the method (from the data);
    in some seeds only, or model utility 0 alone (retrain here), does not."""
    rows = []
    for seed in SEEDS:
        for method, ratio, mu, gap in (("retrain", 1.0, 0.0, 0.0), ("gradient_ascent", 40.0 + seed, 0.0, -0.1),
                                       ("gradient_difference", 5.0 if seed < 3 else 2.0, 0.1, -0.3)):
            rows.append(_row(campaign="lora", variant="lora", model="EleutherAI/pythia-1b", seed=seed, method=method,
                             auc_post=0.45 + gap, gap_post=gap, adv_post=0.05, model_utility_post=mu,
                             retain_loss_pre=2.0, retain_loss_post=2.0 * ratio,
                             rouge_real_authors_post=0.0 if mu == 0 else 0.3, rouge_retain_post=0.4,
                             gap_post_outlier_basediff=gap, gap_post_typical_basediff=gap,
                             adv_post_outlier_basediff=0.1, adv_post_typical_basediff=0.1))
    pd.DataFrame(rows).to_csv(tmp_path / "lora.csv", index=False)
    a = analysis.analyse(tmp_path, tmp_path / "fig", tmp_path / "tab")
    c = pd.read_csv(tmp_path / "tab" / "collapsed.csv")
    assert c[["method", "over", "rows"]].values.tolist() == [["gradient_ascent", 5, 5]]
    comp = pd.read_csv(tmp_path / "tab" / "mu_components.csv")
    assert comp[["method", "metric", "split", "zero", "rows"]].values.tolist() == [
        ["retrain", "rouge", "real_authors", 5, 5], ["gradient_ascent", "rouge", "real_authors", 5, 5]]
    note = "Model collapsed: retain loss rose more than threefold in 5/5 seeds. GA: Pythia-1B (LoRA)."
    with_ga = []
    for p in a["tables"]:
        text = p.read_text()
        if "& GA" in text:  # every table with GA rows
            with_ga.append(p.name)
            assert "GA$^{\\dagger}$" in text and note in text, p.name
        assert "GD$^{\\dagger}$" not in text and "Retrain$^{\\dagger}$" not in text, p.name
        assert "& lora &" not in text, p.name  # Run reads central for the LoRA campaign
    assert {"tab_central_loss_threshold_primary.tex", "tab_h1.tex", "tab_h1_paper.tex", "tab_h1_two_sided.tex",
            "tab_h3.tex", "tab_update.tex", "tab_tofu.tex"} <= set(with_ga)
    tofu = (tmp_path / "tab" / "tab_tofu.tex").read_text()
    assert "Pythia-1B (LoRA) & central & GA$^{\\dagger}$" in tofu
    assert ("Retrain rows with model utility 0 are not collapsed: one of the nine harmonic-mean components is 0, "
            "ROUGE on real authors (Pythia-1B (LoRA), central: 5/5 seeds).") in tofu
    # grouped by method: "all N" when every run of the method in the table is marked
    central = [("distilgpt2", "central"), ("gpt2", "central"), ("gpt2-medium", "central"),
               ("EleutherAI/pythia-1b", "lora")]
    fed = [("gpt2", "iid_K3"), ("gpt2", "iid_K5")]
    marked = {**{(m, v, "gradient_ascent"): (5, 5) for m, v in central},
              ("EleutherAI/pythia-1b", "lora", "gradient_difference"): (5, 5),
              **{(m, v, "client_ga_upload"): (3, 3) for m, v in fed}}
    seen = {"gradient_ascent": set(central), "gradient_difference": set(central), "client_ga_upload": set(fed)}
    kinds = {**{r: "central" for r in central}, **{r: "federated" for r in fed}}
    assert analysis._collapse_note(marked, seen, kinds, marked) == (
        "$^{\\dagger}$Model collapsed: retain loss rose more than threefold in every seed. GA: all four central "
        "models; GD: Pythia-1B (LoRA); client-GA upload: both federated runs (intermediate model before recovery; "
        "the released global model is not collapsed).")
    seen["client_ga_upload"].add(("gpt2", "iid_K10"))  # a run shown but not marked: named, not "all"
    flags = {**marked, ("gpt2", "iid_K3", "client_ga"): (3, 3)}  # a collapsed global model drops the clause
    assert analysis._collapse_note({k: v for k, v in marked.items() if k[2] == "client_ga_upload"}, seen, kinds,
                                   flags) == ("$^{\\dagger}$Model collapsed: retain loss rose more than threefold in "
                                              "3/3 seeds. client-GA upload: GPT-2 iid\\_K3, GPT-2 iid\\_K5 "
                                              "(intermediate model before recovery).")


def test_end_to_end_on_a_smoke_campaign(tmp_path):
    import shutil

    import yaml

    from unlearning.pilot import run_pilot

    cfg = yaml.safe_load(open("configs/revision/smoke_cpu.yaml"))
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "results"
    campaign.run_campaign(path, "smoke", out / "smoke.csv", out_dir=out)
    run_pilot(path, "cpu", out)
    shutil.copy(out / "smoke.csv", out / "smoke_gpu0.csv")  # must be ignored
    a = analysis.analyse(out, tmp_path / "fig", tmp_path / "tab")
    assert set(a["rows"]["variant"]) == {"central", "iid_K3", "by_author_K3", "lora"}
    assert len(a["rows"]) == len(campaign.read_rows(out / "smoke.csv"))
    assert {"basediff", "rarity", "legacy", "legacy_pooled"} == set(a["h3"]["definition"])
    lf = a["low_fpr"]
    assert lf["tpr_at_1_outlier_basediff"].notna().any()  # strata tags are in the .npz files
    assert not a["slowdown"].empty and a["tables"] and a["figures"]
