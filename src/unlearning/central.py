"""Centralized runs: fine-tune once per (model, seed), then retrain and unlearn.

The base and fine-tuned weights are kept on CPU and restored before each
method, so retrain, GA, GD and NPO all start from the same weights and the
fine-tune runs once instead of once per method. The base model is scored
first (baseline AUCs) and used to tag outliers; the legacy tags come from the
fine-tuned model (see `outliers`). Pre-unlearning scores and losses come from
the fine-tuned model and are shared by every method. Each method is reseeded
before it runs, so its result does not depend on which other methods ran
before it (a resumed run gives the same numbers).

Every attack metric is computed against each non-member set (`primary` and
`holdout10` on the official splits, `holdout` on the legacy ones). One
forward pass per batch gives the loss and Min-K scores; the
reference-calibrated score is the loss score minus the cached base-model
loss score. On the official splits the fine-tuned model and every method also
get the TOFU metrics (`tofu_metrics`); forget quality is measured against the
retrained model of the same run.

Update-signal attacks (`update_attacks`): every row carries the AUC of the
per-record score change from the fine-tuned model to the method (pre −
post). Every method also gets the white-box gradient alignment with
Δθ = θ_post − θ_pre, θ_pre being the restored fine-tuned snapshot, and the
same with a random direction of the same norm as a negative control. For
retrain, Δθ = θ_retrain − θ_pre is the version change a user would observe.
"""

from __future__ import annotations

from dataclasses import replace

METHODS = ("retrain", "gradient_ascent", "gradient_difference", "npo")
ATTACKS = ("loss_threshold", "min_k_prob", "ref_calibrated")
TPR_FPR = 0.01  # TPR at 1% FPR per run; raw scores are saved for pooling (`save_scores`)
ALIGN_SEED = 20_000  # random control direction: ALIGN_SEED + 10 * seed + method index


def _snapshot(model) -> dict:
    """CPU copy of the weights, restored later with load_state_dict."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _scores(model, splits, cfg, device: str, base_loss: dict | None = None) -> dict:
    """attack -> {"forget": scores, <non-member set>: scores, ...}.

    `base_loss` (set -> base-model loss scores) is the reference of
    `ref_calibrated`; None means `model` is the base model (all scores 0).
    """
    from . import attacks
    from .data import nonmember_sets

    sets = {"forget": splits.forget, **nonmember_sets(splits)}
    raw = {k: attacks.score_all(model, ds, cfg, device) for k, ds in sets.items()}
    scores = {a: {k: raw[k][a] for k in sets} for a in ("loss_threshold", "min_k_prob")}
    ref = scores["loss_threshold"] if base_loss is None else base_loss
    scores["ref_calibrated"] = {k: attacks.ref_calibrated(scores["loss_threshold"][k], ref[k]) for k in sets}
    return scores


def _evaluate(model, splits, cfg, device: str, base_loss: dict, eval_sets) -> dict:
    """Per-attack scores for every set, forget / retain losses and raw TOFU values."""
    from . import evaluate, tofu_metrics

    losses = {s: evaluate.mean_loss(model, getattr(splits, s), cfg, device) for s in ("forget", "retain")}
    out = {"scores": _scores(model, splits, cfg, device, base_loss), "losses": losses}
    if eval_sets:
        out["tofu"] = tofu_metrics.evaluate_model(model, eval_sets, cfg, device)
    return out


def run_central(cfg, device: str, methods=METHODS) -> dict:
    """Fine-tune once, then evaluate each method in `methods` from restored weights.

    Returns {"base", "pre", "post": {method: ...}, "nonmember_sets",
    "row_ids", "forget_is_outlier", "strata"}. `base` holds the pre-trained
    model's scores; `pre` is the fine-tuned model, the pre-unlearning
    reference for every method, including retrain. `pre` and each method
    carry raw TOFU values under "tofu" on the official splits.
    """
    import torch

    from . import attacks, data, outliers, tofu_metrics, unlearn
    from .seed import set_seed

    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; expected a subset of {METHODS}.")

    set_seed(cfg.seed)
    model, tokenizer = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg, tokenizer)
    eval_sets = tofu_metrics.load_eval_sets(cfg, tokenizer, splits)
    base = _snapshot(model) if "retrain" in methods else None
    with torch.random.fork_rng(devices=[]):  # scoring must not shift the fine-tune's data order
        base_scores = _scores(model, splits, cfg, device)
        outliers.tag_splits(splits, cfg, model, device, base_logprobs=base_scores["loss_threshold"])
    unlearn.finetune_base(cfg, splits, device, model=model)
    if not any(splits.forget_is_outlier):
        attacks.tag_outliers(model, splits, cfg, device)
    finetuned = _snapshot(model)
    base_loss = base_scores["loss_threshold"]
    pre = _evaluate(model, splits, cfg, device, base_loss, eval_sets)
    outliers.add_legacy(splits, cfg, model, device, finetuned_logprobs=pre["scores"]["loss_threshold"])

    result = {"base": {"scores": base_scores}, "pre": pre, "post": {},
              "nonmember_sets": list(data.nonmember_sets(splits)),
              "row_ids": {k: _row_ids(ds) for k, ds in {"forget": splits.forget,
                                                         **data.nonmember_sets(splits)}.items()},
              "forget_is_outlier": list(splits.forget_is_outlier), "strata": splits.strata}
    for method in methods:
        set_seed(cfg.seed)
        if method == "retrain":
            model.load_state_dict(base)
            unlearn.retrain_base(cfg, splits, device, model=model)
        else:
            model.load_state_dict(finetuned)
            unlearn.apply_unlearning(model, splits, replace(cfg, unlearn=replace(cfg.unlearn, method=method)),
                                     device)
        result["post"][method] = _evaluate(model, splits, cfg, device, base_loss, eval_sets)
        if cfg.attack.grad_alignment:
            result["post"][method]["alignment"] = _alignment(model, finetuned, splits, cfg, device,
                                                             ALIGN_SEED + 10 * cfg.seed + METHODS.index(method))
    return result


def _alignment(model, finetuned: dict, splits, cfg, device: str, seed: int) -> dict:
    """White-box scores of Δθ = θ_post − θ_pre and of a random direction of the
    same norm, on forget and every non-member set. Leaves `model` at θ_pre."""
    from . import update_attacks
    from .data import nonmember_sets

    delta = update_attacks.parameter_delta(model, finetuned)
    dirs = {"delta": delta, "random": update_attacks.random_direction(delta, seed)}
    norm = update_attacks._norm(delta)
    model.load_state_dict(finetuned)
    sets = {"forget": splits.forget, **nonmember_sets(splits)}
    raw = {k: update_attacks.grad_alignment_scores(model, dirs, ds, device) for k, ds in sets.items()}
    return {"delta_norm": norm, "scores": {d: {k: raw[k][d] for k in sets} for d in dirs}}


def _row_ids(ds) -> list[int]:
    from torch.utils.data import Subset

    if isinstance(ds, Subset):
        return [_row_ids(ds.dataset)[i] for i in ds.indices]
    return list(getattr(ds, "row_ids", range(len(ds))))


def _stages(result: dict) -> dict:
    return {"base": result["base"], "pre": result["pre"],
            **{f"post_{m}": post for m, post in result["post"].items()}}


def save_scores(result: dict, path) -> None:
    """Raw per-record values to one .npz: `{stage}__{attack}__{set}` scores,
    `row_ids__{set}`, and on the official splits `{stage}__truth_ratio__forget`,
    `{stage}__rouge__{set}` and `{stage}__generated__{set}`; the gradient
    alignment of each method as `post_<method>__grad_align__{set}` and
    `post_<method>__grad_align_random__{set}`. Stages: base, pre,
    post_<method>."""
    from pathlib import Path

    import numpy as np

    from . import tofu_metrics

    arrays = {f"row_ids__{k}": np.asarray(v) for k, v in result["row_ids"].items()}
    for stage, res in _stages(result).items():
        for a, by_set in res["scores"].items():
            arrays.update({f"{stage}__{a}__{k}": v.numpy() for k, v in by_set.items()})
        if "tofu" in res:
            arrays[f"{stage}__truth_ratio__forget"] = tofu_metrics.raw_truth_ratio(res["tofu"])
            for k, r in res["tofu"].items():
                arrays[f"{stage}__rouge__{k}"] = r["rouge"]
                arrays[f"{stage}__generated__{k}"] = np.array(r["generated"], dtype=str)
        for d, by_set in res.get("alignment", {}).get("scores", {}).items():
            name = "grad_align" if d == "delta" else f"grad_align_{d}"
            arrays.update({f"{stage}__{name}__{k}": v.numpy() for k, v in by_set.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _metrics(result: dict, stage: dict, attack: str, nm: str) -> dict:
    from . import evaluate

    s = stage["scores"][attack]
    return evaluate.mia_metrics({"forget": s["forget"], "holdout": s[nm],
                                 "forget_is_outlier": result["forget_is_outlier"],
                                 "strata": result["strata"].get(nm, {})})


def _gap(auc, ref):
    return None if auc is None or ref is None else auc - ref


def _tpr(stage: dict, attack: str, nm: str) -> float:
    from . import evaluate

    s = stage["scores"][attack]
    return evaluate.tpr_at_fpr(s["forget"].numpy(), s[nm].numpy(), TPR_FPR)


def _update_columns(result: dict, post: dict, attack: str, nm: str) -> dict:
    """Score change pre − post of `attack` and the gradient alignment with Δθ
    and with the random control (None when the alignment was not run)."""
    from . import evaluate, update_attacks

    pre_s, post_s = result["pre"]["scores"][attack], post["scores"][attack]
    pre_set = {"forget": pre_s["forget"].numpy(), nm: pre_s[nm].numpy()}
    post_set = {k: post_s[k].numpy() for k in pre_set}
    change = update_attacks.score_change(pre_set, post_set)
    auc = update_attacks.score_change_auc(pre_set, post_set)[nm]
    cols = {"auc_score_change": auc, "adv_score_change": abs(auc - 0.5),
            "tpr1_score_change": evaluate.tpr_at_fpr(change["forget"], change[nm], TPR_FPR)}
    align = post.get("alignment")
    cols["delta_norm"] = align["delta_norm"] if align else None
    for d, name in (("delta", "grad_align"), ("random", "grad_align_random")):
        s = align["scores"][d] if align else None
        a = evaluate._auc(s["forget"].numpy(), s[nm].numpy()) if s else None
        cols[f"auc_{name}"] = a
        cols[f"adv_{name}"] = None if a is None else abs(a - 0.5)
        cols[f"tpr1_{name}"] = evaluate.tpr_at_fpr(s["forget"].numpy(), s[nm].numpy(), TPR_FPR) if s else None
    return cols


def _tofu_columns(result: dict, post: dict) -> dict:
    """TOFU metrics of the fine-tuned model (`_pre`) and of the method (`_post`);
    forget quality against retrain (None when retrain is not in `result`)."""
    from . import tofu_metrics

    if "tofu" not in post:
        return {}
    ref = result["post"].get("retrain", {}).get("tofu")
    cols = {}
    for tag, raw in (("pre", result["pre"]["tofu"]), ("post", post["tofu"])):
        cols[f"forget_quality_{tag}"] = None if ref is None else tofu_metrics.forget_quality(
            tofu_metrics.raw_truth_ratio(raw), tofu_metrics.raw_truth_ratio(ref))
        cols.update({f"{k}_{tag}": v for k, v in tofu_metrics.summarise(raw).items()})
    return cols


def central_rows(cfg, result: dict) -> list[dict]:
    """One flat row per (method, attack, non-member set) with MIA metrics and losses.

    Columns: `auc_{base,pre,post}` (pre-trained, fine-tuned and post-method
    model), advantages |AUC - 0.5|, `tpr1_{base,pre,post}` (TPR at 1% FPR),
    and the retrain-relative gaps `gap_{pre,post}` = AUC - AUC of retrain
    (same seed, attack and non-member set; empty when retrain is not in
    `result`). Outlier and typical columns are suffixed by definition, e.g.
    `gap_post_outlier_basediff`. On the official splits the TOFU metrics of
    the fine-tuned model and the method follow (`forget_quality_post`,
    `model_utility_post`, `prob_retain_post`, ...); they repeat across the
    attack and non-member rows of a method. Update-signal columns:
    `{auc,adv,tpr1}_score_change` (score change pre − post of the row's
    attack), `{auc,adv,tpr1}_grad_align` and `..._grad_align_random`
    (white-box, the same for every attack of a method; None when
    `attack.grad_alignment` is off) and `delta_norm` = ‖Δθ‖.
    """
    from .outliers import DEFINITIONS

    keys = [("", "mia_auc", "mia_advantage")]
    keys += [(f"_{s}_{d}", f"mia_auc_{s}_{d}", f"mia_advantage_{s}_{d}")
             for d in DEFINITIONS for s in ("outlier", "typical")]
    rows = []
    for method, post in result["post"].items():
        tofu = _tofu_columns(result, post)
        for a in ATTACKS:
            for nm in result["nonmember_sets"]:
                bm = _metrics(result, result["base"], a, nm)
                pm = _metrics(result, result["pre"], a, nm)
                qm = _metrics(result, post, a, nm)
                rm = _metrics(result, result["post"]["retrain"], a, nm) if "retrain" in result["post"] else {}
                row = {"model": cfg.model.name, "seed": cfg.seed, "method": method, "attack": a, "nonmember": nm}
                for suffix, auc, adv in keys:
                    ref = rm.get(auc)
                    row[f"auc_base{suffix}"] = bm.get(auc)
                    row[f"auc_pre{suffix}"] = pm.get(auc)
                    row[f"auc_post{suffix}"] = qm.get(auc)
                    row[f"adv_pre{suffix}"] = pm.get(adv)
                    row[f"adv_post{suffix}"] = qm.get(adv)
                    row[f"gap_pre{suffix}"] = _gap(pm.get(auc), ref)
                    row[f"gap_post{suffix}"] = _gap(qm.get(auc), ref)
                    if suffix:
                        row[f"n_forget{suffix}"] = qm.get(f"n_forget{suffix}")
                        row[f"n_nonmember{suffix}"] = qm.get(f"n_holdout{suffix}")
                for tag, stage in (("base", result["base"]), ("pre", result["pre"]), ("post", post)):
                    row[f"tpr1_{tag}"] = _tpr(stage, a, nm)
                row.update({
                    "forget_loss_pre": result["pre"]["losses"]["forget"],
                    "forget_loss_post": post["losses"]["forget"],
                    "retain_loss_pre": result["pre"]["losses"]["retain"],
                    "retain_loss_post": post["losses"]["retain"],
                    **_update_columns(result, post, a, nm),
                    **tofu,
                })
                rows.append(row)
    return rows
