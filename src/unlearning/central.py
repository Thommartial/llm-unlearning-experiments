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
`holdout10` on the official splits, `holdout` on the legacy ones).
"""

from __future__ import annotations

from dataclasses import replace

METHODS = ("retrain", "gradient_ascent", "gradient_difference", "npo")
ATTACKS = ("loss_threshold", "min_k_prob")


def _snapshot(model) -> dict:
    """CPU copy of the weights, restored later with load_state_dict."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _scores(model, splits, cfg, device: str) -> dict:
    """attack -> {"forget": scores, <non-member set>: scores, ...}."""
    from . import attacks
    from .data import nonmember_sets

    sets = {"forget": splits.forget, **nonmember_sets(splits)}
    return {a: {k: attacks.score_dataset(model, ds, replace(cfg, attack=replace(cfg.attack, method=a)), device)
                for k, ds in sets.items()}
            for a in ATTACKS}


def _evaluate(model, splits, cfg, device: str) -> dict:
    """Per-attack scores for every set, and forget / retain losses."""
    from . import evaluate

    losses = {s: evaluate.mean_loss(model, getattr(splits, s), cfg, device) for s in ("forget", "retain")}
    return {"scores": _scores(model, splits, cfg, device), "losses": losses}


def run_central(cfg, device: str, methods=METHODS) -> dict:
    """Fine-tune once, then evaluate each method in `methods` from restored weights.

    Returns {"base", "pre", "post": {method: ...}, "nonmember_sets",
    "forget_is_outlier", "strata"}. `base` holds the pre-trained model's
    scores; `pre` is the fine-tuned model, the pre-unlearning reference for
    every method, including retrain.
    """
    import torch

    from . import attacks, data, outliers, unlearn
    from .seed import set_seed

    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; expected a subset of {METHODS}.")

    set_seed(cfg.seed)
    model, tokenizer = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg, tokenizer)
    base = _snapshot(model) if "retrain" in methods else None
    with torch.random.fork_rng(devices=[]):  # scoring must not shift the fine-tune's data order
        base_scores = _scores(model, splits, cfg, device)
        outliers.tag_splits(splits, cfg, model, device, base_logprobs=base_scores["loss_threshold"])
    unlearn.finetune_base(cfg, splits, device, model=model)
    if not any(splits.forget_is_outlier):
        attacks.tag_outliers(model, splits, cfg, device)
    finetuned = _snapshot(model)
    pre = _evaluate(model, splits, cfg, device)
    outliers.add_legacy(splits, cfg, model, device, finetuned_logprobs=pre["scores"]["loss_threshold"])

    result = {"base": {"scores": base_scores}, "pre": pre, "post": {},
              "nonmember_sets": list(data.nonmember_sets(splits)),
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
        result["post"][method] = _evaluate(model, splits, cfg, device)
    return result


def _metrics(result: dict, stage: dict, attack: str, nm: str) -> dict:
    from . import evaluate

    s = stage["scores"][attack]
    return evaluate.mia_metrics({"forget": s["forget"], "holdout": s[nm],
                                 "forget_is_outlier": result["forget_is_outlier"],
                                 "strata": result["strata"].get(nm, {})})


def _gap(auc, ref):
    return None if auc is None or ref is None else auc - ref


def central_rows(cfg, result: dict) -> list[dict]:
    """One flat row per (method, attack, non-member set) with MIA metrics and losses.

    Columns: `auc_{base,pre,post}` (pre-trained, fine-tuned and post-method
    model), advantages |AUC - 0.5|, and the retrain-relative gaps
    `gap_{pre,post}` = AUC - AUC of retrain (same seed, attack and non-member
    set; empty when retrain is not in `result`). Outlier and typical columns
    are suffixed by definition, e.g. `gap_post_outlier_basediff`.
    """
    from .outliers import DEFINITIONS

    keys = [("", "mia_auc", "mia_advantage")]
    keys += [(f"_{s}_{d}", f"mia_auc_{s}_{d}", f"mia_advantage_{s}_{d}")
             for d in DEFINITIONS for s in ("outlier", "typical")]
    rows = []
    for method, post in result["post"].items():
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
                row.update({
                    "forget_loss_pre": result["pre"]["losses"]["forget"],
                    "forget_loss_post": post["losses"]["forget"],
                    "retain_loss_pre": result["pre"]["losses"]["retain"],
                    "retain_loss_post": post["losses"]["retain"],
                })
                rows.append(row)
    return rows
