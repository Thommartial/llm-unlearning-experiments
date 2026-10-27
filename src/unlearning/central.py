"""Centralized runs: fine-tune once per (model, seed), then retrain and unlearn.

The base and fine-tuned weights are kept on CPU and restored before each
method, so retrain, GA, GD and NPO all start from the same weights and the
fine-tune runs once instead of once per method. Outliers are tagged with the
base model before fine-tuning (see `outliers`). Pre-unlearning scores and
losses come from the fine-tuned model and are shared by every method. Each
method is reseeded before it runs, so its result does not depend on which
other methods ran before it (a resumed run gives the same numbers).
"""

from __future__ import annotations

from dataclasses import replace

METHODS = ("retrain", "gradient_ascent", "gradient_difference", "npo")
ATTACKS = ("loss_threshold", "min_k_prob")


def _snapshot(model) -> dict:
    """CPU copy of the weights, restored later with load_state_dict."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _evaluate(model, splits, cfg, device: str) -> dict:
    """Per-attack scores (forget / holdout) and forget / retain losses."""
    from . import attacks, evaluate

    scores = {a: attacks.run_attack(model, splits, replace(cfg, attack=replace(cfg.attack, method=a)), device)
              for a in ATTACKS}
    losses = {s: evaluate.mean_loss(model, getattr(splits, s), cfg, device) for s in ("forget", "retain")}
    return {"scores": scores, "losses": losses}


def run_central(cfg, device: str, methods=METHODS) -> dict:
    """Fine-tune once, then evaluate each method in `methods` from restored weights.

    Returns {"pre": ..., "post": {method: ...}, "forget_is_outlier": [...], "strata": {...}}.
    `pre` is the fine-tuned model; it is the pre-unlearning reference for every
    method, including retrain.
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
        outliers.tag_splits(splits, cfg, model, device)
    unlearn.finetune_base(cfg, splits, device, model=model)
    if not any(splits.forget_is_outlier):
        attacks.tag_outliers(model, splits, cfg, device)
    outliers.add_legacy(splits)
    finetuned = _snapshot(model)

    result = {"pre": _evaluate(model, splits, cfg, device), "post": {},
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


def central_rows(cfg, result: dict) -> list[dict]:
    """One flat row per (method, attack) with MIA metrics and losses.

    Outlier and typical columns are suffixed by definition (`_basediff`,
    `_rarity`, `_legacy`), e.g. `adv_post_outlier_basediff`.
    """
    from . import evaluate
    from .outliers import DEFINITIONS

    pre = result["pre"]
    rows = []
    for method, post in result["post"].items():
        for a in ATTACKS:
            pm = evaluate.mia_metrics(pre["scores"][a])
            qm = evaluate.mia_metrics(post["scores"][a])
            row = {
                "model": cfg.model.name, "seed": cfg.seed, "method": method, "attack": a,
                "auc_pre": pm["mia_auc"], "auc_post": qm["mia_auc"],
                "adv_pre": pm["mia_advantage"], "adv_post": qm["mia_advantage"],
            }
            for d in DEFINITIONS:
                for s in ("outlier", "typical"):
                    row[f"auc_pre_{s}_{d}"] = pm.get(f"mia_auc_{s}_{d}")
                    row[f"auc_post_{s}_{d}"] = qm.get(f"mia_auc_{s}_{d}")
                    row[f"adv_pre_{s}_{d}"] = pm.get(f"mia_advantage_{s}_{d}")
                    row[f"adv_post_{s}_{d}"] = qm.get(f"mia_advantage_{s}_{d}")
                    row[f"n_forget_{s}_{d}"] = qm.get(f"n_forget_{s}_{d}")
                    row[f"n_holdout_{s}_{d}"] = qm.get(f"n_holdout_{s}_{d}")
            row.update({
                "forget_loss_pre": pre["losses"]["forget"], "forget_loss_post": post["losses"]["forget"],
                "retain_loss_pre": pre["losses"]["retain"], "retain_loss_post": post["losses"]["retain"],
            })
            rows.append(row)
    return rows
