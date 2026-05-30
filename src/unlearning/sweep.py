"""Multi-model, multi-seed sweep across two membership-inference attacks.

For each (model, seed) we fine-tune once, tag outliers, unlearn once, and then
score the pre- and post-unlearning models with BOTH attacks (no retraining per
attack). Results are appended to a CSV incrementally and the run is
resumable---safe for free Colab sessions that may disconnect.

Top-level imports are kept free of heavy deps so the aggregation logic can be
unit-tested without PyTorch; model/training imports are loaded lazily.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ATTACKS = ["loss_threshold", "min_k_prob"]

ROW_FIELDS = [
    "model", "seed", "attack",
    "auc_pre", "auc_post", "adv_pre", "adv_post",
    "adv_pre_outlier", "adv_post_outlier", "adv_pre_typical", "adv_post_typical",
    "forget_loss_pre", "forget_loss_post", "retain_loss_pre", "retain_loss_post",
]

_NUMERIC = set(ROW_FIELDS) - {"model", "attack", "seed"}


def run_single(cfg, device: str) -> list[dict]:
    """Fine-tune + unlearn once for `cfg`; return one row per attack."""
    from . import attacks, data, evaluate, unlearn
    from .seed import set_seed

    set_seed(cfg.seed)
    _, tokenizer = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg, tokenizer)
    model = unlearn.finetune_base(cfg, splits, device)
    if not any(splits.forget_is_outlier):
        attacks.tag_outliers(model, splits, cfg, device)

    f_pre = evaluate.mean_loss(model, splits.forget, cfg, device)
    r_pre = evaluate.mean_loss(model, splits.retain, cfg, device)
    pre = {a: attacks.run_attack(model, splits,
                                 replace(cfg, attack=replace(cfg.attack, method=a)), device)
           for a in ATTACKS}

    model = unlearn.apply_unlearning(model, splits, cfg, device)

    post = {a: attacks.run_attack(model, splits,
                                  replace(cfg, attack=replace(cfg.attack, method=a)), device)
            for a in ATTACKS}
    f_post = evaluate.mean_loss(model, splits.forget, cfg, device)
    r_post = evaluate.mean_loss(model, splits.retain, cfg, device)

    rows = []
    for a in ATTACKS:
        pm, qm = evaluate.mia_metrics(pre[a]), evaluate.mia_metrics(post[a])
        rows.append({
            "model": cfg.model.name, "seed": cfg.seed, "attack": a,
            "auc_pre": pm["mia_auc"], "auc_post": qm["mia_auc"],
            "adv_pre": pm["mia_advantage"], "adv_post": qm["mia_advantage"],
            "adv_pre_outlier": pm.get("mia_advantage_outlier"),
            "adv_post_outlier": qm.get("mia_advantage_outlier"),
            "adv_pre_typical": pm.get("mia_advantage_typical"),
            "adv_post_typical": qm.get("mia_advantage_typical"),
            "forget_loss_pre": f_pre, "forget_loss_post": f_post,
            "retain_loss_pre": r_pre, "retain_loss_post": r_post,
        })
    return rows


def _read_rows(csv_path: Path) -> list[dict]:
    """Read the raw CSV back with numeric coercion (empty -> None)."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if k == "seed":
                    out[k] = int(v)
                elif k in _NUMERIC:
                    out[k] = float(v) if v not in ("", "None") else None
                else:
                    out[k] = v
            rows.append(out)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean and (population) std across seeds, grouped by (model, attack)."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["attack"])].append(r)
    metrics = ["auc_pre", "auc_post", "adv_pre", "adv_post",
               "adv_post_outlier", "adv_post_typical",
               "forget_loss_post", "retain_loss_post"]
    summary = []
    for (model, attack), rs in sorted(groups.items()):
        entry = {"model": model, "attack": attack, "n_seeds": len(rs)}
        for k in metrics:
            vals = [r[k] for r in rs if r.get(k) is not None and math.isfinite(r[k])]
            if vals:
                entry[f"{k}_mean"] = round(statistics.mean(vals), 4)
                entry[f"{k}_std"] = round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0
        summary.append(entry)
    return summary


def main(argv: list[str] | None = None) -> None:
    from . import unlearn
    from .config import load_config

    parser = argparse.ArgumentParser(description="Multi-model/multi-seed unlearning sweep")
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument("--models", nargs="+", required=True, help="Model names")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Seeds")
    parser.add_argument("--out", default="results/sweep", help="Output directory")
    args = parser.parse_args(argv)

    base = load_config(args.config)
    device = unlearn.resolve_device(base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_csv = out / "sweep_raw.csv"

    done = {(r["model"], r["seed"]) for r in _read_rows(raw_csv)}
    write_header = not raw_csv.exists()
    with open(raw_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS, restval="")
        if write_header:
            writer.writeheader()
            f.flush()
        for model_name in args.models:
            for seed in args.seeds:
                if (model_name, seed) in done:
                    print(f"[sweep] skip {model_name} seed={seed} (already done)")
                    continue
                cfg = replace(base, model=replace(base.model, name=model_name), seed=seed,
                              name=f"{model_name.replace('/', '_')}_seed{seed}")
                print(f"[sweep] running model={model_name} seed={seed} device={device}")
                for row in run_single(cfg, device):
                    writer.writerow(row)
                f.flush()

    rows = _read_rows(raw_csv)
    summary = aggregate(rows)
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[sweep] {len(rows)} rows -> {raw_csv}")
    print(f"[sweep] summary -> {out / 'sweep_summary.json'}")
    try:
        from .plotting import render_sweep

        render_sweep(out)
        print(f"[sweep] figure  -> {out / 'figure_sweep.png'}")
    except Exception as exc:
        print(f"[sweep] figure skipped ({exc})")


if __name__ == "__main__":
    main()
