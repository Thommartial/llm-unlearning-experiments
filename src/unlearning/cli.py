"""Command-line entry point for the unlearning privacy experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import ExperimentConfig, load_config
from .seed import set_seed


def _plan(cfg: ExperimentConfig, out: Path) -> None:
    print(f"[plan] experiment '{cfg.name}' (seed={cfg.seed})")
    print(f"[plan] model={cfg.model.name}  data={cfg.data.dataset}  "
          f"forget_fraction={cfg.data.forget_fraction}")
    print(f"[plan] unlearn={cfg.unlearn.method}  attack={cfg.attack.method}")
    print(f"[plan] resolved config -> {out / 'config.resolved.json'}")


def run(cfg: ExperimentConfig, execute: bool) -> None:
    """Set up the run, write the resolved config, and (optionally) execute."""
    set_seed(cfg.seed)
    out = Path(cfg.output_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2))
    _plan(cfg, out)

    if not execute:
        print("[plan] dry run only. Re-run with --run to execute the pipeline.")
        return

    from . import attacks, data, evaluate, unlearn

    device = unlearn.resolve_device(cfg)
    print(f"[run] device={device}")

    model, tokenizer = unlearn.build_model_and_tokenizer(cfg)
    splits = data.load_splits(cfg, tokenizer)
    print(f"[run] splits: forget={len(splits.forget)} retain={len(splits.retain)} "
          f"holdout={len(splits.holdout)}")

    model = unlearn.finetune_base(cfg, splits, device)
    pre = attacks.run_attack(model, splits, cfg, device)
    pre_losses = {
        "forget": evaluate.mean_loss(model, splits.forget, cfg, device),
        "retain": evaluate.mean_loss(model, splits.retain, cfg, device),
    }
    print(f"[run] pre-unlearning  MIA AUC = {evaluate.mia_metrics(pre)['mia_auc']:.3f}  "
          f"(forget loss {pre_losses['forget']:.2f}, retain loss {pre_losses['retain']:.2f})")

    model = unlearn.apply_unlearning(model, splits, cfg, device)
    post = attacks.run_attack(model, splits, cfg, device)

    metrics = evaluate.summarise(cfg, model, splits, pre, post, pre_losses, device, out)
    print(f"[run] post-unlearning MIA AUC = {metrics['post_unlearning']['mia_auc']:.3f}  "
          f"(forget loss {metrics['forget_loss_post']:.2f}, retain loss {metrics['retain_loss_post']:.2f})")
    print(f"[run] metrics -> {out / 'metrics.json'}")

    try:  # figure generation is non-critical; never fail the run over a plot
        from .plotting import render

        render(out)
        print(f"[run] figure  -> {out / 'figure_mia.png'}")
    except Exception as exc:
        print(f"[run] figure skipped ({exc})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LLM unlearning privacy experiments")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--run", action="store_true",
                        help="Execute the pipeline (default: dry-run plan)")
    args = parser.parse_args(argv)
    run(load_config(args.config), execute=args.run)


if __name__ == "__main__":
    main()
