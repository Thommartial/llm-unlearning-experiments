"""Run one federated unlearning experiment from a YAML config.

    python scripts/run_federated.py --config configs/smoke_fed_synthetic.yaml --run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from unlearning.config import load_config
from unlearning.federated import run_federated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Federated unlearning experiment")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--run", action="store_true",
                        help="Execute the pipeline (default: dry-run plan)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(cfg.output_dir) / cfg.name

    if not args.run:
        print(f"[plan] federated '{cfg.name}': model={cfg.model.name} "
              f"clients={cfg.federated.num_clients} rounds={cfg.federated.rounds} "
              f"unlearn={cfg.federated.unlearn_method}")
        print("[plan] dry run only. Re-run with --run to execute.")
        return

    metrics = run_federated(cfg, out)
    post = metrics["post_unlearning"]
    print(f"[done] post MIA AUC={post['mia_auc']:.3f} "
          f"advantage(outlier)={post.get('mia_advantage_outlier', float('nan')):.3f} "
          f"update-signal AUC={metrics['update_signal_auc']:.3f}")
    print(f"[done] metrics -> {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
