"""Figure generation from saved results (ROC + AUC bar chart)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


def _roc(ax, forget, holdout, label):
    y = np.concatenate([np.ones(len(forget)), np.zeros(len(holdout))])
    s = np.concatenate([forget, holdout])
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, label=label)


def render(run_dir: Path) -> Path:
    data = np.load(run_dir / "scores.npz")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    _roc(ax1, data["pre_forget"], data["pre_holdout"], "before unlearning")
    _roc(ax1, data["post_forget"], data["post_holdout"], "after unlearning")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="MIA ROC")
    ax1.legend()

    post = metrics["post_unlearning"]
    keys = [k for k in ("mia_auc", "mia_auc_typical", "mia_auc_outlier") if k in post]
    ax2.bar([k.replace("mia_auc", "all").replace("_", " ") for k in keys],
            [post[k] for k in keys])
    ax2.axhline(0.5, color="k", ls="--", lw=0.8)
    ax2.set(ylim=(0, 1), ylabel="AUC", title="Post-unlearning MIA AUC")

    fig.tight_layout()
    out = run_dir / "figure_mia.pdf"
    fig.savefig(out)
    fig.savefig(run_dir / "figure_mia.png", dpi=150)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render figures from results")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--name", help="Run name (subdirectory). Default: all runs.")
    args = parser.parse_args(argv)
    root = Path(args.results)
    runs = [root / args.name] if args.name else [p.parent for p in root.glob("*/scores.npz")]
    for run in runs:
        print(f"rendered {render(run)}")


if __name__ == "__main__":
    main()
