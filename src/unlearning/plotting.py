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
    labels = {"mia_auc": "all", "mia_auc_typical": "typical", "mia_auc_outlier": "outlier"}
    keys = [k for k in labels if k in post]
    ax2.bar([labels[k] for k in keys], [post[k] for k in keys])
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


def render_sweep(out_dir, attack: str = "min_k_prob"):
    """Grouped bar of post-unlearning MIA advantage (typical vs outlier) per model."""
    import csv as _csv
    from collections import defaultdict

    rows = defaultdict(lambda: {"typical": [], "outlier": []})
    with open(Path(out_dir) / "sweep_raw.csv", newline="") as f:
        for r in _csv.DictReader(f):
            if r["attack"] != attack:
                continue
            if r.get("adv_post_typical") not in (None, "", "None"):
                rows[r["model"]]["typical"].append(float(r["adv_post_typical"]))
            if r.get("adv_post_outlier") not in (None, "", "None"):
                rows[r["model"]]["outlier"].append(float(r["adv_post_outlier"]))

    models = list(rows)
    x = np.arange(len(models))
    width = 0.38
    typ_m = [float(np.mean(rows[m]["typical"])) for m in models]
    typ_s = [float(np.std(rows[m]["typical"])) for m in models]
    out_m = [float(np.mean(rows[m]["outlier"])) for m in models]
    out_s = [float(np.std(rows[m]["outlier"])) for m in models]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - width / 2, typ_m, width, yerr=typ_s, capsize=3, label="typical")
    ax.bar(x + width / 2, out_m, width, yerr=out_s, capsize=3, label="outlier")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("post-unlearning MIA advantage")
    ax.set_title(f"Residual leakage by model ({attack}); mean $\\pm$ std over seeds")
    ax.legend()
    fig.tight_layout()
    out = Path(out_dir) / "figure_sweep.pdf"
    fig.savefig(out)
    fig.savefig(Path(out_dir) / "figure_sweep.png", dpi=150)
    return out
