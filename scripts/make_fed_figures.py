#!/usr/bin/env python
"""Publication-quality figures for the federated experiment, matching the
single-node style in make_paper_figures.py.

Produces three vector PDFs (and 300-dpi PNGs) from the federated sweep CSV:
  * fed_scaling.pdf -- membership-inference AUC before vs after federated
                       unlearning, by model size.
  * fed_equity.pdf  -- post-unlearning MIA advantage by model and subgroup
                       (typical / outlier), mean with std error bars.
  * fed_update.pdf  -- the update-signal AUC by model size: how much the
                       unlearning update itself reveals the forgotten records.

Usage:
    python scripts/make_fed_figures.py --csv results/fed_sweep.csv --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL_ORDER = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "EleutherAI/pythia-410m"]


def _order(models) -> list[str]:
    present = [m for m in MODEL_ORDER if m in set(models)]
    return present + [m for m in models if m not in MODEL_ORDER]


def scaling_figure(df: pd.DataFrame, out: Path) -> None:
    long = df.melt(
        id_vars=["model", "seed"], value_vars=["pre_auc", "post_auc"],
        var_name="phase", value_name="auc",
    )
    long["phase"] = long["phase"].map({
        "pre_auc": "before unlearning", "post_auc": "after unlearning",
    })
    plt.figure(figsize=(6.0, 3.6))
    ax = sns.pointplot(
        data=long, x="model", y="auc", hue="phase",
        order=_order(df["model"].unique()),
        errorbar="sd", dodge=0.3, markers=["o", "s"], linestyles=["-", "--"],
        palette="deep",
    )
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"model (increasing size $\rightarrow$)")
    ax.set_ylabel("membership-inference AUC")
    ax.legend(title="", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()


def equity_figure(df: pd.DataFrame, out: Path) -> None:
    long = df.melt(
        id_vars=["model", "seed"],
        value_vars=["post_adv_typical", "post_adv_outlier"],
        var_name="subgroup", value_name="advantage",
    )
    long["subgroup"] = long["subgroup"].map({
        "post_adv_typical": "typical", "post_adv_outlier": "outlier",
    })
    plt.figure(figsize=(6.0, 3.6))
    ax = sns.barplot(
        data=long, x="model", y="advantage", hue="subgroup",
        order=_order(df["model"].unique()), hue_order=["typical", "outlier"],
        errorbar="sd", capsize=0.08, err_kws={"linewidth": 1.2}, palette="muted",
    )
    ax.set_xlabel(r"model (increasing size $\rightarrow$)")
    ax.set_ylabel(r"post-unlearning MIA advantage $|\mathrm{AUC}-0.5|$", fontsize=9)
    ax.legend(title="subgroup", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()


def update_figure(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(6.0, 3.6))
    ax = sns.pointplot(
        data=df, x="model", y="update_signal_auc",
        order=_order(df["model"].unique()),
        errorbar="sd", markers="D", linestyles="-", color="#b5651d",
    )
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"model (increasing size $\rightarrow$)")
    ax.set_ylabel("update-signal AUC")
    sns.despine()
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate federated figures from the sweep CSV")
    parser.add_argument("--csv", default="results/fed_sweep.csv")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args(argv)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif"})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    scaling_figure(df, out / "fed_scaling.pdf")
    equity_figure(df, out / "fed_equity.pdf")
    update_figure(df, out / "fed_update.pdf")
    print(f"wrote fed_scaling, fed_equity, fed_update to {out}/ (.pdf + .png)")


if __name__ == "__main__":
    main()
