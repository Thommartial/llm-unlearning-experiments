#!/usr/bin/env python
"""Publication-quality result figures (matplotlib + seaborn) from the sweep CSV.

Produces two vector PDFs (and 300-dpi PNGs) suitable for an IEEE/Elsevier paper:
  * fig_equity.pdf  -- post-unlearning MIA advantage by model and subgroup
                       (all / typical / outlier), mean with std error bars.
  * fig_scaling.pdf -- membership-inference AUC before vs after unlearning,
                       by model size (the leakage-vs-scale trend).

Usage:
    python scripts/make_paper_figures.py --csv results/sweep/sweep_raw.csv --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Display order from smallest to largest model.
MODEL_ORDER = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "EleutherAI/pythia-410m"]


def _order(models) -> list[str]:
    present = [m for m in MODEL_ORDER if m in set(models)]
    return present + [m for m in models if m not in MODEL_ORDER]


def equity_figure(df: pd.DataFrame, attack: str, out: Path) -> None:
    d = df[df["attack"] == attack]
    long = d.melt(
        id_vars=["model", "seed"],
        value_vars=["adv_post", "adv_post_typical", "adv_post_outlier"],
        var_name="subgroup", value_name="advantage",
    )
    long["subgroup"] = long["subgroup"].map({
        "adv_post": "all", "adv_post_typical": "typical", "adv_post_outlier": "outlier",
    })
    plt.figure(figsize=(6.0, 3.6))
    ax = sns.barplot(
        data=long, x="model", y="advantage", hue="subgroup",
        order=_order(d["model"].unique()), hue_order=["all", "typical", "outlier"],
        errorbar="sd", capsize=0.08, err_kws={"linewidth": 1.2}, palette="muted",
    )
    ax.set_xlabel(r"model (increasing size $\rightarrow$)")
    ax.set_ylabel(r"post-unlearning MIA advantage $|\mathrm{AUC}-0.5|$")
    ax.legend(title="subgroup", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()


def scaling_figure(df: pd.DataFrame, out: Path) -> None:
    long = df.melt(
        id_vars=["model", "seed", "attack"], value_vars=["auc_pre", "auc_post"],
        var_name="phase", value_name="auc",
    )
    long["phase"] = long["phase"].map({
        "auc_pre": "before unlearning", "auc_post": "after unlearning",
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
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from the sweep CSV")
    parser.add_argument("--csv", default="results/sweep/sweep_raw.csv")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--attack", default="min_k_prob")
    args = parser.parse_args(argv)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    # Embed TrueType fonts (publisher-friendly) and use a serif face to match IEEE.
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif"})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    equity_figure(df, args.attack, out / "fig_equity.pdf")
    scaling_figure(df, out / "fig_scaling.pdf")
    print(f"wrote {out/'fig_equity.pdf'} and {out/'fig_scaling.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
