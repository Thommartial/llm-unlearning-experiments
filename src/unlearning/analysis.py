"""Revision analysis (ANALYSIS_PLAN §4-§6): statistics, LaTeX tables and figures.

Reads a results folder written by `scripts/run_revision.py`: the merged
campaign CSVs (`<campaign>.csv`; per-GPU CSVs are skipped), the pilot's
`pilot.csv` and the LoRA pilot's `pilot_lora.csv`, the raw per-record scores `scores/<campaign>/<unit>.npz` and the
resolved configs `configs/<campaign>/<unit>.json`. Rows are told apart by
their `kind` (central or federated), so the smoke and micro outputs work too.

Statistics: mean and 95% t-interval over seeds; differences within a seed
(H1 gaps, H3 outlier - typical, update-attack gaps to retrain, cost ratios)
are paired by seed. Dropped as uninformative: `auc_base` for ref_calibrated
(0.5 by construction) and the ref_calibrated score change (equal to the loss
score change). Everything is CPU only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ORDER = ["tiny", "tiny_llama", "tiny_neox", "distilgpt2", "gpt2", "gpt2-medium", "EleutherAI/pythia-1b",
               "TinyLlama/TinyLlama_v1.1"]
MODEL_NAMES = {"distilgpt2": "DistilGPT-2", "gpt2": "GPT-2", "gpt2-medium": "GPT-2-medium",
               "EleutherAI/pythia-1b": "Pythia-1B (LoRA)", "TinyLlama/TinyLlama_v1.1": "TinyLlama (LoRA)",
               "tiny": "tiny", "tiny_llama": "tiny Llama", "tiny_neox": "tiny GPT-NeoX"}
GPT2_PARAMS = {"distilgpt2": 82e6, "gpt2": 124e6, "gpt2-medium": 355e6}  # ANALYSIS_PLAN §2
ATTACKS = ["loss_threshold", "min_k_prob", "ref_calibrated"]
ATTACK_NAMES = {"loss_threshold": "Loss thr.", "min_k_prob": r"Min-K\%", "ref_calibrated": "Ref.-calib."}
METHOD_ORDER = ["retrain", "gradient_ascent", "gradient_difference", "npo", "client_ga_upload", "client_ga"]
METHOD_NAMES = {"retrain": "Retrain", "gradient_ascent": "GA", "gradient_difference": "GD", "npo": "NPO",
                "client_ga_upload": "Client GA, upload", "client_ga": "Client GA, global"}
NONMEMBERS = ["primary", "holdout10"]
VIEW_NAMES = {"central": "release", "any_client": "any client",
              "server, and all clients at the first recovery broadcast": "server$^{*}$"}
DEFINITIONS = {"basediff": "primary", "rarity": "secondary", "legacy": "ablation", "legacy_pooled": "ablation"}
KEY = ["campaign", "kind", "model", "variant"]
RUN_NAMES = {"lora": "central"}  # the LoRA campaign is centralized; the model name says LoRA
COLLAPSED_MARK = "$^{\\dagger}$"
COLLAPSE_RATIO = 3.0  # retain loss after unlearning over the pre-unlearning retain loss, in every seed
NOTE_METHOD_NAMES = {"client_ga_upload": "client-GA upload", "client_ga": "client-GA global"}
NUMBER_WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
MU_SPLITS = {"retain": "retain", "real_authors": "real authors", "world_facts": "world facts"}
MU_METRICS = {"prob": "probability", "rouge": "ROUGE", "truth_ratio": "truth ratio"}
SUPPLEMENT_SECTIONS = [  # after the central/federated tables; full tables only, no paper versions
    ("Hypotheses", ["tab_h1.tex", "tab_h1_two_sided.tex", "tab_h2.tex", "tab_h3.tex", "tab_h3_ablation.tex"]),
    ("Low false-positive rates, version-change attacks and control",
     ["tab_low_fpr.tex", "tab_update.tex", "tab_control.tex"]),
    ("Costs, TOFU metrics and settings",
     ["tab_costs.tex", "tab_tofu.tex", "tab_hyperparameters.tex", "tab_pilot_slowdown.tex"])]
MAIN_TEXT_REFS = {"tab:h1-paper": "the H1 table in the main text",
                  "tab:h1-two-sided-paper": "the two-sided table in the main text"}
CITATIONS = {"shi2024muse": "Shi et al. (ICLR 2025)"}
SUPPLEMENT_PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[letterpaper,landscape,margin=0.65in]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\setlength{\LTcapwidth}{\textwidth}
\renewcommand{\thetable}{S\arabic{table}}
\title{Supplementary tables}
\date{}
\begin{document}
\maketitle
Every table is written by \texttt{scripts/analyze\_revision.py} from the campaign results in
\texttt{results/revision}. Cells give the mean and the half-width of the 95\% $t$-interval over seeds;
differences are paired by seed. Tables in the main text select the loss-threshold attack and the
primary non-members from these.

"""
PAPER_SCOPE = ("loss_threshold", "primary")  # attack and non-member set of the paper's H1 tables
PILOT_CSVS = ("pilot.csv", "pilot_lora.csv")  # timing rows of the pilot and the LoRA pilot, not campaigns


# ---------------------------------------------------------------- loading


def load(results) -> dict:
    """{"rows": campaign rows, "pilot": pilot rows or None, "configs": [(campaign, unit, config)],
    "scores": folder of the .npz files}. Applies the drops and adds adv_base."""
    results = Path(results)
    frames = [pd.read_csv(p) for p in sorted(results.glob("*.csv"))
              if "_gpu" not in p.stem and p.name not in PILOT_CSVS]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not rows.empty:
        rows = _prepare(rows)
    pilots = [pd.read_csv(results / n) for n in PILOT_CSVS if (results / n).exists()]
    configs = [(p.parent.name, p.stem, json.loads(p.read_text())) for p in sorted(results.glob("configs/*/*.json"))]
    return {"rows": rows, "pilot": _prepare(pd.concat(pilots, ignore_index=True)) if pilots else None,
            "configs": configs, "scores": results / "scores"}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ref = df["attack"] == "ref_calibrated"
    for col in [c for c in df.columns if c.startswith("auc_base") or c.endswith("_score_change")]:
        df.loc[ref, col] = np.nan
    df["adv_base"] = (df["auc_base"] - 0.5).abs()
    df["seed"] = df["seed"].astype(int)
    return df


def _order(values, order) -> list:
    present = list(dict.fromkeys(values))
    return [v for v in order if v in present] + [v for v in present if v not in order]


def per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (campaign, model, variant, seed, method): for values that
    repeat across the attack and non-member rows (TOFU, costs, alignment)."""
    first_nm = df["nonmember"].iloc[0] if len(df) else None
    keep = (df["attack"] == "loss_threshold") & (df["nonmember"] == first_nm)
    return df[keep]


# ---------------------------------------------------------------- statistics


def t_ci(values, level: float = 0.95) -> dict:
    """Mean and t-interval half-width over the finite values (half NaN for n < 2)."""
    from scipy import stats

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": np.nan, "half": np.nan, "lo": np.nan, "hi": np.nan}
    mean = float(v.mean())
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    return {"n": n, "mean": mean, "half": half, "lo": mean - half, "hi": mean + half}


def summarise(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    """Long table: one row per group and column with n, mean, half, lo, hi."""
    out = []
    for keys, g in df.groupby(by, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for c in cols:
            if c in g:
                out.append({**dict(zip(by, keys, strict=True)), "column": c, **t_ci(g[c])})
    return pd.DataFrame(out)


def _excludes_zero(s: dict) -> bool | None:
    if not np.isfinite(s["half"]):
        return None
    return bool(s["lo"] > 0 or s["hi"] < 0)


def _verdict(s: dict, above: str, below: str) -> str | None:
    """Three-way reading of a paired CI: `above` (lower bound > 0), `below`
    (upper bound < 0) or "inconclusive"; None without a CI (one seed)."""
    if not np.isfinite(s["half"]):
        return None
    return above if s["lo"] > 0 else below if s["hi"] < 0 else "inconclusive"


def h1(df: pd.DataFrame) -> pd.DataFrame:
    """H1: post-unlearning AUC minus the retrained model's (same seed), per
    method, attack and non-member set; mean, CI and the verdict: "above retrain"
    (lower bound > 0, supports H1), "below retrain" (upper bound < 0,
    contradicts H1) or "inconclusive"."""
    out = []
    d = df[df["method"] != "retrain"]
    for keys, g in d.groupby(KEY + ["method", "attack", "nonmember"], sort=False):
        s = t_ci(g["gap_post"])
        out.append({**dict(zip(KEY + ["method", "attack", "nonmember"], keys, strict=True)), **s,
                    "verdict": _verdict(s, "above retrain", "below retrain")})
    return pd.DataFrame(out)


def h1_two_sided(df: pd.DataFrame) -> pd.DataFrame:
    """Exploratory, post hoc (ANALYSIS_PLAN §10): the two-sided distance
    |AUC - AUC_retrain| per seed (|gap_post|; federated rows against the
    federated retrain), overall and per stratum, with mean and CI over seeds.
    Also the signed mean gap, the number of seeds below retrain, MUSE's
    PrivLeak (AUC_post - AUC_retrain) / AUC_retrain per seed (Shi et al., ICLR
    2025; mean and CI) and the TOFU forget quality. H1 itself is unchanged."""
    out = []
    d = df[df["method"] != "retrain"]
    strata = [("all", "", None)] + [(f"{part}_{defn}", f"_{part}_{defn}", defn)
                                    for defn in DEFINITIONS for part in ("typical", "outlier")]
    for keys, g in d.groupby(KEY + ["method", "attack", "nonmember"], sort=False):
        fq = t_ci(g["forget_quality_post"]) if "forget_quality_post" in g else t_ci([])
        for stratum, suffix, defn in strata:
            col = f"gap_post{suffix}"
            if col not in g or g[col].isna().all():
                continue
            s = t_ci(g[col].abs())
            auc = f"auc_post{suffix}"
            leak = t_ci(g[col] / (g[auc] - g[col]) if auc in g else [])  # AUC_retrain = AUC_post - gap
            out.append({**dict(zip(KEY + ["method", "attack", "nonmember"], keys, strict=True)), "stratum": stratum,
                        "definition": defn, "role": DEFINITIONS.get(defn, "overall"), **s,
                        "gap_mean": t_ci(g[col])["mean"], "n_below_retrain": int((g[col] < 0).sum()),
                        **{f"privleak_{k}": leak[k] for k in ("mean", "half", "lo", "hi")},
                        "forget_quality_mean": fq["mean"], "forget_quality_half": fq["half"]})
    return pd.DataFrame(out)


def h2(df: pd.DataFrame) -> pd.DataFrame:
    """H2 over the GPT-2 series (central): slope of gap_post on log10(parameters),
    fitted within each seed (the seeds share the non-member draw across models),
    as the mean and t-interval over seeds (df = seeds - 1); per model: gap_post,
    |AUC - 0.5| after unlearning and the relative drop (adv_pre - adv_post) / adv_pre."""
    from scipy import stats

    d = df[(df["kind"] == "central") & df["model"].isin(GPT2_PARAMS) & (df["method"] != "retrain")]
    out = []
    for keys, g in d.groupby(["method", "attack", "nonmember"], sort=False):
        row = dict(zip(["method", "attack", "nonmember"], keys, strict=True))
        slopes = []
        for _, gs in g.groupby("seed", sort=True):
            x = np.log10(gs["model"].map(GPT2_PARAMS).to_numpy(float))
            y = gs["gap_post"].to_numpy(float)
            ok = np.isfinite(y)
            if len(set(x[ok])) > 1:
                slopes.append(stats.linregress(x[ok], y[ok]).slope)
        if slopes:
            fit = t_ci(slopes)
            row.update(slope=fit["mean"], slope_half=fit["half"], slope_lo=fit["lo"], slope_hi=fit["hi"],
                       n_seeds=fit["n"])
        for m in _order(g["model"], MODEL_ORDER):
            gm = g[g["model"] == m]
            drop = (gm["adv_pre"] - gm["adv_post"]) / gm["adv_pre"]
            for name, vals in (("gap_post", gm["gap_post"]), ("adv_post", gm["adv_post"]), ("rel_drop", drop)):
                s = t_ci(vals)
                row[f"{m}__{name}"] = s["mean"]
                row[f"{m}__{name}_half"] = s["half"]
        out.append(row)
    return pd.DataFrame(out)


def h3(df: pd.DataFrame) -> pd.DataFrame:
    """H3: gap_post of outliers minus typical records (paired within a seed),
    per definition (basediff primary, rarity secondary, legacy ablations),
    with the stratum advantages |AUC - 0.5|."""
    out = []
    d = df[df["method"] != "retrain"]
    for keys, g in d.groupby(KEY + ["method", "attack", "nonmember"], sort=False):
        for defn, role in DEFINITIONS.items():
            o, t = f"gap_post_outlier_{defn}", f"gap_post_typical_{defn}"
            if o not in g:
                continue
            s = t_ci(g[o] - g[t])
            out.append({**dict(zip(KEY + ["method", "attack", "nonmember"], keys, strict=True)), "definition": defn,
                        "role": role, **s, "verdict": _verdict(s, "outliers higher", "typical higher"),
                        "gap_outlier": t_ci(g[o])["mean"], "gap_typical": t_ci(g[t])["mean"],
                        "adv_outlier": t_ci(g[f"adv_post_outlier_{defn}"])["mean"],
                        "adv_typical": t_ci(g[f"adv_post_typical_{defn}"])["mean"]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- pooled low-FPR


def _slug(model: str, seed: int, variant: str) -> str:
    return f"{model.replace('/', '_')}_seed{seed}_{variant}"


def pooled_low_fpr(df: pd.DataFrame, scores_dir) -> pd.DataFrame:
    """TPR at 0.1% FPR with forget and non-member scores pooled over the seeds
    (400 negatives per run cannot resolve 0.1%; 5 seeds give 2,000), and TPR
    at 1% FPR within the outlier and typical strata (basediff, rarity: forget
    outliers against non-member outliers), pooled the same way. Next to the
    pooled TPR@0.1%: its per-seed mean and t-interval; with 400 non-members a
    seed allows no false positive at 0.1%, so this is the TPR at zero false
    positives (`n_nonmember_per_seed` gives the count)."""
    from .evaluate import tpr_at_fpr

    out = []
    for keys, g in df.groupby(KEY, sort=False):
        campaign, kind, model, variant = keys
        files = {s: Path(scores_dir) / campaign / f"{_slug(model, s, variant)}.npz" for s in sorted(g["seed"].unique())}
        files = {s: np.load(f) for s, f in files.items() if f.exists()}
        if not files:
            continue
        stages = ["base", "pre"] + [f"post_{m}" for m in _order(g["method"].unique(), METHOD_ORDER)]
        for stage in stages:
            for attack in ATTACKS:
                for nm in NONMEMBERS:
                    k_f, k_h = f"{stage}__{attack}__forget", f"{stage}__{attack}__{nm}"
                    zs = [z for z in files.values() if k_f in z and k_h in z]
                    if not zs:
                        continue
                    f = np.concatenate([z[k_f] for z in zs])
                    h = np.concatenate([z[k_h] for z in zs])
                    row = {**dict(zip(KEY, keys, strict=True)), "stage": stage, "attack": attack, "nonmember": nm,
                           "n_seeds": len(zs), "n_forget": len(f), "n_nonmember": len(h),
                           "tpr_at_0.1": tpr_at_fpr(f, h, 0.001), "tpr_at_1": tpr_at_fpr(f, h, 0.01)}
                    per_seed = t_ci([tpr_at_fpr(z[k_f], z[k_h], 0.001) for z in zs])
                    sizes = sorted({len(z[k_h]) for z in zs})
                    row.update({"tpr_at_0.1_seed_mean": per_seed["mean"], "tpr_at_0.1_seed_half": per_seed["half"],
                                "n_nonmember_per_seed": sizes[0] if len(sizes) == 1 else np.nan})
                    for defn in ("basediff", "rarity"):
                        kt_f, kt_h = f"strata__{nm}__{defn}__forget", f"strata__{nm}__{defn}__holdout"
                        if not all(kt_f in z and kt_h in z for z in zs):
                            continue
                        tf = np.concatenate([z[kt_f] for z in zs])
                        th = np.concatenate([z[kt_h] for z in zs])
                        row[f"tpr_at_1_outlier_{defn}"] = tpr_at_fpr(f[tf], h[th], 0.01)
                        row[f"tpr_at_1_typical_{defn}"] = tpr_at_fpr(f[~tf], h[~th], 0.01)
                        row[f"n_nonmember_outlier_{defn}"] = int(th.sum())
                    out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- update attacks


def update_attacks(df: pd.DataFrame) -> pd.DataFrame:
    """H4/H5: score-change AUC (loss and Min-K) and gradient-alignment AUC per
    method or stage, with the gap to retrain (same seed, attack and set)."""
    d = df[df["attack"] != "ref_calibrated"].copy()
    ref = d[d["method"] == "retrain"].set_index(KEY + ["seed", "attack", "nonmember"])
    idx = d.set_index(KEY + ["seed", "attack", "nonmember"]).index
    for col in ("auc_score_change", "auc_grad_align"):
        d[f"gap_{col}"] = d[col].to_numpy() - ref[col].reindex(idx).to_numpy()
    out = []
    for keys, g in d.groupby(KEY + ["method", "attack", "nonmember"], sort=False):
        row = dict(zip(KEY + ["method", "attack", "nonmember"], keys, strict=True))
        # central CSVs have no view column: NaN once concatenated with federated rows
        row["view"] = g["view"].iloc[0] if "view" in g and pd.notna(g["view"].iloc[0]) else "central"
        for col in ("auc_score_change", "gap_auc_score_change", "auc_grad_align", "gap_auc_grad_align",
                    "auc_grad_align_random"):
            s = t_ci(g[col])
            row[col], row[f"{col}_half"] = s["mean"], s["half"]
            if col.startswith("gap"):
                row[f"{col}_excludes_zero"] = _excludes_zero(s)
        row["n"] = g["seed"].nunique()
        out.append(row)
    return pd.DataFrame(out)


def random_control(df: pd.DataFrame, scores_dir=None) -> pd.DataFrame:
    """The random-direction control as a distribution over all (method or
    stage, seed) draws per model and non-member set: count, mean, SD, range,
    and the SD of the AUC for independent records (from the set sizes)."""
    d = df[df["attack"] == "loss_threshold"]
    out = []
    for keys, g in d.groupby(["kind", "model", "nonmember"], sort=False):
        v = g["auc_grad_align_random"].to_numpy(float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        sizes = _set_sizes(g, scores_dir, keys[2]) if scores_dir else None
        null = math.sqrt((sum(sizes) + 1) / (12 * sizes[0] * sizes[1])) if sizes else np.nan
        out.append({"kind": keys[0], "model": keys[1], "nonmember": keys[2], "n_draws": len(v),
                    "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                    "min": float(v.min()), "max": float(v.max()), "null_sd": null,
                    "share_outside_2_null_sd": float(np.mean(np.abs(v - 0.5) > 2 * null)) if sizes else np.nan,
                    "auc_grad_align_mean": float(np.nanmean(g["auc_grad_align"]))})
    return pd.DataFrame(out)


def _set_sizes(g: pd.DataFrame, scores_dir, nm: str):
    r = g.iloc[0]
    f = Path(scores_dir) / r["campaign"] / f"{_slug(r['model'], int(r['seed']), r['variant'])}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return len(z["row_ids__forget"]), len(z[f"row_ids__{nm}"])


# ---------------------------------------------------------------- costs, TOFU


def costs(df: pd.DataFrame) -> pd.DataFrame:
    """H6 (federated): client GA (+ recovery) against the federated retrain,
    per setting: means of time, peak memory (allocated, reserved) and
    communication, and the per-seed ratios client_ga / retrain."""
    d = per_unit(df[df["kind"] == "federated"])
    cols = ["time_s_method", "peak_mem_mib_method", "peak_mem_reserved_mib_method", "comm_bytes_method"]
    out = []
    for keys, g in d.groupby(KEY, sort=False):
        row = {**dict(zip(KEY, keys, strict=True)), "num_clients": int(g["num_clients"].iloc[0]),
               "sharding": g["sharding"].iloc[0]}
        ga = g[g["method"] == "client_ga"].set_index("seed")
        rt = g[g["method"] == "retrain"].set_index("seed")
        for c in cols:
            if c not in g:
                continue
            for name, part in (("client_ga", ga), ("retrain", rt)):
                s = t_ci(part[c])
                row[f"{c}__{name}"], row[f"{c}__{name}_half"] = s["mean"], s["half"]
            s = t_ci((ga[c] / rt[c]).dropna())
            row[f"{c}__ratio"], row[f"{c}__ratio_half"] = s["mean"], s["half"]
        row["n"] = len(ga.index.intersection(rt.index))
        out.append(row)
    return pd.DataFrame(out)


def central_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Wall time and peak memory per phase for the central runs (mean over seeds)."""
    d = per_unit(df[df["kind"] == "central"])
    phases = [c for c in d.columns if c.startswith(("time_s_", "peak_mem_mib_", "peak_mem_reserved_mib_"))]
    return summarise(d, KEY + ["method"], phases)


def pilot_slowdown(df: pd.DataFrame, pilot: pd.DataFrame | None) -> pd.DataFrame:
    """Time of each phase in the campaign (two processes) divided by the
    pilot's (GPT-2 seed 0, and the LoRA pilot's LoRA unit, each run alone),
    for the same model, seed, variant and method."""
    if pilot is None or df.empty:
        return pd.DataFrame()
    p = per_unit(pilot)
    out = []
    for _, r in p.iterrows():
        m = per_unit(df)
        m = m[(m["model"] == r["model"]) & (m["seed"] == r["seed"]) & (m["variant"] == r["variant"])
              & (m["method"] == r["method"])]
        if m.empty:
            continue
        c = m.iloc[0]
        for col in [k for k in r.index if k.startswith("time_s_") and pd.notna(r[k]) and pd.notna(c.get(k))]:
            out.append({"model": r["model"], "variant": r["variant"], "method": r["method"],
                        "phase": col[len("time_s_"):], "alone_s": r[col], "campaign_s": c[col],
                        "slowdown": c[col] / r[col] if r[col] else np.nan,
                        "same_gpu": r.get("gpu_name") == c.get("gpu_name")})
    return pd.DataFrame(out)


def tofu(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forget quality and model utility per method (mean and CI over seeds),
    model utility without the rows where it is 0 (`model_utility_post_nonzero`,
    a sensitivity check), and the count of rows with model utility 0 (one row
    per seed and method)."""
    d = per_unit(df)
    d = d.assign(model_utility_post_nonzero=d["model_utility_post"].where(d["model_utility_post"] != 0))
    summary = summarise(d, KEY + ["method"], ["forget_quality_post", "model_utility_post",
                                              "model_utility_post_nonzero", "model_utility_pre"])
    zeros = []
    for keys, g in d.groupby(KEY, sort=False):
        for col in ("model_utility_pre", "model_utility_post"):
            v = g[col] if col != "model_utility_pre" else g.drop_duplicates("seed")[col]
            zeros.append({**dict(zip(KEY, keys, strict=True)), "column": col, "rows": int(v.notna().sum()),
                          "zero": int((v == 0).sum())})
    return summary, pd.DataFrame(zeros)


def collapsed(df: pd.DataFrame) -> pd.DataFrame:
    """(model, variant, method) whose retain loss after unlearning exceeds
    COLLAPSE_RATIO x the pre-unlearning retain loss in every seed with a value:
    marked as a collapsed model in the tables (never retrain, whose ratio is ~1)."""
    out = []
    if not {"retain_loss_post", "retain_loss_pre"} <= set(df.columns):
        return pd.DataFrame(columns=["model", "variant", "method", "over", "rows"])
    for keys, g in per_unit(df).groupby(["model", "variant", "method"], sort=False):
        v = (g["retain_loss_post"] / g["retain_loss_pre"]).dropna()
        if len(v) and (v > COLLAPSE_RATIO).all():
            out.append({**dict(zip(["model", "variant", "method"], keys, strict=True)), "over": int(len(v)),
                        "rows": int(len(v))})
    return pd.DataFrame(out, columns=["model", "variant", "method", "over", "rows"])


def mu_zero_components(df: pd.DataFrame) -> pd.DataFrame:
    """For the rows with model utility 0: which of its nine harmonic-mean
    components (probability, ROUGE, truth ratio on retain, real authors and
    world facts) is 0, and in how many seeds."""
    out = []
    d = per_unit(df)
    if "model_utility_post" not in d:
        return pd.DataFrame(columns=["model", "variant", "method", "metric", "split", "zero", "rows"])
    for keys, g in d.groupby(["model", "variant", "method"], sort=False):
        z = g[g["model_utility_post"] == 0]
        for split in MU_SPLITS:
            for metric in MU_METRICS:
                col = f"{metric}_{split}_post"
                if col in z and (z[col] == 0).any():
                    out.append({**dict(zip(["model", "variant", "method"], keys, strict=True)), "metric": metric,
                                "split": split, "zero": int((z[col] == 0).sum()),
                                "rows": int(g["model_utility_post"].notna().sum())})
    return pd.DataFrame(out, columns=["model", "variant", "method", "metric", "split", "zero", "rows"])


def _flatten(name: str, value) -> dict:
    """{dotted name: JSON value}, one entry per key of nested dicts."""
    if isinstance(value, dict) and value:
        return {k: v for sub, val in sorted(value.items()) for k, v in _flatten(f"{name}.{sub}", val).items()}
    return {name: json.dumps(value, sort_keys=True)}


def hyperparameters(configs: list[tuple[str, str, dict]], kinds: dict) -> pd.DataFrame:
    """Settings of each campaign from the resolved configs; a setting that
    differs between the units of a campaign lists its values. Dict settings
    (model.lora) get one row per key."""
    keys = [("model", "precision"), ("model", "lora"), ("data", "max_length"), ("data", "max_samples"),
            ("data", "outlier_fraction"), ("data", "max_new_tokens"), ("unlearn", "finetune_epochs"),
            ("unlearn", "finetune_lr"), ("unlearn", "train_batch_size"), ("unlearn", "epochs"),
            ("unlearn", "learning_rate"), ("unlearn", "batch_size"), ("unlearn", "eval_batch_size"),
            ("unlearn", "retain_weight"), ("unlearn", "beta"), ("attack", "min_k")]
    fed = [("federated", "num_clients"), ("federated", "sharding"), ("federated", "rounds"),
           ("federated", "local_epochs"), ("federated", "client_unlearn_epochs"), ("federated", "rho_factor"),
           ("federated", "recovery_rounds")]
    by_campaign = {}
    for campaign, _, cfg in configs:
        by_campaign.setdefault(campaign, []).append(cfg)
    out = {}
    for campaign, cfgs in by_campaign.items():
        col = {"models": ", ".join(_order({c["model"]["name"] for c in cfgs}, MODEL_ORDER)),
               "seeds": ", ".join(str(s) for s in sorted({c["seed"] for c in cfgs}))}
        for section, key in keys + (fed if kinds.get(campaign) == "federated" else []):
            flat = [_flatten(f"{section}.{key}", c[section][key]) for c in cfgs]
            for name in dict.fromkeys(k for f in flat for k in f):
                col[name] = " / ".join(sorted({f.get(name, "null") for f in flat}))
        out[campaign] = col
    return pd.DataFrame(out)


# ---------------------------------------------------------------- LaTeX


def _fmt(mean, half=np.nan, digits: int = 2) -> str:
    if mean is None or not np.isfinite(mean):
        return "--"
    if half is None or not np.isfinite(half):
        return f"${mean:.{digits}f}$"
    return f"${mean:.{digits}f} \\pm {half:.{digits}f}$"


def _tex(text) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def latex_table(path, caption: str, label: str, header: list[str], rows: list[list[str]], note: str = "",
                align: str | None = None, wide: bool = True) -> None:
    """A booktabs table in the paper's style (footnotesize, arraystretch 1.2,
    full-width tabular*); `header` holds the header lines without `\\\\`
    (a `\\cmidrule` line is kept as is)."""
    env = "table*" if wide else "table"
    align = align or "l" + "c" * (len(rows[0]) - 1 if rows else 1)
    sep = 5 if len(align) <= 8 else 3 if len(align) <= 10 else 2  # narrower columns for wide tables
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", "\\renewcommand{\\arraystretch}{1.2}", f"\\setlength{{\\tabcolsep}}{{{sep}pt}}",
             f"\\begin{{tabular*}}{{\\{'textwidth' if wide else 'columnwidth'}}}{{@{{\\extracolsep{{\\fill}}}}{align}@{{}}}}",
             "\\toprule", *[h if h.startswith("\\cmidrule") else h + " \\\\" for h in header], "\\midrule",
             *[" & ".join(r) + " \\\\" for r in rows], "\\bottomrule", "\\end{tabular*}"]
    if note:
        lines += ["", "\\smallskip", f"{{\\footnotesize {note}\\par}}"]
    lines.append(f"\\end{{{env}}}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def latex_longtable(path, caption: str, label: str, header: list[str], rows: list[list[str]], note: str = "",
                    align: str | None = None, **_) -> None:
    """The same table as a full-width longtable that breaks across pages (for
    the supplementary document); the header repeats on each page."""
    align = align or "l" + "c" * (len(rows[0]) - 1 if rows else 1)
    sep = 5 if len(align) <= 8 else 3 if len(align) <= 10 else 2
    n = len(align)
    head = ["\\toprule", *[h if h.startswith("\\cmidrule") else h + " \\\\" for h in header], "\\midrule"]
    lines = ["{\\footnotesize", "\\renewcommand{\\arraystretch}{1.2}", f"\\setlength{{\\tabcolsep}}{{{sep}pt}}",
             f"\\begin{{longtable}}{{@{{\\extracolsep{{\\fill}}}}{align}@{{}}}}",
             f"\\caption{{{caption}}}\\label{{{label}}}\\\\", *head, "\\endfirsthead",
             f"\\multicolumn{{{n}}}{{@{{}}l}}{{\\itshape Table~\\thetable{{}} (continued)}}\\\\", *head, "\\endhead",
             "\\midrule", f"\\multicolumn{{{n}}}{{r@{{}}}}{{\\itshape continued on the next page}}\\\\", "\\endfoot",
             "\\bottomrule", "\\endlastfoot", *[" & ".join(r) + " \\\\" for r in rows], "\\end{longtable}"]
    if note:
        lines += [f"{note}\\par"]
    lines.append("}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def _supplementary_text(text: str, labels: set[str]) -> str:
    """References to tables outside the supplementary document as plain text;
    citations as author and venue."""
    import re

    def ref(m):
        return m[0] if m[1] in labels else MAIN_TEXT_REFS.get(m[1], "a table in the main text")

    def cite(m):
        if m[1] not in CITATIONS:
            raise ValueError(f"no plain-text citation for {m[1]}")
        return f" from {CITATIONS[m[1]]}"
    text = re.sub(r"Table~\\ref\{([^}]*)\}", ref, text)
    return re.sub(r"~?\\cite\{([^}]*)\}", cite, text)


def write_supplementary(out, names: list[str]) -> Path:
    """supplementary.tex: every full table as a longtable (from supplementary/),
    grouped by section; built with `make revision-supplementary`."""
    sections = [("Membership inference per attack and non-member set",
                 [n for n in names if n.startswith(("tab_central_", "tab_federated_"))])]
    sections += [(title, [n for n in group if n in names]) for title, group in SUPPLEMENT_SECTIONS]
    body = []
    for title, group in sections:
        if group:
            body += [f"\\section{{{title}}}", *[f"\\input{{supplementary/{n[:-4]}}}" for n in group], ""]
    path = Path(out) / "supplementary.tex"
    path.write_text(SUPPLEMENT_PREAMBLE + "\n".join(body) + "\\end{document}\n")
    return path


def _b(text: str) -> str:
    return f"\\textbf{{{text}}}"


def _scoped(h: pd.DataFrame, scope: tuple[str, str] | None) -> pd.DataFrame:
    """Rows of one (attack, non-member set), or all rows for scope None."""
    if scope is None or h.empty:
        return h
    return h[(h["attack"] == scope[0]) & (h["nonmember"] == scope[1])]


def _model(m: str) -> str:
    return _tex(MODEL_NAMES.get(m, m))


def _run(variant: str) -> str:
    return _tex(RUN_NAMES.get(variant, variant))


def _setting(r) -> str:
    return f"{_tex(r['sharding'])}, $K={int(r['num_clients'])}$" if "num_clients" in r and pd.notna(r["num_clients"]) \
        else _run(r["variant"])


def _ci_cell(s: pd.DataFrame, col: str, digits: int = 2) -> str:
    g = s[s["column"] == col]
    return _fmt(g["mean"].iloc[0], g["half"].iloc[0], digits) if len(g) else "--"


def _collapse_note(marked: dict, seen: dict, kinds: dict, flags: dict) -> str:
    """The note for the marked rows, grouped by method: "all N central models" when
    every run of that method in the table is marked, else the runs by name."""
    counts = set(marked.values())
    seeds = "{}/{} seeds".format(*next(iter(counts))) if len(counts) == 1 else "every seed"
    parts = []
    for method in _order([x for _, _, x in marked], METHOD_ORDER):
        runs = [(m, v) for m, v, x in marked if x == method]
        kind = {kinds.get(r) for r in runs}
        if len(runs) > 1 and set(runs) == seen.get(method, set()):
            what = {frozenset({"central"}): "central models", frozenset({"federated"}): "federated runs"}.get(
                frozenset(kind), "runs")
            desc = f"both {what}" if len(runs) == 2 else f"all {NUMBER_WORDS.get(len(runs), len(runs))} {what}"
        else:
            desc = ", ".join(_model(m) if kinds.get((m, v)) == "central" else f"{_model(m)} {_run(v)}"
                             for m, v in runs)
        if method == "client_ga_upload":
            released = any((m, v, "client_ga") in flags for m, v in runs)
            desc += " (intermediate model before recovery" + (
                "" if released else "; the released global model is not collapsed") + ")"
        parts.append(f"{NOTE_METHOD_NAMES.get(method, METHOD_NAMES.get(method, _tex(method)))}: {desc}")
    return (f"{COLLAPSED_MARK}Model collapsed: retain loss rose more than threefold in {seeds}. "
            + "; ".join(parts) + ".")


def _retrain_mu_note(comp: pd.DataFrame) -> str:
    """One line on retrain rows with model utility 0: the zero component(s), from the data."""
    r = comp[comp["method"] == "retrain"] if not comp.empty else comp
    if r.empty:
        return ""
    parts = []
    for (metric, split), g in r.groupby(["metric", "split"], sort=False):
        runs = "; ".join(f"{_model(x.model)}, {_run(x.variant)}: {x.zero}/{x.rows} seeds" for x in g.itertuples())
        parts.append(f"{MU_METRICS[metric]} on {MU_SPLITS[split]} ({runs})")
    return (" Retrain rows with model utility 0 are not collapsed: one of the nine harmonic-mean components is 0, "
            + "; ".join(parts) + ".")


def write_tables(a: dict, out) -> list[Path]:
    """Every LaTeX table: the `*_paper.tex` tables (two-column table*) and the
    full tables as longtables in supplementary/ with supplementary.tex; returns
    the table paths written."""
    out = Path(out)
    df = a["rows"]
    written = []
    flags = {(r["model"], r["variant"], r["method"]): (r["over"], r["rows"])
             for r in a.get("collapsed", pd.DataFrame()).to_dict("records")}
    kinds = {(r["model"], r["variant"]): r["kind"] for r in df[["model", "variant", "kind"]].drop_duplicates()
             .to_dict("records")} if len(df) else {}
    marked, seen = {}, {}  # collapsed rows, and every run per method, shown in the table being built

    def meth(method, *runs):
        """Method name, marked when its model collapsed in any of the (model, variant) runs."""
        hit = [(m, v, method) for m, v in runs if (m, v, method) in flags]
        marked.update({k: flags[k] for k in hit})
        seen.setdefault(method, set()).update(runs)
        return METHOD_NAMES.get(method, _tex(method)) + (COLLAPSED_MARK if hit else "")

    long = {}  # supplementary name -> the table's arguments

    def save(name, caption, label, header, rows, note="", **kw):
        """`*_paper.tex`: a two-column table* for the main text; every other (full)
        table: a longtable in supplementary/ only."""
        if marked:
            note = f"{note} {_collapse_note(marked, seen, kinds, flags)}".strip()
        marked.clear()
        seen.clear()
        if name.endswith("_paper.tex"):
            latex_table(out / name, caption, label, header, rows, note, **kw)
            written.append(out / name)
        else:
            long[name] = (caption, label, header, rows, note, kw)

    n_seeds = df.groupby(KEY)["seed"].nunique().max() if len(df) else 0
    ci_note = (f"Mean~$\\pm$~half-width of the 95\\% $t$-interval over seeds (up to {n_seeds}); "
               "a single value means one seed.")
    stat_cols = ["auc_pre", "auc_post", "gap_post", "tpr1_post", "adv_post", "adv_post_typical_basediff",
                 "adv_post_outlier_basediff", "adv_base", "auc_score_change", "auc_grad_align"]
    for kind in ("central", "federated"):
        d = df[df["kind"] == kind]
        if d.empty:
            continue
        for attack in _order(d["attack"].unique(), ATTACKS):
            for nm in _order(d["nonmember"].unique(), NONMEMBERS):
                g = d[(d["attack"] == attack) & (d["nonmember"] == nm)]
                s = summarise(g, KEY + ["method"], stat_cols + ["num_clients"])
                rows = []
                for (campaign, _, model, variant), u in g.groupby(KEY, sort=False):
                    for m in _order(u["method"].unique(), METHOD_ORDER):
                        sm = s[(s["campaign"] == campaign) & (s["model"] == model) & (s["variant"] == variant)
                               & (s["method"] == m)]
                        first = u[u["method"] == m].iloc[0]
                        cells = [_model(model)] + ([_setting(first)] if kind == "federated" else []) + [
                            meth(m, (model, variant)),
                            *[_ci_cell(sm, c) for c in ("auc_pre", "auc_post", "gap_post", "tpr1_post", "adv_post",
                                                         "adv_post_typical_basediff", "adv_post_outlier_basediff")]]
                        if kind == "federated":
                            cells += [_ci_cell(sm, "auc_score_change"), _ci_cell(sm, "auc_grad_align")]
                        rows.append(cells)
                lead = ["\\textbf{Model}"] + (["\\textbf{Setting}"] if kind == "federated" else [])
                n_lead = len(lead) + 1
                top = (f" {'& ' * (n_lead - 1)}& \\multicolumn{{2}}{{c}}{{\\textbf{{MIA AUC}}}} & & & "
                       f"\\multicolumn{{3}}{{c}}{{\\textbf{{Post-unlearning advantage}}}}"
                       + (" & \\multicolumn{2}{c}{\\textbf{Update AUC}}" if kind == "federated" else ""))
                rule = (f"\\cmidrule(lr){{{n_lead + 1}-{n_lead + 2}}}\\cmidrule(lr){{{n_lead + 5}-{n_lead + 7}}}"
                        + (f"\\cmidrule(lr){{{n_lead + 8}-{n_lead + 9}}}" if kind == "federated" else ""))
                head = " & ".join(lead + ["\\textbf{Method}" if kind == "central" else "\\textbf{Stage}",
                                          _b("Pre"), _b("Post"), _b("Gap"), _b("TPR@1\\%"), _b("All"),
                                          _b("Typical"), _b("Outlier")]
                                  + ([_b("Score chg."), _b("Grad. align.")] if kind == "federated" else []))
                what = "centralized" if kind == "central" else "federated"
                save(f"tab_{kind}_{attack}_{nm}.tex",
                     f"Membership inference before and after {what} unlearning ({ATTACK_NAMES[attack]}, "
                     f"{_tex(nm)} non-members). Gap is the post-unlearning AUC minus the retrained model's "
                     "(same seed); advantage is $|\\mathrm{AUC}-0.5|$; outliers by base-model difficulty, "
                     "stratum-matched.", f"tab:{kind}-{attack.replace('_', '-')}-{nm}", [top, rule, head],
                     rows, ci_note)
    verdict_note = (" Verdict: above retrain (CI lower bound $>0$, supports H1), below retrain (CI upper bound "
                    "$<0$, contradicts H1) or inconclusive.")
    for paper in (False, True):  # full tables for the appendix; paper tables: loss threshold, primary only
        sfx, scope = ("_paper", PAPER_SCOPE) if paper else ("", None)

        def where(full_label, paper=paper):
            return (" Loss-threshold attack, primary non-members; every attack and non-member set in "
                    f"Table~\\ref{{{full_label}}} (appendix)." if paper else "")
        lead = ("Model", "Run", "Method") + (() if paper else ("Attack", "Non-members"))

        def head(r, paper=paper):
            cells = [_model(r["model"]), _run(r["variant"]), meth(r["method"], (r["model"], r["variant"]))]
            return cells if paper else cells + [ATTACK_NAMES[r["attack"]], _tex(r["nonmember"])]
        h = _scoped(a["h1"], scope)
        if not h.empty:
            rows = [head(r) + [_fmt(r["mean"], r["half"], 3), r["verdict"] if isinstance(r["verdict"], str)
                               else "--"] for r in h.to_dict("records")]
            save(f"tab_h1{sfx}.tex", "H1: post-unlearning AUC minus the retrained model's AUC, paired by seed."
                 + where("tab:h1"), f"tab:h1{sfx.replace('_', '-')}",
                 [" & ".join(_b(x) for x in lead + ("Gap", "Verdict"))], rows, ci_note + verdict_note,
                 align="l" * len(lead) + "cc")
        h = _scoped(a["h1_two_sided"], scope)
        if not h.empty:
            rows = []
            for _, g in h.groupby(KEY + ["method", "attack", "nonmember"], sort=False):
                cell = {r["stratum"]: r for r in g.to_dict("records")}
                dist = [_fmt(cell[s]["mean"], cell[s]["half"], 3) if s in cell else "--"
                        for s in ("all", "typical_basediff", "outlier_basediff")]
                r = cell["all"]
                rows.append(head(r) + [*dist, f"{r['n_below_retrain']}/{r['n']}",
                                       _fmt(100 * r["privleak_mean"], 100 * r["privleak_half"], 1),
                                       _fmt(r["forget_quality_mean"], r["forget_quality_half"], 3)])
            save(f"tab_h1_two_sided{sfx}.tex", "Exploratory (post hoc): two-sided distance $|\\mathrm{AUC}-"
                 "\\mathrm{AUC}_{\\mathrm{retrain}}|$ to the retrained model of the same seed (federated: the "
                 "federated retrain), overall and per stratum, added after seeing the GPT-2 results; next to "
                 "MUSE's PrivLeak~\\cite{shi2024muse}, $(\\mathrm{AUC}-\\mathrm{AUC}_{\\mathrm{retrain}})/"
                 "\\mathrm{AUC}_{\\mathrm{retrain}}$, and TOFU forget quality. H1 "
                 f"(Table~\\ref{{tab:h1{sfx.replace('_', '-')}}}) is the pre-registered one-sided test."
                 + where("tab:h1-two-sided"), f"tab:h1-two-sided{sfx.replace('_', '-')}",
                 [" & ".join(_b(x) for x in lead + ("All", "Typical", "Outlier", "Below", "PrivLeak (\\%)",
                                                    "Forget quality"))],
                 rows, ci_note + " Strata by base-model difficulty, stratum-matched. Below: seeds whose AUC is "
                 "under the retrained model's. PrivLeak: per seed, over all records, in percent (MUSE's "
                 "convention; near 0 for good unlearning).", align="l" * len(lead) + "c" * 6)
    if not a["h2"].empty:
        models = [m for m in GPT2_PARAMS if f"{m}__gap_post" in a["h2"]]
        rows = []
        for r in a["h2"].to_dict("records"):
            slope = _fmt(r["slope"], r["slope_half"], 3) if pd.notna(r.get("slope")) else "--"
            rows.append([meth(r["method"], *[(m, "central") for m in models]), ATTACK_NAMES[r["attack"]],
                         _tex(r["nonmember"]),
                         *[_fmt(r[f"{m}__gap_post"], r[f"{m}__gap_post_half"]) for m in models],
                         *[_fmt(r[f"{m}__rel_drop"], r[f"{m}__rel_drop_half"]) for m in models], slope])
        save("tab_h2.tex", "H2: retrain-relative gap and relative advantage drop "
             "$(\\mathrm{adv}_{pre}-\\mathrm{adv}_{post})/\\mathrm{adv}_{pre}$ across the GPT-2 series; "
             "slope of the gap per decade of parameters, fitted within each seed (mean and t-interval over "
             "seeds).", "tab:h2",
             [" & ".join([_b("Method"), _b("Attack"), _b("Non-members")]
                         + [_b("Gap " + _model(m)) for m in models] + [_b("Drop " + _model(m)) for m in models]
                         + [_b("Slope")])], rows, ci_note, align="lll" + "c" * (2 * len(models) + 1))
    for name, roles, title in (("tab_h3.tex", ("primary", "secondary"), "H3"),
                               ("tab_h3_ablation.tex", ("ablation",), "H3 ablation (appendix)")):
        h = a["h3"][a["h3"]["role"].isin(roles)] if not a["h3"].empty else a["h3"]
        if h.empty:
            continue
        rows = [[_model(r.model), meth(r.method, (r.model, r.variant)), ATTACK_NAMES[r.attack], _tex(r.nonmember),
                 _tex(r.definition), _fmt(r.gap_outlier), _fmt(r.gap_typical), _fmt(r.mean, r.half, 3),
                 r.verdict if isinstance(r.verdict, str) else "--", _fmt(r.adv_outlier), _fmt(r.adv_typical)]
                for r in h.itertuples()]
        save(name, f"{title}: retrain-relative gap of outliers minus typical records, paired by seed.",
             "tab:" + name[4:-4].replace("_", "-"),
             [" & ".join(_b(x) for x in ("Model", "Method", "Attack", "Non-members", "Definition", "Gap out.",
                                         "Gap typ.", "Difference", "Verdict", "Adv. out.", "Adv. typ."))],
             rows, ci_note + " Verdict: outliers higher (CI lower bound $>0$), typical higher (CI upper bound "
             "$<0$) or inconclusive.", align="lllll" + "c" * 6)
    lf = a["low_fpr"]
    if not lf.empty:
        lf = lf[lf["stage"].str.startswith("post_") & (lf["attack"] != "ref_calibrated")]
        rows = [[_model(r["model"]), _run(r["variant"]), meth(r["stage"][5:], (r["model"], r["variant"])),
                 ATTACK_NAMES[r["attack"]], _tex(r["nonmember"]), str(r["n_seeds"]), str(r["n_nonmember"]),
                 _fmt(r["tpr_at_0.1"], digits=3), _fmt(r["tpr_at_0.1_seed_mean"], r["tpr_at_0.1_seed_half"], 3),
                 _fmt(r.get("tpr_at_1_outlier_basediff"), digits=3),
                 _fmt(r.get("tpr_at_1_typical_basediff"), digits=3)] for r in lf.to_dict("records")]
        sizes = sorted({int(n) for n in lf["n_nonmember_per_seed"] if pd.notna(n)})
        count = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]}--{sizes[-1]}" if sizes else "?"
        # below 1,000 non-members a seed's 0.1% FPR allows no false positive
        zero_fp = bool(sizes) and sizes[-1] < 1000
        per_seed = (f"TPR at zero false positives ({count} non-members)" if zero_fp
                    else f"TPR@0.1\\% per seed ({count} non-members)")
        save("tab_low_fpr.tex", "TPR at 0.1\\% FPR with scores pooled over seeds, its per-seed mean, and TPR at "
             "1\\% FPR within the base-difficulty strata (outlier forget vs outlier non-members), pooled the same "
             "way.",
             "tab:low-fpr", [" & ".join(_b(x) for x in ("Model", "Run", "Method", "Attack", "Non-members", "Seeds",
                                                        "Negatives", "\\shortstack{TPR@0.1\\%\\\\pooled}",
                                                        "\\shortstack{" + per_seed.replace(" (", "\\\\(").replace(
                                                            " false", "\\\\false") + "}",
                                                        "\\shortstack{TPR@1\\%\\\\outlier}",
                                                        "\\shortstack{TPR@1\\%\\\\typical}"))],
             rows, "Pooling puts runs with different models on one threshold; with 2,000 negatives, 0.1\\% FPR "
             "corresponds to 2 false positives. " + per_seed + ": the per-seed TPR@0.1\\%"
             + (", which allows no false positive at that size," if zero_fp else "")
             + " as mean and t-interval over seeds (no pooling).", align="lllll" + "c" * 6)
    if not a["update"].empty:
        u = a["update"]
        rows = [[_model(r["model"]), _run(r["variant"]), meth(r["method"], (r["model"], r["variant"])),
                 VIEW_NAMES.get(r["view"], _tex(r["view"])),
                 ATTACK_NAMES[r["attack"]], _tex(r["nonmember"]),
                 _fmt(r["auc_score_change"], r["auc_score_change_half"]),
                 _fmt(r["gap_auc_score_change"], r["gap_auc_score_change_half"]),
                 _fmt(r["auc_grad_align"], r["auc_grad_align_half"]),
                 _fmt(r["gap_auc_grad_align"], r["gap_auc_grad_align_half"])] for r in u.to_dict("records")]
        save("tab_update.tex", "H4/H5: update-signal attacks. Score change (black-box, pre $-$ post) and gradient "
             "alignment with $\\Delta\\theta$ (white-box), and their gaps to retrain (same seed).", "tab:update",
             [" & & & & & & \\multicolumn{2}{c}{\\textbf{Score change}} & \\multicolumn{2}{c}{\\textbf{Grad. alignment}}",
              "\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
              " & ".join(_b(x) for x in ("Model", "Run", "Method", "View", "Attack", "NM", "AUC", "Gap", "AUC",
                                         "Gap"))], rows,
             ci_note + " NM: non-member set. View: who observes the change from $\\theta_{pre}$ (release: the "
             "two model versions; $^{*}$the server, and all clients at the first recovery broadcast).",
             align="llllll" + "c" * 4)
    if not a["control"].empty:
        rows = [[_tex(r["kind"]), _model(r["model"]), _tex(r["nonmember"]), str(r["n_draws"]), _fmt(r["mean"], digits=3),
                 _fmt(r["sd"], digits=3), f"${r['min']:.3f}$--${r['max']:.3f}$", _fmt(r["null_sd"], digits=3),
                 _fmt(r["share_outside_2_null_sd"]), _fmt(r["auc_grad_align_mean"], digits=3)]
                for r in a["control"].to_dict("records")]
        save("tab_control.tex", "Random-direction control of the white-box attack: AUC over all (method, seed) "
             "draws, against the SD of the AUC for independent records.", "tab:control",
             [" & ".join(_b(x) for x in ("Runs", "Model", "Non-members", "Draws", "Mean", "SD", "Range", "Null SD",
                                         "Outside $2\\sigma$", "Grad. align."))], rows, align="lll" + "c" * 7)
    if not a["costs"].empty:
        rows = []
        for r in a["costs"].to_dict("records"):
            rows.append([_model(r["model"]), _tex(r["sharding"]), str(r["num_clients"]),
                         _fmt(r["time_s_method__client_ga"] / 60, r["time_s_method__client_ga_half"] / 60, 1),
                         _fmt(r["time_s_method__retrain"] / 60, r["time_s_method__retrain_half"] / 60, 1),
                         _fmt(r["time_s_method__ratio"], r["time_s_method__ratio_half"]),
                         _fmt(r.get("peak_mem_mib_method__ratio"), r.get("peak_mem_mib_method__ratio_half")),
                         _fmt(r.get("peak_mem_reserved_mib_method__ratio"),
                              r.get("peak_mem_reserved_mib_method__ratio_half")),
                         _fmt(r["comm_bytes_method__client_ga"] / 2**30, digits=2),
                         _fmt(r["comm_bytes_method__retrain"] / 2**30, digits=2),
                         _fmt(r["comm_bytes_method__ratio"])])
        save("tab_costs.tex", "H6: cost of client-level gradient ascent with recovery against federated "
             "retraining, per setting (ratios paired by seed).", "tab:costs",
             [" & & & \\multicolumn{3}{c}{\\textbf{Wall time (min)}} & \\multicolumn{2}{c}{\\textbf{Peak memory}} & "
              "\\multicolumn{3}{c}{\\textbf{Communication (GiB)}}",
              "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}",
              " & ".join(_b(x) for x in ("Model", "Sharding", "$K$", "GA", "Retrain", "Ratio", "Alloc.", "Reserved",
                                         "GA", "Retrain", "Ratio"))], rows,
             ci_note + " GA: client gradient ascent plus recovery rounds. Peak memory as the ratio GA / retrain "
             "(allocated and reserved). Time and peak GPU memory are measured; communication is analytic "
             "(fp32 parameters).",
             align="llc" + "c" * 8)
    if not a["tofu"].empty:
        s = a["tofu"]
        rows = []
        for keys, g in s.groupby(KEY + ["method"], sort=False):
            rows.append([_model(keys[2]), _run(keys[3]), meth(keys[4], (keys[2], keys[3])),
                         _ci_cell(g, "forget_quality_post", 3), _ci_cell(g, "model_utility_post", 3),
                         _ci_cell(g, "model_utility_post_nonzero", 3)])
        zero = a["mu_zero"]
        z = int(zero[zero["column"] == "model_utility_post"]["zero"].sum()) if not zero.empty else 0
        n = int(zero[zero["column"] == "model_utility_post"]["rows"].sum()) if not zero.empty else 0
        save("tab_tofu.tex", "TOFU forget quality (KS test against the retrained model of the same seed) and model "
             "utility.", "tab:tofu", [" & ".join(_b(x) for x in ("Model", "Run", "Method", "Forget quality",
                                                                    "Model utility", "MU excl. 0"))], rows,
             ci_note + f" Rows with model utility 0: {z} of {n}. MU excl.\\ 0: model utility without those rows "
             "(sensitivity check)." + _retrain_mu_note(a.get("mu_components", pd.DataFrame())), align="lllccc")
    if not a["hyper"].empty:
        h = a["hyper"]
        rows = [[_tex(k)] + [_tex(v) if isinstance(v, str) else "--" for v in h.loc[k]] for k in h.index]
        save("tab_hyperparameters.tex", "Settings of each campaign, from the resolved configs.",
             "tab:hyperparameters", [" & ".join([_b("Setting")] + [_b(_tex(c)) for c in h.columns])], rows,
             align="l" + "l" * len(h.columns))
    if not a["slowdown"].empty:
        rows = [[_model(r["model"]), _run(r["variant"]), meth(r["method"], (r["model"], r["variant"])),
                 _tex(r["phase"]),
                 f"${r['alone_s']:.1f}$", f"${r['campaign_s']:.1f}$", _fmt(r["slowdown"]),
                 "yes" if r["same_gpu"] else "no"] for r in a["slowdown"].to_dict("records")]
        save("tab_pilot_slowdown.tex", "Phase times of seed 0 run alone (pilot, LoRA pilot) and with a second "
             "process on the other GPU (campaign).", "tab:pilot-slowdown",
             [" & ".join(_b(x) for x in ("Model", "Run", "Method", "Phase", "Alone (s)", "Campaign (s)", "Ratio",
                                         "Same GPU"))], rows, align="llll" + "c" * 4)
    labels = {args[1] for args in long.values()}
    for name, (caption, label, header, rows, note, kw) in long.items():
        latex_longtable(out / "supplementary" / name, _supplementary_text(caption, labels), label, header, rows,
                        _supplementary_text(note, labels), **kw)
        written.append(out / "supplementary" / name)
    write_supplementary(out, list(long))
    return written


# ---------------------------------------------------------------- figures


def _save_fig(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _points(ax, s: pd.DataFrame, x: str, hue: str, x_order, hue_order, names=None, markers="osD^vP"):
    """Means with t-interval error bars, dodged by `hue`."""
    import seaborn as sns

    colours = sns.color_palette("deep", len(hue_order))
    width = 0.6 / max(1, len(hue_order))
    for i, h in enumerate(hue_order):
        g = s[s[hue] == h].set_index(x).reindex(x_order)
        pos = np.arange(len(x_order)) + (i - (len(hue_order) - 1) / 2) * width
        ax.errorbar(pos, g["mean"], yerr=g["half"].fillna(0), fmt=markers[i % len(markers)], color=colours[i],
                    capsize=3, lw=1.2, label=(names or {}).get(h, h))
    ax.set_xticks(np.arange(len(x_order)))
    ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in x_order] if x == "model" else x_order)


def write_figures(a: dict, out) -> list[Path]:
    """The revision figures (PDF + 300-dpi PNG) in the style of make_paper_figures.py."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = a["rows"]
    written = []
    sns.set_theme(style="ticks")
    central = df[(df["kind"] == "central") & (df["nonmember"] == "primary")]
    if not central.empty:
        models = _order(central["model"].unique(), MODEL_ORDER)
        methods = _order(central["method"].unique(), METHOD_ORDER)
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=False)
        for ax, (col, label) in zip(axes, (("auc_post", "post-unlearning MIA AUC"),
                                           ("gap_post", "gap to retrain (AUC)")), strict=True):
            s = summarise(central[central["attack"] == "loss_threshold"], ["model", "method"], [col])
            _points(ax, s, "model", "method", models, methods, METHOD_NAMES)
            ax.axhline(0.5 if col == "auc_post" else 0.0, color="gray", ls=":", lw=1)
            ax.set_xlabel(r"model (increasing size $\rightarrow$)")
            ax.set_ylabel(label)
        axes[0].legend(frameon=False, fontsize=8)
        sns.despine(fig)
        _save_fig(fig, out / "fig_rev_central.pdf")
        written.append(out / "fig_rev_central.pdf")

        if not a["h3"].empty:
            d = central[(central["attack"] == "loss_threshold") & (central["method"] != "retrain")]
            unl = _order(d["method"].unique(), METHOD_ORDER)
            fig, axes = plt.subplots(1, len(unl), figsize=(3.0 * len(unl), 3.6), sharey=True, squeeze=False)
            for ax, m in zip(axes[0], unl, strict=True):
                dm = d[d["method"] == m]
                long = pd.concat([summarise(dm, ["model"], [f"gap_post_{s}_basediff"]).assign(stratum=s)
                                  for s in ("typical", "outlier")])
                _points(ax, long, "model", "stratum", models, ["typical", "outlier"])
                ax.axhline(0, color="gray", ls=":", lw=1)
                ax.set_title(METHOD_NAMES.get(m, m), fontsize=10)
                ax.set_xlabel("model")
            axes[0][0].set_ylabel("gap to retrain (AUC)")
            axes[0][0].legend(title="stratum (base difficulty)", frameon=False, fontsize=8)
            sns.despine(fig)
            _save_fig(fig, out / "fig_rev_strata.pdf")
            written.append(out / "fig_rev_strata.pdf")

    u = df[(df["nonmember"] == "primary") & (df["attack"] == "loss_threshold")].copy()
    for kind, name in (("central", "fig_rev_update.pdf"), ("federated", "fig_rev_fed_update.pdf")):
        d = u[u["kind"] == kind]
        if d.empty or d["auc_grad_align"].isna().all():
            continue
        # one x position per run (model and, for federated runs, setting); intervals over seeds only
        d = d.assign(run=[MODEL_NAMES.get(m, m) + ("" if kind == "central" else f"\n{v}")
                          for m, v in zip(d["model"], d["variant"], strict=True)])
        runs = list(dict.fromkeys(d.sort_values("model", key=lambda c: c.map(
            {m: i for i, m in enumerate(MODEL_ORDER)})).get("run")))
        methods = _order(d["method"].unique(), METHOD_ORDER)
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
        for ax, (col, label) in zip(axes, (("auc_score_change", "score change (black-box)"),
                                           ("auc_grad_align", "gradient alignment (white-box)")), strict=True):
            _points(ax, summarise(d, ["run", "method"], [col]), "run", "method", runs, methods, METHOD_NAMES)
            ax.axhline(0.5, color="gray", ls=":", lw=1)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("model" + ("" if kind == "central" else " and setting"))
        axes[0].set_ylabel("AUC, forget vs non-members")
        axes[0].legend(frameon=False, fontsize=8)
        sns.despine(fig)
        _save_fig(fig, out / name)
        written.append(out / name)

    if not a["control"].empty:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        sns.stripplot(data=df[df["attack"] == "loss_threshold"], x="model", y="auc_grad_align_random",
                      hue="nonmember", order=_order(df["model"].unique(), MODEL_ORDER), dodge=True, alpha=0.7,
                      palette="muted", ax=ax)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xticks(ax.get_xticks(), [MODEL_NAMES.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
        ax.set_xlabel("model")
        ax.set_ylabel("random-direction AUC (all draws)")
        ax.legend(title="non-members", frameon=False, fontsize=8)
        sns.despine(fig)
        _save_fig(fig, out / "fig_rev_control.pdf")
        written.append(out / "fig_rev_control.pdf")

    c = a["costs"]
    if not c.empty and len(c):
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        for col, label, marker in (("time_s_method", "wall time", "o"), ("comm_bytes_method", "communication", "s"),
                                   ("peak_mem_mib_method", "peak memory", "D")):
            if f"{col}__ratio" not in c:
                continue
            for (model, sh), g in c.groupby(["model", "sharding"], sort=False):
                g = g.sort_values("num_clients")
                if g[f"{col}__ratio"].isna().all():  # e.g. peak memory on CPU
                    continue
                ax.errorbar(g["num_clients"], g[f"{col}__ratio"], yerr=g[f"{col}__ratio_half"].fillna(0),
                            marker=marker, capsize=3, label=f"{label}, {MODEL_NAMES.get(model, model)} ({sh})")
        ax.set_xticks(sorted(c["num_clients"].unique()))
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel("clients $K$")
        ax.set_ylabel("client GA + recovery / federated retrain")
        ax.legend(frameon=False, fontsize=8)
        sns.despine(fig)
        _save_fig(fig, out / "fig_rev_costs.pdf")
        written.append(out / "fig_rev_costs.pdf")
    return written


# ---------------------------------------------------------------- driver


def analyse(results, figures_dir, tables_dir) -> dict:
    """Every analysis; writes the CSV summaries, LaTeX tables and figures."""
    a = load(results)
    df = a["rows"]
    if df.empty:
        raise SystemExit(f"no campaign CSVs in {results}")
    kinds = df.drop_duplicates("campaign").set_index("campaign")["kind"].to_dict()
    a.update(h1=h1(df), h1_two_sided=h1_two_sided(df), h2=h2(df), h3=h3(df),
             low_fpr=pooled_low_fpr(df, a["scores"]),
             update=update_attacks(df), control=random_control(df, a["scores"]), costs=costs(df),
             central_costs=central_costs(df), slowdown=pilot_slowdown(df, a["pilot"]),
             hyper=hyperparameters(a["configs"], kinds))
    a["tofu"], a["mu_zero"] = tofu(df)
    a["collapsed"] = collapsed(df)
    a["mu_components"] = mu_zero_components(df)
    a["summary"] = summarise(df, KEY + ["method", "attack", "nonmember"],
                             ["auc_base", "adv_base", "auc_pre", "auc_post", "adv_pre", "adv_post", "gap_pre",
                              "gap_post", "tpr1_base", "tpr1_pre", "tpr1_post", "auc_score_change", "auc_grad_align",
                              "auc_grad_align_random", "forget_quality_post", "model_utility_post"])
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name in ("summary", "h1", "h1_two_sided", "h2", "h3", "low_fpr", "update", "control", "costs",
                 "central_costs", "slowdown", "tofu", "mu_zero", "mu_components", "collapsed"):
        a[name].to_csv(tables_dir / f"{name}.csv", index=False)
    a["hyper"].to_csv(tables_dir / "hyperparameters.csv")
    a["tables"] = write_tables(a, tables_dir)
    a["figures"] = write_figures(a, figures_dir)
    return a
