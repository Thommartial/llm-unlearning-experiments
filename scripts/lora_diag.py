"""LoRA training diagnostic (a few hundred steps per arm, one GPU): 05_lora_diag, 06_1b_diag.

    python scripts/lora_diag.py --arms amp fp32_adapters amp_no_ckpt amp_eager amp_fp32_base   # 05
    python scripts/lora_diag.py --arms tinyllama_fp32 pythia_amp pythia_fp32 --eval-check      # 06
    python scripts/lora_diag.py --config configs/revision/smoke_cpu.yaml --model tiny_llama --device cpu --steps 5

Fine-tunes fresh adapters on the members (forget + retain) at the LoRA
config's `finetune_lr` and `train_batch_size`, with the step code of
`unlearn.train_lm` (`unlearn._update`), and logs every step: loss, whether
it was skipped (non-finite loss, GradScaler, or a non-finite gradient without a
scaler), the scaler's scale, the
gradient norm before clipping, the change of the adapter weights from their
start, the norm of the lora_B weights (zero at the start) and the seconds
the step took. The campaign recipe is `amp`: fp16 base weights, fp32
adapters, fp16 autocast + GradScaler, gradient checkpointing.
  amp             the campaign recipe (the config's model)
  fp32_adapters   no autocast and no GradScaler (fp16 base, fp32 adapters; no bf16)
  amp_no_ckpt     gradient checkpointing off
  amp_eager       eager attention instead of SDPA
  amp_fp32_base   fp32 base weights (autocast + GradScaler as in the full fine-tunes)
  tinyllama_fp32  TinyLlama, fp32 base, no autocast, no GradScaler
  pythia_amp      Pythia-1B, the campaign recipe
  pythia_fp32     Pythia-1B, fp32 base, no autocast, no GradScaler
The LoRA target modules come from the config's `lora.target_modules` for the
model's type (Pythia: query_key_value and dense). `--model` replaces every
arm's model (e.g. tiny_llama or tiny_neox for a CPU check).
Per arm: the retain loss on the first `--eval-records` retain records before
and after, the step counters (`unlearn.StepStats`), the median seconds per
step after 5 warm-up steps, the mean loss per 25 steps, whether every adapter
got a gradient at the first step (every lora_B a nonzero one), and the
parameters with non-finite gradients at the first such step.
`--eval-check` first scores each arm's model with fresh adapters (the base
model) on the same records under the settings in EVAL_CELLS (weights,
compute dtype, attention kernel, batch size, train mode), to locate a loss
that depends on the eval path. Per-step rows go to <out>/lora_diag_steps.csv,
the arm summary to <out>/lora_diag.json, the eval check to <out>/lora_diag_eval.json.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TINYLLAMA = "TinyLlama/TinyLlama_v1.1"
PYTHIA = "EleutherAI/pythia-1b"
FP32 = {"precision": "fp32", "lora": {"base_dtype": "float32"}}
ARMS = {
    "amp": {},
    "fp32_adapters": {"precision": "fp32"},
    "amp_no_ckpt": {"lora": {"gradient_checkpointing": False}},
    "amp_eager": {"attn": "eager"},
    "amp_fp32_base": {"lora": {"base_dtype": "float32"}},
    "tinyllama_fp32": {"model": TINYLLAMA, **FP32},
    "pythia_amp": {"model": PYTHIA},
    "pythia_fp32": {"model": PYTHIA, **FP32},
}
# name, base weights, fp16 autocast, attention, eval batch size (None: the config's), train mode
EVAL_CELLS = [
    ("fp32", "float32", False, "sdpa", None, False),
    ("fp32_math", "float32", False, "math", None, False),
    ("fp32_eager", "float32", False, "eager", None, False),
    ("fp32_batch1", "float32", False, "sdpa", 1, False),
    ("fp32_train_mode", "float32", False, "sdpa", None, True),
    ("fp32_autocast", "float32", True, "sdpa", None, False),
    ("fp16", "float16", False, "sdpa", None, False),
    ("fp16_eager", "float16", False, "eager", None, False),
]
WARMUP = 5
WINDOW = 25


def _arm_cfg(cfg, arm: str, model: str | None = None):
    a = ARMS[arm]
    lora = {**(cfg.model.lora or {}), **a.get("lora", {})}
    return replace(cfg, model=replace(cfg.model, name=model or a.get("model", cfg.model.name),
                                      precision=a.get("precision", cfg.model.precision), lora=lora))


def _sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _free(device: str) -> None:
    import gc

    import torch

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def eval_check(cfg, eval_set, device: str) -> list[dict]:
    """Retain loss of the base model (fresh adapters) under each EVAL_CELLS setting."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from unlearning import evaluate, unlearn
    from unlearning.seed import set_seed

    rows, model, loaded = [], None, None
    for name, weights, amp, attn, batch, train in EVAL_CELLS:
        row = {"model": cfg.model.name, "cell": name, "weights": weights, "autocast": amp, "attn": attn,
               "batch": batch or cfg.unlearn.eval_batch_size, "train_mode": train, "retain_loss": None, "s": None}
        if amp and not device.startswith("cuda"):
            rows.append(row)  # fp16 autocast runs on CUDA only
            continue
        if loaded != weights:
            model = None
            _free(device)
            c = replace(cfg, model=replace(cfg.model, lora={**(cfg.model.lora or {}), "base_dtype": weights}))
            set_seed(cfg.seed)  # the same weights as the arms for the untrained tiny models
            model, _ = unlearn.build_model_and_tokenizer(c)
            model.to(device)
            loaded = weights
        model.base_model.model.set_attn_implementation("eager" if attn == "eager" else "sdpa")
        model.train(train)
        c = replace(cfg, unlearn=replace(cfg.unlearn, eval_batch_size=row["batch"]))
        kernel = sdpa_kernel(SDPBackend.MATH) if attn == "math" else contextlib.nullcontext()
        cast = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
        start = time.perf_counter()
        with kernel, cast:
            row["retain_loss"] = evaluate._token_mean_loss(model, eval_set, c, device)
        row["s"] = time.perf_counter() - start
        print(f"[eval] {cfg.model.name} {name:16s} retain loss {row['retain_loss']:.4f}  ({row['s']:.1f} s)",
              flush=True)
        rows.append(row)
    model = None
    _free(device)
    return rows


def run_arm(cfg, arm: str, splits, device: str, steps: int, eval_set) -> tuple[dict, list[dict]]:
    import torch
    from torch.utils.data import ConcatDataset

    from unlearning import evaluate, unlearn
    from unlearning.seed import set_seed

    c = cfg
    set_seed(c.seed)
    model, _ = unlearn.build_model_and_tokenizer(c)
    if ARMS[arm].get("attn"):
        model.base_model.model.set_attn_implementation(ARMS[arm]["attn"])
    model.to(device)
    lora = {n: p for n, p in model.named_parameters() if p.requires_grad}
    start = {n: p.detach().clone() for n, p in lora.items()}
    model_type = model.base_model.model.config.model_type
    info = {"arm": arm, "model": c.model.name, "precision": c.model.precision, "amp": unlearn._use_amp(c, device),
            "lora": unlearn.lora_settings(c), "targets": unlearn.lora_targets(unlearn.lora_settings(c), model_type),
            "attn": model.config._attn_implementation, "trainable_tensors": len(lora),
            "trainable_dtypes": sorted({str(p.dtype) for p in lora.values()}),
            "base_dtypes": sorted({str(p.dtype) for n, p in model.named_parameters() if not p.requires_grad}),
            "retain_loss_before": evaluate.mean_loss(model, eval_set, c, device)}
    lr = c.unlearn.finetune_lr or c.unlearn.learning_rate
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=unlearn._use_amp(c, device))
    stats = unlearn.StepStats()
    members = ConcatDataset([splits.forget, splits.retain])
    batch_size = c.unlearn.train_batch_size or c.unlearn.batch_size
    rows, step = [], 0
    while step < steps:
        for batch in unlearn._loader(members, c, batch_size=batch_size):
            _sync(device)
            t0 = time.perf_counter()
            opt.zero_grad()
            stats.batches += 1
            step += 1
            with unlearn._autocast(c, device):
                loss = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                             labels=batch["labels"].to(device)).loss
            skipped_scaler, skipped_grad = stats.skipped_scaler, stats.skipped_grad
            step_applied = bool(torch.isfinite(loss))
            if not step_applied:
                stats.skipped_loss += 1
            else:
                scaler.scale(loss).backward()
                if step == 1:  # lora_A gets a zero gradient at the first step: lora_B starts at zero
                    info["adapters_with_grad_step1"] = sum(p.grad is not None for p in lora.values())
                    info["lora_B_nonzero_grad_step1"] = sum(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                                            for n, p in lora.items() if "lora_B" in n)
                bad = [n for n, p in lora.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
                if bad and "nonfinite_grad_params" not in info:
                    info["nonfinite_grad_params"] = {"step": step, "count": len(bad), "first": bad[:5]}
                unlearn._update(model, opt, scaler, stats)
            _sync(device)
            step_s = time.perf_counter() - t0
            with torch.no_grad():
                delta = sum(float((p.detach().float() - start[n].float()).pow(2).sum()) for n, p in lora.items()) ** 0.5
                b_norm = sum(float(p.detach().float().pow(2).sum()) for n, p in lora.items() if "lora_B" in n) ** 0.5
            rows.append({"arm": arm, "model": c.model.name, "step": step, "loss": float(loss.detach()),
                         "skipped_loss": int(not step_applied),
                         "skipped_scaler": stats.skipped_scaler - skipped_scaler,
                         "skipped_grad": stats.skipped_grad - skipped_grad, "scale": stats.scale_last,
                         "grad_norm": stats.grad_norm_last if step_applied else None,
                         "adapter_delta": delta, "lora_B_norm": b_norm, "step_s": step_s})
            if step % WINDOW == 0 or step == 1:
                r = rows[-1]
                print(f"[diag] {arm} step {step:4d} loss {r['loss']:.4f} scaler skips {stats.skipped_scaler} "
                      f"scale {r['scale']} adapter change {delta:.4g} |B| {b_norm:.4g} {step_s:.2f} s", flush=True)
            if step >= steps:
                break
    stats.log(f"lora_diag {arm}")
    info["steps"] = stats.summary()
    info["retain_loss_after"] = evaluate.mean_loss(model, eval_set, c, device)
    info["adapter_delta"] = rows[-1]["adapter_delta"] if rows else 0.0
    info["lora_B_norm"] = rows[-1]["lora_B_norm"] if rows else 0.0
    timed = [r["step_s"] for r in rows[WARMUP:]] or [r["step_s"] for r in rows]
    info["sec_per_step"] = statistics.median(timed) if timed else None
    info["loss_by_window"] = [_finite_mean([r["loss"] for r in rows[i:i + WINDOW]]) for i in range(0, len(rows), WINDOW)]
    model = opt = None
    _free(device)
    return info, rows


def _finite_mean(values: list[float]) -> float | None:
    import math

    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/revision/lora.yaml")
    ap.add_argument("--model", help="replace every arm's model (e.g. tiny_llama or tiny_neox for a CPU check)")
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS)[:5])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-records", type=int, default=256)
    ap.add_argument("--eval-check", action="store_true", help="score each arm's base model under EVAL_CELLS first")
    ap.add_argument("--device", help="override the config's device (cpu, cuda)")
    ap.add_argument("--out", default="results/revision/lora_diag")
    args = ap.parse_args(argv)

    from unlearning import data, unlearn
    from unlearning.config import load_config

    cfg = load_config(args.config)
    if cfg.model.lora is None:
        cfg.model = replace(cfg.model, lora={})
    if args.device:
        cfg.device = args.device
    device = unlearn.resolve_device(cfg)
    arms = {arm: _arm_cfg(cfg, arm, args.model) for arm in args.arms}
    loaded = {}  # model name -> (splits, eval records)

    def splits_of(name: str):
        if name not in loaded:
            tok = None if name in ("tiny", "tiny_llama", "tiny_neox") else _tokenizer(name)
            splits = data.load_splits(replace(cfg, model=replace(cfg.model, name=name)), tok)
            eval_set = splits.retain.head(args.eval_records) if hasattr(splits.retain, "head") else splits.retain
            loaded[name] = splits, eval_set
        return loaded[name]

    print(f"[diag] on {device}: lr {cfg.unlearn.finetune_lr}, batch {cfg.unlearn.train_batch_size}, "
          f"arms {[(a, c.model.name) for a, c in arms.items()]}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.eval_check:
        checks = []
        for name in dict.fromkeys(c.model.name for c in arms.values()):
            checks += eval_check(replace(cfg, model=replace(cfg.model, name=name)), splits_of(name)[1], device)
            (out / "lora_diag_eval.json").write_text(json.dumps(checks, indent=2))
        print("\n[eval] model                     " + "  ".join(f"{c[0]:>15s}" for c in EVAL_CELLS))
        for name in dict.fromkeys(r["model"] for r in checks):
            cells = {r["cell"]: r["retain_loss"] for r in checks if r["model"] == name}
            print(f"[eval] {name:25s} " + "  ".join("              -" if cells[c[0]] is None
                                                     else f"{cells[c[0]]:15.4f}" for c in EVAL_CELLS))
        print(flush=True)
    summary, all_rows = [], []
    for arm, c in arms.items():
        splits, eval_set = splits_of(c.model.name)
        info, rows = run_arm(c, arm, splits, device, args.steps, eval_set)
        summary.append(info)
        all_rows += rows
        (out / "lora_diag.json").write_text(json.dumps(summary, indent=2, default=str))
        with open(out / "lora_diag_steps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    print("\n[diag] arm             model                     retain loss before -> after  applied/batches  "
          "scaler / grad skips  min scale  adapter change  s/step")
    for s in summary:
        st = s["steps"]
        print(f"[diag] {s['arm']:15s} {s['model']:25s} {s['retain_loss_before']:.4f} -> {s['retain_loss_after']:.4f}"
              f"            {st['applied']}/{st['batches']}          {st['skipped_scaler']} / {st['skipped_grad']}"
              f"            {st['scale_min']}     {s['adapter_delta']:.4g}     {s['sec_per_step']:.3f}")
    print(f"\n[diag] mean training loss per {WINDOW} steps")
    for s in summary:
        print(f"[diag] {s['arm']:15s} " + " ".join("   nan" if v is None else f"{v:.4f}" for v in s["loss_by_window"]))
    print(f"[diag] -> {out / 'lora_diag.json'}, {out / 'lora_diag_steps.csv'}"
          + (f", {out / 'lora_diag_eval.json'}" if args.eval_check else ""))
    return 0


def _tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


if __name__ == "__main__":
    sys.exit(main())
