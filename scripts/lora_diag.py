"""LoRA training diagnostic (TinyLlama, a few hundred steps per arm, one GPU).

    python scripts/lora_diag.py                       # every arm, 200 steps each
    python scripts/lora_diag.py --arms amp fp32_adapters --steps 100
    python scripts/lora_diag.py --config configs/revision/smoke_cpu.yaml --model tiny_llama --device cpu --steps 5

Fine-tunes fresh adapters on the members (forget + retain) at the LoRA
config's `finetune_lr` and `train_batch_size`, with the step code of
`unlearn.train_lm` (`unlearn._update`), and logs every step: loss, whether
it was skipped (non-finite loss or GradScaler), the scaler's scale, the
gradient norm before clipping, the change of the adapter weights from their
start and the norm of the lora_B weights (zero at the start). Each arm
changes one thing from the campaign recipe (`amp`: fp16 base weights, fp32
adapters, fp16 autocast + GradScaler, gradient checkpointing):
  amp            the campaign recipe
  fp32_adapters  no autocast and no GradScaler (fp16 base, fp32 adapters; no bf16)
  amp_no_ckpt    gradient checkpointing off
  amp_eager      eager attention instead of SDPA
  amp_fp32_base  fp32 base weights (autocast + GradScaler as in the full fine-tunes)
Per arm: the retain loss on the first `--eval-records` retain records before
and after, the step counters (`unlearn.StepStats`), whether every adapter got a
gradient at the first step (every lora_B a nonzero one), and the parameters with non-finite gradients at
the first skipped step. Per-step rows go to <out>/lora_diag_steps.csv, the
summary to <out>/lora_diag.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ARMS = {
    "amp": {},
    "fp32_adapters": {"precision": "fp32"},
    "amp_no_ckpt": {"lora": {"gradient_checkpointing": False}},
    "amp_eager": {"attn": "eager"},
    "amp_fp32_base": {"lora": {"base_dtype": "float32"}},
}


def _arm_cfg(cfg, arm: str):
    a = ARMS[arm]
    lora = {**(cfg.model.lora or {}), **a.get("lora", {})}
    return replace(cfg, model=replace(cfg.model, precision=a.get("precision", cfg.model.precision), lora=lora))


def run_arm(cfg, arm: str, splits, device: str, steps: int, eval_set) -> tuple[dict, list[dict]]:
    import torch
    from torch.utils.data import ConcatDataset

    from unlearning import evaluate, unlearn
    from unlearning.seed import set_seed

    c = _arm_cfg(cfg, arm)
    set_seed(c.seed)
    model, _ = unlearn.build_model_and_tokenizer(c)
    if ARMS[arm].get("attn"):
        model.base_model.model.set_attn_implementation(ARMS[arm]["attn"])
    model.to(device)
    lora = {n: p for n, p in model.named_parameters() if p.requires_grad}
    start = {n: p.detach().clone() for n, p in lora.items()}
    info = {"arm": arm, "precision": c.model.precision, "amp": unlearn._use_amp(c, device),
            "lora": unlearn.lora_settings(c), "attn": model.config._attn_implementation,
            "trainable_tensors": len(lora), "trainable_dtypes": sorted({str(p.dtype) for p in lora.values()}),
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
            opt.zero_grad()
            stats.batches += 1
            step += 1
            with unlearn._autocast(c, device):
                loss = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                             labels=batch["labels"].to(device)).loss
            skipped_scaler = stats.skipped_scaler
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
            with torch.no_grad():
                delta = sum(float((p.detach().float() - start[n].float()).pow(2).sum()) for n, p in lora.items()) ** 0.5
                b_norm = sum(float(p.detach().float().pow(2).sum()) for n, p in lora.items() if "lora_B" in n) ** 0.5
            rows.append({"arm": arm, "step": step, "loss": float(loss.detach()), "skipped_loss": int(not torch.isfinite(loss)),
                         "skipped_scaler": stats.skipped_scaler - skipped_scaler, "scale": stats.scale_last,
                         "grad_norm": stats.grad_norm_last if step_applied else None,
                         "adapter_delta": delta, "lora_B_norm": b_norm})
            if step % 25 == 0 or step == 1:
                r = rows[-1]
                print(f"[diag] {arm} step {step:4d} loss {r['loss']:.4f} scaler skips {stats.skipped_scaler} "
                      f"scale {r['scale']} adapter change {delta:.4g} |B| {b_norm:.4g}", flush=True)
            if step >= steps:
                break
    stats.log(f"lora_diag {arm}")
    info["steps"] = stats.summary()
    info["retain_loss_after"] = evaluate.mean_loss(model, eval_set, c, device)
    info["adapter_delta"] = rows[-1]["adapter_delta"] if rows else 0.0
    info["lora_B_norm"] = rows[-1]["lora_B_norm"] if rows else 0.0
    del model, opt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return info, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/revision/lora.yaml")
    ap.add_argument("--model", help="override the config's model (e.g. tiny_llama for a CPU check)")
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-records", type=int, default=256)
    ap.add_argument("--device", help="override the config's device (cpu, cuda)")
    ap.add_argument("--out", default="results/revision/lora_diag")
    args = ap.parse_args(argv)

    from unlearning import data, unlearn
    from unlearning.config import load_config

    cfg = load_config(args.config)
    if args.model:
        cfg.model = replace(cfg.model, name=args.model)
    if cfg.model.lora is None:
        cfg.model = replace(cfg.model, lora={})
    if args.device:
        cfg.device = args.device
    device = unlearn.resolve_device(cfg)
    tok = None if cfg.model.name in ("tiny", "tiny_llama") else _tokenizer(cfg.model.name)
    splits = data.load_splits(cfg, tok)
    eval_set = splits.retain.head(args.eval_records) if hasattr(splits.retain, "head") else splits.retain
    print(f"[diag] {cfg.model.name} on {device}: {len(splits.forget) + len(splits.retain)} members, "
          f"lr {cfg.unlearn.finetune_lr}, batch {cfg.unlearn.train_batch_size}, arms {args.arms}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, all_rows = [], []
    for arm in args.arms:
        info, rows = run_arm(cfg, arm, splits, device, args.steps, eval_set)
        summary.append(info)
        all_rows += rows
        (out / "lora_diag.json").write_text(json.dumps(summary, indent=2, default=str))
        with open(out / "lora_diag_steps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    print("\n[diag] arm             retain loss before -> after   applied/batches  scaler skips  min scale  "
          "adapter change")
    for s in summary:
        st = s["steps"]
        print(f"[diag] {s['arm']:15s} {s['retain_loss_before']:.4f} -> {s['retain_loss_after']:.4f}"
              f"            {st['applied']}/{st['batches']}          {st['skipped_scaler']}"
              f"            {st['scale_min']}     {s['adapter_delta']:.4g}")
    print(f"[diag] -> {out / 'lora_diag.json'}, {out / 'lora_diag_steps.csv'}")
    return 0


def _tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


if __name__ == "__main__":
    sys.exit(main())
