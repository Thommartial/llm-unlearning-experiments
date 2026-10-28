"""Model construction, base fine-tuning, and approximate unlearning methods."""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass

import torch
from torch.utils.data import ConcatDataset, DataLoader

from . import attacks
from .config import ExperimentConfig
from .data import Splits, collate_for


def resolve_device(cfg: ExperimentConfig) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


# target_modules: a list, or lists by the model's `config.model_type`
LORA_DEFAULTS = {"r": 16, "alpha": 32, "dropout": 0.05,
                 "target_modules": {"llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
                                    "gpt_neox": ["query_key_value", "dense"]},
                 "base_dtype": "float16", "gradient_checkpointing": True}
NPO_REFERENCE = "npo_reference"  # second, frozen adapter holding the NPO reference under LoRA


def lora_settings(cfg: ExperimentConfig) -> dict | None:
    """`model.lora` with the defaults filled in; None without LoRA."""
    return None if cfg.model.lora is None else {**LORA_DEFAULTS, **cfg.model.lora}


def lora_targets(settings: dict, model_type: str) -> list[str]:
    """The LoRA target modules of a `model_type` model (`settings` from `lora_settings`)."""
    targets = settings["target_modules"]
    if isinstance(targets, dict):
        if model_type not in targets:
            raise ValueError(f"model.lora.target_modules has no entry for model type '{model_type}' "
                             f"(it has {sorted(targets)})")
        targets = targets[model_type]
    return list(targets)


def is_peft(model) -> bool:
    return hasattr(model, "peft_config")


def lora_filter(cfg: ExperimentConfig):
    """Parameter-name filter for the update attacks: the LoRA weights, or None."""
    return (lambda name: "lora_" in name) if cfg.model.lora is not None else None


def _with_lora(model, cfg: ExperimentConfig):
    """Wrap the base model in LoRA adapters: base weights in `base_dtype` and
    frozen, adapters in fp32 (peft's default), gradient checkpointing on."""
    s = lora_settings(cfg)
    if s is None:
        return model
    from peft import LoraConfig, get_peft_model

    model = model.to(getattr(torch, s["base_dtype"]))
    if s["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return get_peft_model(model, LoraConfig(r=s["r"], lora_alpha=s["alpha"], lora_dropout=s["dropout"],
                                            target_modules=lora_targets(s, model.config.model_type), task_type="CAUSAL_LM"))


def build_model_and_tokenizer(cfg: ExperimentConfig):
    """Return (model, tokenizer). 'tiny' builds a small untrained GPT-2,
    'tiny_llama' a small untrained Llama and 'tiny_neox' a small untrained
    GPT-NeoX (Pythia's architecture), all offline. With `model.lora` the
    model comes wrapped in LoRA adapters (`_with_lora`)."""
    if cfg.model.name == "tiny_llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=cfg.data.vocab_size, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=max(cfg.data.seq_len, cfg.data.max_length, 64),
            bos_token_id=None, eos_token_id=None, pad_token_id=None,
        )
        return _with_lora(LlamaForCausalLM(config), cfg), None
    if cfg.model.name == "tiny_neox":
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

        config = GPTNeoXConfig(
            vocab_size=cfg.data.vocab_size, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            max_position_embeddings=max(cfg.data.seq_len, cfg.data.max_length, 64),
            bos_token_id=None, eos_token_id=None,
        )
        return _with_lora(GPTNeoXForCausalLM(config), cfg), None
    if cfg.model.name == "tiny":
        from transformers import GPT2Config, GPT2LMHeadModel

        n_pos = max(cfg.data.seq_len, cfg.data.max_length, 64)
        config = GPT2Config(
            vocab_size=cfg.data.vocab_size, n_positions=n_pos,
            n_embd=64, n_layer=2, n_head=2,
            bos_token_id=None, eos_token_id=None,
        )
        return _with_lora(GPT2LMHeadModel(config), cfg), None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        if cfg.model.lora is None:
            model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg.model.name,
                                                         dtype=getattr(torch, lora_settings(cfg)["base_dtype"]))
    except Exception as e:
        raise RuntimeError(
            f"Could not load the weights of {cfg.model.name} ({type(e).__name__}: {e}). Check internet access and "
            f"disk space; TinyLlama v1.1 ships pytorch_model.bin, which transformers reads with torch.load."
        ) from e
    return _with_lora(model, cfg), tok


def _loader(dataset, cfg: ExperimentConfig, shuffle: bool = True,
            batch_size: int | None = None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size or cfg.unlearn.batch_size,
                      shuffle=shuffle, collate_fn=collate_for(dataset))


def _use_amp(cfg: ExperimentConfig, device: str) -> bool:
    """fp16 mixed precision runs on CUDA only; the CPU path stays plain fp32."""
    return cfg.model.precision == "fp16_amp" and str(device).startswith("cuda")


def _autocast(cfg: ExperimentConfig, device: str):
    if _use_amp(cfg, device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@dataclass
class StepStats:
    """Optimizer-step counters of one training loop (`train_lm`,
    `apply_unlearning`, client GA), printed at its end by `log`.

    `batches`: batches seen; `skipped_loss`: non-finite loss, no step;
    `skipped_scaler`: the GradScaler skipped the step (inf/NaN gradients
    under fp16 AMP; its scale was lowered); `skipped_grad`: no scaler and a
    non-finite gradient norm, no step (`_update`); `nonfinite_grad`: gradient
    norm not finite (the steps of `skipped_scaler` or `skipped_grad`);
    `zero_grad`: gradient norm exactly 0 (the step only applies weight
    decay); the scaler's scale (1.0 without AMP) and the mean finite
    gradient norm before clipping.
    """
    batches: int = 0
    skipped_loss: int = 0
    skipped_scaler: int = 0
    skipped_grad: int = 0
    nonfinite_grad: int = 0
    zero_grad: int = 0
    scale_first: float | None = None
    scale_min: float | None = None
    scale_last: float | None = None
    grad_norm_sum: float = 0.0
    grad_norm_n: int = 0
    grad_norm_last: float | None = None

    @property
    def applied(self) -> int:
        return self.batches - self.skipped_loss - self.skipped_scaler - self.skipped_grad

    @property
    def grad_norm_mean(self) -> float | None:
        return self.grad_norm_sum / self.grad_norm_n if self.grad_norm_n else None

    def record(self, grad_norm: float, scale_before: float, scale_after: float, skipped: bool = False) -> None:
        self.grad_norm_last = grad_norm
        self.skipped_grad += int(skipped)
        if not math.isfinite(grad_norm):
            self.nonfinite_grad += 1
        else:
            self.zero_grad += int(grad_norm == 0)
            self.grad_norm_sum += grad_norm
            self.grad_norm_n += 1
        self.skipped_scaler += int(scale_after < scale_before)
        self.scale_first = scale_before if self.scale_first is None else self.scale_first
        self.scale_min = min(scale_after, self.scale_min if self.scale_min is not None else scale_after)
        self.scale_last = scale_after

    def summary(self) -> dict:
        return {"batches": self.batches, "applied": self.applied, "skipped_loss": self.skipped_loss,
                "skipped_scaler": self.skipped_scaler, "skipped_grad": self.skipped_grad,
                "nonfinite_grad": self.nonfinite_grad,
                "zero_grad": self.zero_grad, "scale_first": self.scale_first, "scale_min": self.scale_min,
                "scale_last": self.scale_last, "grad_norm_mean": self.grad_norm_mean}

    def log(self, where: str) -> None:
        def fmt(v):
            return "-" if v is None else f"{v:.4g}"
        print(f"[steps] {where}: {self.batches} batches, {self.applied} applied, {self.skipped_loss} non-finite "
              f"loss, {self.skipped_scaler} scaler skips, {self.skipped_grad} non-finite grad skips (no scaler), "
              f"{self.nonfinite_grad} non-finite / {self.zero_grad} zero "
              f"grad norms, scale {fmt(self.scale_first)} -> {fmt(self.scale_last)} (min {fmt(self.scale_min)}), "
              f"mean grad norm {fmt(self.grad_norm_mean)}", flush=True)


def _step(loss, model, opt, scaler, stats: StepStats | None = None) -> None:
    """Backward, then `_update`."""
    scaler.scale(loss).backward()
    _update(model, opt, scaler, stats)


def _update(model, opt, scaler, stats: StepStats | None = None) -> None:
    """Clip at 1.0 and step; the scaler is a pass-through when disabled. A
    step the scaler skips (inf/NaN gradients) lowers its scale; without a
    scaler, a step with a non-finite gradient norm is skipped here instead
    (applying it would turn the weights into NaN)."""
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scale = scaler.get_scale()
    skipped = not scaler.is_enabled() and not bool(torch.isfinite(norm))
    if not skipped:
        scaler.step(opt)
        scaler.update()
    if stats is not None:
        stats.record(float(norm), scale, scaler.get_scale(), skipped)


def _scored(labels) -> int:
    """Tokens the causal-LM loss averages over (shifted labels that are not -100)."""
    return int((labels[:, 1:] != -100).sum())


def _micro_batches(batch: dict, n: int) -> list[dict]:
    """Split a right-padded batch into `n` micro-batches, each cut to its longest record."""
    out = []
    for idx in torch.arange(len(batch["input_ids"])).chunk(n):
        width = max(1, int(batch["attention_mask"][idx].sum(1).max()))
        out.append({k: batch[k][idx, :width] for k in ("input_ids", "attention_mask", "labels")})
    return out


def _accumulate(model, batch: dict, n: int, cfg: ExperimentConfig, device: str, scaler) -> bool:
    """Backward over `n` micro-batches of `batch`, each loss weighted by its share
    of the batch's scored tokens, so the summed gradient is the whole batch's.
    Stops and returns False on a non-finite loss."""
    total = max(1, _scored(batch["labels"]))
    for micro in _micro_batches(batch, n):
        share = _scored(micro["labels"]) / total
        if share == 0:
            continue
        with _autocast(cfg, device):
            loss = model(input_ids=micro["input_ids"].to(device),
                         attention_mask=micro["attention_mask"].to(device),
                         labels=micro["labels"].to(device)).loss
        if not torch.isfinite(loss):
            return False
        scaler.scale(loss * share).backward()
    return True


def train_lm(model, dataset, cfg: ExperimentConfig, device: str, epochs: int, lr: float,
             stats: StepStats | None = None) -> StepStats:
    """Standard causal-LM fine-tuning loop (batch size `train_batch_size`).

    With `grad_accum_steps` n > 1 each batch runs as n micro-batches
    (`_accumulate`): the same gradient, effective batch and learning rate,
    with about 1/n of the activation memory. Returns the step counters
    (added to `stats` if given), also printed.
    """
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(cfg, device))
    batch_size = cfg.unlearn.train_batch_size or cfg.unlearn.batch_size
    accum = cfg.unlearn.grad_accum_steps
    if accum < 1:
        raise ValueError(f"unlearn.grad_accum_steps must be at least 1, got {accum}.")
    stats = StepStats() if stats is None else stats
    for _ in range(epochs):
        for batch in _loader(dataset, cfg, batch_size=batch_size):
            opt.zero_grad()
            stats.batches += 1
            if accum > 1:
                if not _accumulate(model, batch, accum, cfg, device, scaler):
                    stats.skipped_loss += 1
                    continue
                _update(model, opt, scaler, stats)
                continue
            with _autocast(cfg, device):
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
            if not torch.isfinite(out.loss):
                stats.skipped_loss += 1
                continue
            _step(out.loss, model, opt, scaler, stats)
    stats.log("train_lm")
    return stats


def finetune_base(cfg: ExperimentConfig, splits: Splits, device: str, model=None,
                  stats: StepStats | None = None):
    """Fine-tune the base model (built fresh unless given) on all members (forget + retain)."""
    if model is None:
        model, _ = build_model_and_tokenizer(cfg)
    members = ConcatDataset([splits.forget, splits.retain])
    lr = cfg.unlearn.finetune_lr or cfg.unlearn.learning_rate
    train_lm(model, members, cfg, device, cfg.unlearn.finetune_epochs, lr, stats)
    return model


def retrain_base(cfg: ExperimentConfig, splits: Splits, device: str, model=None):
    """Retrain baseline: fine-tune the base model on the retain set only (same epochs and lr)."""
    if model is None:
        model, _ = build_model_and_tokenizer(cfg)
    lr = cfg.unlearn.finetune_lr or cfg.unlearn.learning_rate
    train_lm(model, splits.retain, cfg, device, cfg.unlearn.finetune_epochs, lr)
    return model


def apply_unlearning(model, splits: Splits, cfg: ExperimentConfig, device: str,
                     stats: StepStats | None = None):
    """Dispatch to the configured unlearning method (utility-anchored).
    The step counters go to `stats` if given and are printed."""
    method = cfg.unlearn.method
    stats = StepStats() if stats is None else stats
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.unlearn.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(cfg, device))

    ref_logprob, release = _npo_reference(model, device) if method == "npo" else (None, None)

    retain_iter = iter(_loader(splits.retain, cfg))

    def next_retain():
        nonlocal retain_iter
        try:
            return next(retain_iter)
        except StopIteration:
            retain_iter = iter(_loader(splits.retain, cfg))
            return next(retain_iter)

    def lm_loss(batch):
        return model(input_ids=batch["input_ids"].to(device),
                     attention_mask=batch["attention_mask"].to(device),
                     labels=batch["labels"].to(device)).loss

    for _ in range(cfg.unlearn.epochs):
        for batch in _loader(splits.forget, cfg):
            opt.zero_grad()
            stats.batches += 1
            with _autocast(cfg, device):
                forget_loss = lm_loss(batch)

                if method == "gradient_ascent":
                    loss = -forget_loss
                elif method == "gradient_difference":
                    loss = -forget_loss + cfg.unlearn.retain_weight * lm_loss(next_retain())
                elif method == "npo":
                    beta = cfg.unlearn.beta
                    # log-probs come back in fp32; keep the NPO terms in fp32
                    lp_theta = attacks.sequence_avg_logprob(model, batch, device)
                    lp_ref = ref_logprob(batch)
                    npo = (2.0 / beta) * torch.nn.functional.softplus(
                        beta * (lp_theta - lp_ref).float()
                    ).mean()
                    loss = npo + cfg.unlearn.retain_weight * lm_loss(next_retain())
                else:
                    raise ValueError(f"Unknown unlearning method '{method}'.")

            if not torch.isfinite(loss):
                stats.skipped_loss += 1
                continue  # skip pathological steps instead of corrupting the model
            _step(loss, model, opt, scaler, stats)
    stats.log(f"apply_unlearning {method}")
    if release is not None:
        release()
    return model


def _npo_reference(model, device: str):
    """Frozen NPO reference: returns (log-prob function, release).

    Full fine-tuning: a deep copy of the model in eval mode. LoRA: a second
    adapter in the same PeftModel holding a copy of the trained adapter,
    frozen, and used through `set_adapter` (in eval mode, no dropout), so
    the base weights are not copied; `release` deletes it.
    """
    if not is_peft(model):
        ref = copy.deepcopy(model).to(device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        @torch.no_grad()
        def logprob(batch):
            return attacks.sequence_avg_logprob(ref, batch, device)
        return logprob, None

    active = model.active_adapter
    model.add_adapter(NPO_REFERENCE, model.peft_config[active])
    model.set_adapter(active)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, p in params.items():
            if f".{NPO_REFERENCE}." in name:
                p.copy_(params[name.replace(f".{NPO_REFERENCE}.", f".{active}.")])
                p.requires_grad_(False)
    model.to(device)

    @torch.no_grad()
    def logprob(batch):
        model.set_adapter(NPO_REFERENCE)
        model.eval()
        try:
            return attacks.sequence_avg_logprob(model, batch, device)
        finally:
            model.set_adapter(active)  # also makes only the trained adapter require grad again
            model.train()

    def release():
        model.delete_adapter(NPO_REFERENCE)
        model.set_adapter(active)
    return logprob, release
