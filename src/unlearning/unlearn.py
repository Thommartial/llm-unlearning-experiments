"""Model construction, base fine-tuning, and approximate unlearning methods."""

from __future__ import annotations

import contextlib
import copy

import torch
from torch.utils.data import ConcatDataset, DataLoader

from . import attacks
from .config import ExperimentConfig
from .data import Splits, collate_for


def resolve_device(cfg: ExperimentConfig) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


LORA_DEFAULTS = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "base_dtype": "float16", "gradient_checkpointing": True}
NPO_REFERENCE = "npo_reference"  # second, frozen adapter holding the NPO reference under LoRA


def lora_settings(cfg: ExperimentConfig) -> dict | None:
    """`model.lora` with the defaults filled in; None without LoRA."""
    return None if cfg.model.lora is None else {**LORA_DEFAULTS, **cfg.model.lora}


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
                                            target_modules=list(s["target_modules"]), task_type="CAUSAL_LM"))


def build_model_and_tokenizer(cfg: ExperimentConfig):
    """Return (model, tokenizer). 'tiny' builds a small untrained GPT-2 and
    'tiny_llama' a small untrained Llama (offline). With `model.lora` the
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
    if cfg.model.lora is None:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=getattr(torch, lora_settings(cfg)["base_dtype"]))
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


def _step(loss, model, opt, scaler) -> None:
    """Backward, clip at 1.0 and step; the scaler is a pass-through when disabled."""
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()


def train_lm(model, dataset, cfg: ExperimentConfig, device: str, epochs: int, lr: float) -> None:
    """Standard causal-LM fine-tuning loop (batch size `train_batch_size`)."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(cfg, device))
    batch_size = cfg.unlearn.train_batch_size or cfg.unlearn.batch_size
    for _ in range(epochs):
        for batch in _loader(dataset, cfg, batch_size=batch_size):
            opt.zero_grad()
            with _autocast(cfg, device):
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
            if not torch.isfinite(out.loss):
                continue
            _step(out.loss, model, opt, scaler)


def finetune_base(cfg: ExperimentConfig, splits: Splits, device: str, model=None):
    """Fine-tune the base model (built fresh unless given) on all members (forget + retain)."""
    if model is None:
        model, _ = build_model_and_tokenizer(cfg)
    members = ConcatDataset([splits.forget, splits.retain])
    lr = cfg.unlearn.finetune_lr or cfg.unlearn.learning_rate
    train_lm(model, members, cfg, device, cfg.unlearn.finetune_epochs, lr)
    return model


def retrain_base(cfg: ExperimentConfig, splits: Splits, device: str, model=None):
    """Retrain baseline: fine-tune the base model on the retain set only (same epochs and lr)."""
    if model is None:
        model, _ = build_model_and_tokenizer(cfg)
    lr = cfg.unlearn.finetune_lr or cfg.unlearn.learning_rate
    train_lm(model, splits.retain, cfg, device, cfg.unlearn.finetune_epochs, lr)
    return model


def apply_unlearning(model, splits: Splits, cfg: ExperimentConfig, device: str):
    """Dispatch to the configured unlearning method (utility-anchored)."""
    method = cfg.unlearn.method
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
                continue  # skip pathological steps instead of corrupting the model
            _step(loss, model, opt, scaler)
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
