"""Model construction, base fine-tuning, and approximate unlearning methods."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import ConcatDataset, DataLoader

from . import attacks
from .config import ExperimentConfig
from .data import Splits, collate


def resolve_device(cfg: ExperimentConfig) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def build_model_and_tokenizer(cfg: ExperimentConfig):
    """Return (model, tokenizer). 'tiny' builds a small untrained GPT-2 (offline)."""
    if cfg.model.name == "tiny":
        from transformers import GPT2Config, GPT2LMHeadModel

        n_pos = max(cfg.data.seq_len, cfg.data.max_length, 64)
        config = GPT2Config(
            vocab_size=cfg.data.vocab_size, n_positions=n_pos,
            n_embd=64, n_layer=2, n_head=2,
            bos_token_id=None, eos_token_id=None,
        )
        return GPT2LMHeadModel(config), None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    return model, tok


def _loader(dataset, cfg: ExperimentConfig, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=cfg.unlearn.batch_size,
                      shuffle=shuffle, collate_fn=collate)


def train_lm(model, dataset, cfg: ExperimentConfig, device: str, epochs: int, lr: float) -> None:
    """Standard causal-LM fine-tuning loop."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in _loader(dataset, cfg):
            opt.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            if not torch.isfinite(out.loss):
                continue
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def finetune_base(cfg: ExperimentConfig, splits: Splits, device: str):
    """Fine-tune the base model on all members (forget + retain)."""
    model, _ = build_model_and_tokenizer(cfg)
    members = ConcatDataset([splits.forget, splits.retain])
    lr = cfg.unlearn.finetune_lr or cfg.unlearn.learning_rate
    train_lm(model, members, cfg, device, cfg.unlearn.finetune_epochs, lr)
    return model


def apply_unlearning(model, splits: Splits, cfg: ExperimentConfig, device: str):
    """Dispatch to the configured unlearning method (utility-anchored)."""
    method = cfg.unlearn.method
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.unlearn.learning_rate)

    ref = None
    if method == "npo":
        ref = copy.deepcopy(model).to(device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)

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
            forget_loss = lm_loss(batch)

            if method == "gradient_ascent":
                loss = -forget_loss
            elif method == "gradient_difference":
                loss = -forget_loss + cfg.unlearn.retain_weight * lm_loss(next_retain())
            elif method == "npo":
                beta = cfg.unlearn.beta
                lp_theta = attacks.sequence_avg_logprob(model, batch, device)
                with torch.no_grad():
                    lp_ref = attacks.sequence_avg_logprob(ref, batch, device)
                npo = (2.0 / beta) * torch.nn.functional.softplus(
                    beta * (lp_theta - lp_ref)
                ).mean()
                loss = npo + cfg.unlearn.retain_weight * lm_loss(next_retain())
            else:
                raise ValueError(f"Unknown unlearning method '{method}'.")

            if not torch.isfinite(loss):
                continue  # skip pathological steps instead of corrupting the model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model
