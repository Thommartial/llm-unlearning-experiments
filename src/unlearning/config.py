"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "gpt2"          # "tiny" builds a small untrained GPT-2 (offline)
    dtype: str = "float32"
    precision: str = "fp32"     # fp32 | fp16_amp (fp16 autocast + GradScaler, CUDA only)


@dataclass
class DataConfig:
    dataset: str = "tofu"        # "tofu" | "tofu_official" | "synthetic" | "synthetic_official"
    forget_fraction: float = 0.05
    holdout_fraction: float = 0.2
    max_samples: int | None = None
    max_length: int = 128
    # synthetic-only knobs (used for offline tests):
    n_synthetic: int = 240
    seq_len: int = 32
    vocab_size: int = 256
    outlier_fraction: float = 0.3


@dataclass
class UnlearnConfig:
    method: str = "gradient_ascent"  # gradient_ascent | gradient_difference | npo
    epochs: int = 3
    finetune_epochs: int = 1
    learning_rate: float = 1.0e-5      # used for unlearning
    finetune_lr: float | None = None    # base fine-tuning; defaults to learning_rate
    batch_size: int = 4                    # unlearning
    train_batch_size: int | None = None    # fine-tuning and FedAvg; defaults to batch_size
    eval_batch_size: int = 64              # per-record MIA scoring
    retain_weight: float = 1.0       # gradient_difference
    beta: float = 0.1                # npo


@dataclass
class AttackConfig:
    method: str = "loss_threshold"   # loss_threshold | min_k_prob
    min_k: float = 0.2


@dataclass
class FederatedConfig:
    """Federated fine-tuning and federated unlearning knobs.

    Client 0 holds the forget set; the remaining clients shard the retain set.
    """
    num_clients: int = 5
    rounds: int = 15                 # FedAvg aggregation rounds
    local_epochs: int = 1            # local epochs per client per round
    unlearn_method: str = "approx"   # approx (server-side, reuse unlearn.method) | retrain (FedAvg over retain clients)


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    output_dir: str = "results"
    device: str = "auto"             # auto | cpu | cuda
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    unlearn: UnlearnConfig = field(default_factory=UnlearnConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML file into a typed ExperimentConfig."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig(
        name=raw["name"],
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir", "results"),
        device=raw.get("device", "auto"),
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        unlearn=UnlearnConfig(**raw.get("unlearn", {})),
        attack=AttackConfig(**raw.get("attack", {})),
        federated=FederatedConfig(**raw.get("federated", {})),
    )
