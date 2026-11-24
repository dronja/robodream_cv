"""Config schema of a training run (``configs/train/*.yaml``)."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING, OmegaConf

from facehall.config import load_config, require_choice
from facehall.data.config import DataConfig, DegradeConfig
from facehall.models import MODELS, ModelConfig

LOSSES = ("mse", "l1")
OPTIMIZERS = ("adam",)
PRECISIONS = ("fp32", "bf16")


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8  # PyTorch default; the legacy recipe sets the Keras default 1e-7
    weight_decay: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = MISSING
    batch_size: int = 16
    grad_accum: int = 1  # micro-batches per optimizer step: effective batch = batch_size * this
    loss: str = "mse"  # one of LOSSES
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    precision: str = "bf16"  # autocast dtype of the training forward pass; loss and val in fp32
    early_stopping: int = 0  # patience: epochs without a lower val loss; 0 disables
    ema_decay: float = 0.0  # EMA of the weights, used for validation and `best`; 0 disables
    workers: int = 8  # DataLoader worker processes
    val_batch_size: int = 64
    log_every: int = 100  # optimizer steps between train-loss records; 0 disables


@dataclass
class SmokeConfig:
    """``--smoke``: a short run on random subsets, to check the pipeline before a full run."""

    train_samples: int = 512
    val_samples: int = 256
    epochs: int = 2


@dataclass
class ExperimentConfig:
    name: str = MISSING
    scale: int = MISSING  # LR side = data.hr_size / scale
    seed: int = 0  # model initialisation, sample order and augmentation
    device: str = "auto"  # auto (CUDA when available), cpu, cuda, cuda:N
    output: str = "runs"  # run directories are created inside
    data: DataConfig = field(default_factory=DataConfig)
    degrade: DegradeConfig = field(default_factory=DegradeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)


def load_experiment(path: Path | str, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load and validate a training config (``include`` and overrides: ``facehall.config``)."""
    config = load_config(path, ExperimentConfig, overrides)
    validate(config)
    return config


def validate(config: ExperimentConfig) -> None:
    """Check values the schema cannot express; the dataset and model check their own."""
    train = config.train
    require_choice(config.model.name, MODELS, "model")
    require_choice(train.loss, LOSSES, "loss")
    require_choice(train.optimizer.name, OPTIMIZERS, "optimizer")
    require_choice(train.precision, PRECISIONS, "precision")
    minimum = {
        "train.epochs": (train.epochs, 1),
        "train.batch_size": (train.batch_size, 1),
        "train.grad_accum": (train.grad_accum, 1),
        "train.val_batch_size": (train.val_batch_size, 1),
        "train.early_stopping": (train.early_stopping, 0),
        "train.workers": (train.workers, 0),
        "train.log_every": (train.log_every, 0),
        "smoke.train_samples": (config.smoke.train_samples, 1),
        "smoke.val_samples": (config.smoke.val_samples, 1),
        "smoke.epochs": (config.smoke.epochs, 1),
    }
    for name, (value, lowest) in minimum.items():
        if value < lowest:
            raise ValueError(f"{name} must be at least {lowest}, got {value}")
    if not 0.0 <= train.ema_decay < 1.0:
        raise ValueError(f"train.ema_decay must be in [0, 1), got {train.ema_decay}")


def experiment_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Plain, JSON-serialisable form of a config, e.g. for checkpoint metadata."""
    container = OmegaConf.to_container(OmegaConf.structured(config))
    assert isinstance(container, dict)
    return {str(key): value for key, value in container.items()}


def experiment_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    """Inverse of ``experiment_to_dict``, checked against the schema."""
    merged = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.create(values))
    config = OmegaConf.to_object(merged)
    if not isinstance(config, ExperimentConfig):
        raise TypeError("values do not describe an ExperimentConfig")
    return config
