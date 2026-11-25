"""Config schema of a training run (``configs/train/*.yaml``)."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING, OmegaConf

from facehall.config import load_config, require_choice
from facehall.data.config import DataConfig, DegradeConfig
from facehall.losses.identity import crop_box
from facehall.losses.perceptual import LAYERS
from facehall.models import MODELS, ModelConfig

LOSSES = ("mse", "l1")
OPTIMIZERS = ("adam",)
SCHEDULERS = ("constant", "cosine")
DISCRIMINATORS = ("vgg",)
BEST_METRICS = ("val_loss", "none")  # none: `best` is the latest epoch
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
class SchedulerConfig:
    """Learning rate per optimizer step. ``cosine`` decays from ``optimizer.lr`` to ``min_lr``
    over all steps of the run (half a cosine period, no restarts)."""

    name: str = "constant"  # one of SCHEDULERS
    min_lr: float = 0.0  # cosine only


@dataclass
class LossWeights:
    """Weights of the generator loss terms (ESRGAN: perceptual 1, adversarial 5e-3, pixel 1e-2)."""

    pixel: float = 1.0  # the pixel loss ``train.loss``
    perceptual: float = 0.0  # VGG19 features (``train.perceptual``)
    adversarial: float = 0.0  # relativistic average GAN (``train.discriminator``)
    identity: float = 0.0  # 1 - cosine of facenet embeddings (``train.identity``)


@dataclass
class PerceptualConfig:
    weights: str = ""  # VGG19 weights file (scripts/download_weights.py vgg19)
    layer: str = "conv5_4"  # facehall.losses.LAYERS; features before the activation


@dataclass
class IdentityConfig:
    """Input square of the recognition network, relative to the HR crop (``crop_box``)."""

    weights: str = ""  # facenet weights file (scripts/download_weights.py facenet_vggface2)
    crop_side: float = 0.875
    crop_center_y: float = 0.55  # vertical centre; the square is centred horizontally


@dataclass
class DiscriminatorConfig:
    name: str = "vgg"  # one of DISCRIMINATORS
    channels: int = 64
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class TrainConfig:
    epochs: int = MISSING
    init: str = ""  # checkpoint (.safetensors) whose weights initialize the model
    batch_size: int = 16
    grad_accum: int = 1  # micro-batches per optimizer step: effective batch = batch_size * this
    loss: str = "mse"  # pixel loss, one of LOSSES; also the validation loss
    losses: LossWeights = field(default_factory=LossWeights)
    perceptual: PerceptualConfig = field(default_factory=PerceptualConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    best_metric: str = "val_loss"  # one of BEST_METRICS: what selects the `best` checkpoint
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
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
    require_choice(train.scheduler.name, SCHEDULERS, "scheduler")
    require_choice(train.best_metric, BEST_METRICS, "best metric")
    weights = train.losses
    values = vars(weights).values()
    if min(values) < 0:
        raise ValueError("train.losses weights must not be negative")
    if sum(values) == 0:
        raise ValueError("train.losses: at least one weight must be positive")
    if weights.perceptual > 0:
        require_choice(train.perceptual.layer, tuple(LAYERS), "VGG19 layer")
        if not train.perceptual.weights:
            raise ValueError("train.perceptual.weights is required with a perceptual loss")
    if weights.identity > 0:
        if not train.identity.weights:
            raise ValueError("train.identity.weights is required with an identity loss")
        crop_box(config.data.hr_size, train.identity.crop_side, train.identity.crop_center_y)
    if weights.adversarial > 0:
        require_choice(train.discriminator.name, DISCRIMINATORS, "discriminator")
        require_choice(train.discriminator.optimizer.name, OPTIMIZERS, "optimizer")
        if train.grad_accum != 1:
            raise ValueError("train.grad_accum must be 1 with an adversarial loss")
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
    if not 0.0 <= train.scheduler.min_lr <= train.optimizer.lr:
        raise ValueError("train.scheduler.min_lr must be in [0, train.optimizer.lr]")
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
