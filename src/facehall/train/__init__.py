"""Training: experiment configs, the training loop, checkpoints (docs/PROJECT.md §9, PR 1.6)."""

from facehall.train.checkpoint import load_checkpoint, read_metadata, save_checkpoint
from facehall.train.config import (
    ExperimentConfig,
    OptimizerConfig,
    SchedulerConfig,
    SmokeConfig,
    TrainConfig,
    load_experiment,
)
from facehall.train.ema import ModelEMA
from facehall.train.trainer import resume, train

__all__ = [
    "ExperimentConfig",
    "ModelEMA",
    "OptimizerConfig",
    "SchedulerConfig",
    "SmokeConfig",
    "TrainConfig",
    "load_checkpoint",
    "load_experiment",
    "read_metadata",
    "resume",
    "save_checkpoint",
    "train",
]
