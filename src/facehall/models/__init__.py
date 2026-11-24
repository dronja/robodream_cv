"""Super-resolution models: (B, 3, h, w) in [0, 1] -> (B, 3, h * scale, w * scale) in [0, 1].

Phase 1 baselines (``docs/PROJECT.md`` §9): ``bicubic`` and the 1:1 ports of the legacy ``fnet``
and ``cnet``. Models are built from a ``ModelConfig`` (``configs/model/*.yaml``).
"""

from dataclasses import dataclass

from omegaconf import MISSING
from torch import nn

from facehall.models.bicubic import Bicubic
from facehall.models.cnet import CNet
from facehall.models.fnet import FNet

MODELS = ("bicubic", "fnet", "cnet")


@dataclass
class ModelConfig:
    name: str = MISSING  # one of MODELS
    channels: int = 64  # fnet, cnet
    residual_blocks: int = 5  # fnet


def build_model(config: ModelConfig, scale: int, lr_size: int) -> nn.Module:
    """Build the model named in ``config`` for ``scale`` and LR crops of side ``lr_size``."""
    if config.name == "bicubic":
        return Bicubic(scale)
    if config.name == "fnet":
        return FNet(scale, channels=config.channels, residual_blocks=config.residual_blocks)
    if config.name == "cnet":
        return CNet(scale, lr_size, channels=config.channels)
    raise ValueError(f"unknown model {config.name!r}; expected one of {MODELS}")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (Keras "Trainable params")."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["MODELS", "Bicubic", "CNet", "FNet", "ModelConfig", "build_model", "count_parameters"]
