"""Bicubic upsampling: the baseline without learned parameters."""

import torch
import torch.nn.functional as F
from torch import nn


class Bicubic(nn.Module):
    """Bicubic upsampling by ``scale`` (``align_corners=False``), clamped to [0, 1]."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = scale

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        sr = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return sr.clamp(0.0, 1.0)
