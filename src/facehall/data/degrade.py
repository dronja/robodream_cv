"""HR -> LR degradations.

v0 (Phase 1) is a plain bicubic downscale with antialias (docs/PROJECT.md §9). Phase 2 adds blur,
noise, interpolation choice and compression behind the same interface. A degradation may draw
from the per-sample generator it receives, so validation and test stay reproducible.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from facehall.data.config import DegradeConfig


class Degradation(Protocol):
    def __call__(self, hr: torch.Tensor, scale: int, rng: np.random.Generator) -> torch.Tensor:
        """Degrade a (C, H, W) float image in [0, 1] to (C, H / scale, W / scale)."""
        ...

    def describe(self) -> dict[str, object]:
        """Exact settings and implementation, recorded with every run."""
        ...


@dataclass(frozen=True)
class Bicubic:
    """Bicubic downscale; deterministic, so ``rng`` is not used."""

    antialias: bool = True
    quantize: bool = True

    def __call__(self, hr: torch.Tensor, scale: int, rng: np.random.Generator) -> torch.Tensor:
        if hr.ndim != 3:
            raise ValueError(f"expected a (C, H, W) image, got shape {tuple(hr.shape)}")
        _, height, width = hr.shape
        if scale < 1 or height % scale or width % scale:
            raise ValueError(f"scale {scale} does not divide the image size {height}x{width}")
        lr = F.interpolate(
            hr[None],
            size=(height // scale, width // scale),
            mode="bicubic",
            antialias=self.antialias,
            align_corners=False,
        )[0].clamp(0.0, 1.0)
        if self.quantize:
            lr = torch.round(lr * 255.0) / 255.0
        return lr

    def describe(self) -> dict[str, object]:
        return {
            "kind": "bicubic",
            "antialias": self.antialias,
            "quantize": self.quantize,
            "function": (
                "torch.nn.functional.interpolate(mode='bicubic', "
                f"antialias={self.antialias}, align_corners=False), clamped to [0, 1]"
            ),
            "torch": torch.__version__,
        }


def build_degradation(config: DegradeConfig) -> Degradation:
    if config.kind == "bicubic":
        return Bicubic(antialias=config.antialias, quantize=config.quantize)
    raise ValueError(f"unknown degradation kind {config.kind!r}")
