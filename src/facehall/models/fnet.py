"""FNet: the legacy "simplified SRResNet" (arXiv:1609.04802), ported 1:1 from ``legacy/FH.ipynb``.

Same layers as ``build_fnet_model`` in the notebook, including BatchNorm in the residual blocks
and the sigmoid output. The only generalisation is the scale: one conv + pixel-shuffle ×2 stage
per factor of 2 (the notebook had two stages, ×4).

Weight layout differs from Keras (NCHW, and PyTorch's pixel-shuffle channel order differs from
``tf.nn.depth_to_space``), so legacy Keras weights cannot be loaded as they are. They are not
needed: the legacy numbers are not reused (``docs/PROJECT.md`` §5).
"""

import torch
from torch import nn

from facehall.models.layers import batch_norm, conv, prelu


class ResidualBlock(nn.Module):
    """Conv → BN → PReLU → Conv → BN, plus the input."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv(channels, channels, 3),
            batch_norm(channels),
            prelu(channels),
            conv(channels, channels, 3),
            batch_norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x + self.body(x)
        return result


class FNet(nn.Module):
    def __init__(self, scale: int, channels: int = 64, residual_blocks: int = 5) -> None:
        super().__init__()
        stages = upsampling_stages(scale)
        self.scale = scale
        self.head = nn.Sequential(conv(3, channels, 9), prelu(channels))
        self.body = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(residual_blocks)),
            conv(channels, channels, 3),
            batch_norm(channels),
        )
        self.upsample = nn.Sequential(
            *(
                layer
                for _ in range(stages)
                for layer in (conv(channels, channels * 4, 3), nn.PixelShuffle(2), prelu(channels))
            )
        )
        self.tail = conv(channels, 3, 9)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        features = self.head(lr)
        features = features + self.body(features)
        return torch.sigmoid(self.tail(self.upsample(features)))


def upsampling_stages(scale: int) -> int:
    """Number of ×2 stages for ``scale``; it must be a power of two (2, 4, 8, …)."""
    if scale < 2 or scale & (scale - 1):
        raise ValueError(f"scale must be a power of two >= 2, got {scale}")
    return scale.bit_length() - 1
