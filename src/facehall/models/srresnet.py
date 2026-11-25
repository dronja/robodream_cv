"""SRResNet-v2: SRResNet (arXiv:1609.04802) with the changes of EDSR (arXiv:1707.02921).

Differences from the legacy FNet (``facehall.models.fnet``), as planned in ``CLAUDE.md`` §9:

- no BatchNorm: EDSR showed it hurts SR, and it makes train and eval behave differently;
- residual scaling: each block adds ``res_scale`` times its output (EDSR uses 0.1);
- more residual blocks (16 by default, as in SRResNet and the EDSR baseline);
- 3×3 head and tail convolutions and ReLU activations, as in EDSR;
- an optional global skip: the bicubic upsampling of the input is added to the output, so the
  network predicts only the difference;
- no sigmoid: the output is linear during training, so the loss sees errors at 0 and 1, and is
  clamped to [0, 1] in eval mode, which is the model contract of ``facehall.models``.

PyTorch default initialisation. The training recipe (L1 loss, Adam, cosine learning-rate decay)
is in ``configs/train/srresnet_x*.yaml``.
"""

import torch
import torch.nn.functional as F
from torch import nn

from facehall.models.fnet import upsampling_stages


class ResidualBlock(nn.Module):
    """Conv → ReLU → Conv, scaled by ``res_scale``, plus the input."""

    def __init__(self, channels: int, res_scale: float) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x + self.res_scale * self.body(x)
        return result


class SRResNet(nn.Module):
    def __init__(
        self,
        scale: int,
        channels: int = 64,
        residual_blocks: int = 16,
        res_scale: float = 0.1,
        bicubic_skip: bool = True,
    ) -> None:
        super().__init__()
        if channels < 1 or residual_blocks < 1:
            raise ValueError("channels and residual_blocks must be positive")
        if not 0.0 < res_scale <= 1.0:
            raise ValueError(f"res_scale must be in (0, 1], got {res_scale}")
        self.scale = scale
        self.bicubic_skip = bicubic_skip
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(
            *(ResidualBlock(channels, res_scale) for _ in range(residual_blocks)),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.upsample = nn.Sequential(
            *(
                layer
                for _ in range(upsampling_stages(scale))
                for layer in (
                    nn.Conv2d(channels, channels * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                )
            )
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        features = self.head(lr)
        features = features + self.body(features)
        sr: torch.Tensor = self.tail(self.upsample(features))
        if self.bicubic_skip:
            sr = sr + F.interpolate(
                lr, scale_factor=self.scale, mode="bicubic", align_corners=False
            )
        return sr if self.training else sr.clamp(0.0, 1.0)
