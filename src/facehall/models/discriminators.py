"""Discriminators for adversarial training (Phase 3, ESRGAN recipe).

``VGGDiscriminator``: a VGG-style discriminator as in SRGAN (arXiv:1609.04802); ESRGAN refers
to "the standard discriminator D in SRGAN". The layer layout follows ``VGGStyleDiscriminator``
of the ESRGAN authors' BasicSR code (Apache-2.0), read on 2026-11-25 and re-implemented here:
pairs of a 3×3 and a strided 4×4 convolution, BatchNorm after every convolution but the first,
LeakyReLU(0.2), the channels doubling from ``channels`` up to 8× ``channels``, then two dense
layers (100 units, then 1). Each stage halves the resolution; five stages take a 128 px image to
4×4. The output is the raw score C(x), one per image.
"""

import torch
from torch import nn


def _block(
    in_channels: int, out_channels: int, kernel: int, stride: int, norm: bool
) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            kernel // 2 if stride == 1 else 1,
            bias=not norm,
        )
    ]
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class VGGDiscriminator(nn.Module):
    def __init__(self, image_size: int = 128, channels: int = 64, stages: int = 5) -> None:
        super().__init__()
        if image_size % 2**stages or image_size // 2**stages < 1:
            raise ValueError(f"image size {image_size} is not divisible by 2^{stages}")
        widths = [min(channels * 2**i, channels * 8) for i in range(stages)]
        layers: list[nn.Module] = _block(3, widths[0], 3, 1, norm=False)
        layers += _block(widths[0], widths[0], 4, 2, norm=True)
        for previous, width in zip(widths, widths[1:], strict=False):
            layers += _block(previous, width, 3, 1, norm=True)
            layers += _block(width, width, 4, 2, norm=True)
        self.features = nn.Sequential(*layers)
        side = image_size // 2**stages
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1] * side * side, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.head(self.features(images)).squeeze(1)
        return result
