"""Random views of an image for losses computed by fixed networks (PR 3.4, PR 3.8).

A generator trained against a fixed network on a fixed pixel grid can learn a regular pixel
pattern that the network rewards: the stripe grid of the identity loss (PR 3.3) or the
hatching of the VGG19 perceptual loss (PR 3.6). ``RandomView`` shows the network a view of the
image that the generator cannot predict, the same view for SR and HR:

- a square ``box`` of the image (``crop_box``), resampled to ``out_size`` pixels;
- ``jitter``: a random similarity of that square per image (shift, rotation, scale; the
  semantics of ``facehall.align.Jitter``), so that a fixed pixel pattern lands at a different
  phase in every step and stops paying off;
- ``blur``: a Gaussian low-pass (sigma in input pixels) before resampling, which removes the
  highest frequencies, where such patterns live, from the network's input and gradient.

The jitter is drawn from the view's own generator (``generator``, seeded from the run seed and
a ``stream`` constant per loss), so a training run can save and restore it.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def crop_box(size: int, side: float, center_y: float) -> tuple[int, int, int]:
    """``(top, left, side)`` in pixels of the square of relative ``side`` in a ``size`` crop."""
    pixels = round(side * size)
    top, left = round(center_y * size - pixels / 2), round((size - pixels) / 2)
    if not 0 < pixels <= size or top < 0 or top + pixels > size:
        raise ValueError(f"the crop (side {side}, centre y {center_y}) leaves the image")
    return top, left, pixels


def gaussian_kernel(sigma: float) -> torch.Tensor:
    """Normalized 1-D Gaussian of radius ``ceil(3 sigma)``."""
    radius = max(1, math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).float()


class RandomView(nn.Module):
    """A square of (B, C, size, size) images, optionally jittered and low-passed, resampled."""

    kernel: torch.Tensor

    def __init__(
        self,
        size: int,
        box: tuple[int, int, int],
        out_size: int,
        *,
        shift: float = 0.0,
        rotation: float = 0.0,
        scale: float = 0.0,
        blur: float = 0.0,
        seed: int = 0,
        stream: int = 0,
    ) -> None:
        super().__init__()
        if min(shift, rotation, scale, blur) < 0:
            raise ValueError("jitter and blur must be non-negative")
        self.size, self.box, self.out_size = size, box, out_size
        self.shift, self.rotation, self.scale, self.blur = shift, rotation, scale, blur
        self.register_buffer("kernel", gaussian_kernel(blur) if blur > 0 else torch.ones(1))
        entropy = np.random.SeedSequence([seed, stream]).generate_state(1)[0]
        self.generator = torch.Generator().manual_seed(int(entropy))

    @property
    def jitters(self) -> bool:
        return max(self.shift, self.rotation, self.scale) > 0

    def transforms(self, batch: int, jitter: bool) -> torch.Tensor:
        """``affine_grid`` matrices (batch, 2, 3) that map the output onto the square."""
        top, left, side = self.box
        half = side / self.size  # half-width of the square in normalized coordinates
        centre = torch.tensor(
            [2.0 * (left + side / 2) / self.size - 1.0, 2.0 * (top + side / 2) / self.size - 1.0]
        )
        theta = torch.zeros(batch, 2, 3)
        theta[:, 0, 0] = theta[:, 1, 1] = half
        theta[:, :, 2] = centre
        if jitter and self.jitters:
            dx, dy, angle, factor = (
                torch.rand(4, batch, generator=self.generator) * 2 - 1
            ).unbind()
            angle = torch.deg2rad(angle * self.rotation)
            factor = (1.0 + self.scale) ** factor
            cos, sin = torch.cos(angle) * half * factor, torch.sin(angle) * half * factor
            theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
            theta[:, :, 2] += 2.0 * half * self.shift * torch.stack([dx, dy], dim=1)
        return theta

    def forward(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Low-pass, then resample the square given by ``theta`` to ``out_size``."""
        if self.blur > 0:
            channels, pad = images.shape[1], self.kernel.numel() // 2
            k = self.kernel.to(images.dtype)
            images = F.pad(images, (pad, pad, pad, pad), mode="reflect")
            images = F.conv2d(
                images, k.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels
            )
            images = F.conv2d(
                images, k.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels
            )
        shape = (images.shape[0], images.shape[1], self.out_size, self.out_size)
        grid = F.affine_grid(
            theta.to(images.device, images.dtype), list(shape), align_corners=False
        )
        return F.grid_sample(
            images, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )
