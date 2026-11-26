"""Identity loss: cosine distance of face-recognition embeddings of SR and HR (PR 3.3, 3.4).

The recognition network is facenet's Inception-ResNet-v1 trained on VGGFace2
(``facehall.losses.facenet``). It is deliberately not the model of the ID-similarity metric
(InsightFace ``buffalo_l``, ArcFace R50 on WebFace600K, D18): another architecture, other
training data and another input crop, so training does not optimize the metric itself.

The input is a square of the aligned HR crop, resampled to 160×160. The square was chosen on
CelebA validation pairs (``docs/DECISIONS.md``, PR 3.3): side 0.875 of the crop, centred
horizontally and at 0.55 of the height. The network is frozen and stays in eval mode.

Optimizing the embedding of a fixed square of fixed pixels lets the generator add a regular
pixel pattern that raises the similarity (a grid of stripes, PR 3.3). Two optional defences
(PR 3.4) act on SR and HR alike:

- ``jitter``: a random similarity of the square per image (shift, rotation, scale; the
  semantics of ``facehall.align.Jitter``), so that a fixed pixel pattern lands at a different
  phase in every step and stops paying off;
- ``blur``: a Gaussian low-pass of the crop (sigma in HR pixels) before resampling.

The jitter is drawn from the loss's own generator (``generator``), so a training run can save
and restore it. ``similarity`` (validation) never jitters.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from facehall.losses.facenet import INPUT_SIZE

GENERATOR_STREAM = 0x1D  # SeedSequence entropy next to the run seed: a stream of its own


def crop_box(size: int, side: float, center_y: float) -> tuple[int, int, int]:
    """``(top, left, side)`` in pixels of the square of relative ``side`` in a ``size`` crop."""
    pixels = round(side * size)
    top, left = round(center_y * size - pixels / 2), round((size - pixels) / 2)
    if not 0 < pixels <= size or top < 0 or top + pixels > size:
        raise ValueError(f"the identity crop (side {side}, centre y {center_y}) leaves the image")
    return top, left, pixels


def gaussian_kernel(sigma: float) -> torch.Tensor:
    """Normalized 1-D Gaussian of radius ``ceil(3 sigma)``."""
    radius = max(1, math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).float()


class IdentityLoss(nn.Module):
    """Mean ``1 - cos`` of the embeddings of SR and HR (both (B, 3, size, size) in [0, 1])."""

    kernel: torch.Tensor

    def __init__(
        self,
        network: nn.Module,
        size: int,
        *,
        side: float = 0.875,
        center_y: float = 0.55,
        shift: float = 0.0,
        rotation: float = 0.0,
        scale: float = 0.0,
        blur: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if min(shift, rotation, scale, blur) < 0:
            raise ValueError("identity jitter and blur must be non-negative")
        self.network = network.eval().requires_grad_(False)
        self.size = size
        self.box = crop_box(size, side, center_y)
        self.shift, self.rotation, self.scale, self.blur = shift, rotation, scale, blur
        self.register_buffer("kernel", gaussian_kernel(blur) if blur > 0 else torch.ones(1))
        entropy = np.random.SeedSequence([seed, GENERATOR_STREAM]).generate_state(1)[0]
        self.generator = torch.Generator().manual_seed(int(entropy))

    @property
    def jitters(self) -> bool:
        return max(self.shift, self.rotation, self.scale) > 0

    def train(self, mode: bool = True) -> "IdentityLoss":
        super().train(mode)
        self.network.eval()  # always: the network is fixed, its BN uses the running statistics
        return self

    def transforms(self, batch: int, jitter: bool) -> torch.Tensor:
        """``affine_grid`` matrices (batch, 2, 3) that map the 160 px input onto the crop."""
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

    def faces(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Low-pass, resample to the network input and standardize as facenet expects."""
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
        shape = (images.shape[0], images.shape[1], INPUT_SIZE, INPUT_SIZE)
        grid = F.affine_grid(
            theta.to(images.device, images.dtype), list(shape), align_corners=False
        )
        faces = F.grid_sample(
            images, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )
        return (faces * 255.0 - 127.5) / 128.0

    def embed(self, images: torch.Tensor, theta: torch.Tensor | None = None) -> torch.Tensor:
        if theta is None:
            theta = self.transforms(images.shape[0], jitter=False)
        result: torch.Tensor = self.network(self.faces(images, theta))
        return result

    def similarity(
        self, sr: torch.Tensor, hr: torch.Tensor, *, jitter: bool = False
    ) -> torch.Tensor:
        """Per-image cosine similarity (B,); gradients flow to ``sr`` only. With ``jitter``,
        SR and HR of one image share one random transform."""
        theta = self.transforms(sr.shape[0], jitter)
        with torch.no_grad():
            target = self.embed(hr, theta)
        return F.cosine_similarity(self.embed(sr, theta).float(), target.float(), dim=1)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.similarity(sr, hr, jitter=True)).mean()
