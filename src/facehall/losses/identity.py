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
(PR 3.4) act on SR and HR alike: a random similarity of the square per image (``jitter``) and a
Gaussian low-pass (``blur``), both from ``facehall.losses.views.RandomView``. The jitter is
drawn from the loss's own generator (``generator``), so a training run can save and restore it.
``similarity`` (validation) never jitters.
"""

import torch
import torch.nn.functional as F
from torch import nn

from facehall.losses.facenet import INPUT_SIZE
from facehall.losses.views import RandomView, crop_box, gaussian_kernel

GENERATOR_STREAM = 0x1D  # SeedSequence entropy next to the run seed: a stream of its own

__all__ = ["GENERATOR_STREAM", "IdentityLoss", "crop_box", "gaussian_kernel"]


class IdentityLoss(nn.Module):
    """Mean ``1 - cos`` of the embeddings of SR and HR (both (B, 3, size, size) in [0, 1])."""

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
        self.network = network.eval().requires_grad_(False)
        self.size = size
        self.view = RandomView(
            size,
            crop_box(size, side, center_y),
            INPUT_SIZE,
            shift=shift,
            rotation=rotation,
            scale=scale,
            blur=blur,
            seed=seed,
            stream=GENERATOR_STREAM,
        )

    @property
    def box(self) -> tuple[int, int, int]:
        return self.view.box

    @property
    def generator(self) -> torch.Generator:
        return self.view.generator

    @property
    def jitters(self) -> bool:
        return self.view.jitters

    def train(self, mode: bool = True) -> "IdentityLoss":
        super().train(mode)
        self.network.eval()  # always: the network is fixed, its BN uses the running statistics
        return self

    def transforms(self, batch: int, jitter: bool) -> torch.Tensor:
        """``affine_grid`` matrices (batch, 2, 3) that map the 160 px input onto the crop."""
        return self.view.transforms(batch, jitter)

    def faces(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Low-pass, resample to the network input and standardize as facenet expects."""
        faces: torch.Tensor = self.view(images, theta)
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
