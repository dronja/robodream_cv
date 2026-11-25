"""Identity loss: cosine distance of face-recognition embeddings of SR and HR (PR 3.3).

The recognition network is facenet's Inception-ResNet-v1 trained on VGGFace2
(``facehall.losses.facenet``). It is deliberately not the model of the ID-similarity metric
(InsightFace ``buffalo_l``, ArcFace R50 on WebFace600K, D18): another architecture, other
training data and another input crop, so training does not optimize the metric itself.

The input is a fixed square of the aligned HR crop, resized to 160×160. The square was chosen on
CelebA validation pairs (``docs/DECISIONS.md``, PR 3.3): side 0.875 of the crop, centred
horizontally and at 0.55 of the height. The network is frozen and stays in eval mode.
"""

import torch
import torch.nn.functional as F
from torch import nn

from facehall.losses.facenet import INPUT_SIZE


def crop_box(size: int, side: float, center_y: float) -> tuple[int, int, int]:
    """``(top, left, side)`` in pixels of the square of relative ``side`` in a ``size`` crop."""
    pixels = round(side * size)
    top, left = round(center_y * size - pixels / 2), round((size - pixels) / 2)
    if not 0 < pixels <= size or top < 0 or top + pixels > size:
        raise ValueError(f"the identity crop (side {side}, centre y {center_y}) leaves the image")
    return top, left, pixels


class IdentityLoss(nn.Module):
    """Mean ``1 - cos`` of the embeddings of SR and HR (both (B, 3, size, size) in [0, 1])."""

    def __init__(
        self, network: nn.Module, size: int, *, side: float = 0.875, center_y: float = 0.55
    ) -> None:
        super().__init__()
        self.network = network.eval().requires_grad_(False)
        self.box = crop_box(size, side, center_y)

    def train(self, mode: bool = True) -> "IdentityLoss":
        super().train(mode)
        self.network.eval()  # always: the network is fixed, its BN uses the running statistics
        return self

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        top, left, side = self.box
        faces = images[:, :, top : top + side, left : left + side]
        faces = F.interpolate(
            faces, size=INPUT_SIZE, mode="bilinear", align_corners=False, antialias=True
        )
        result: torch.Tensor = self.network((faces * 255.0 - 127.5) / 128.0)
        return result

    def similarity(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Per-image cosine similarity (B,); gradients flow to ``sr`` only."""
        with torch.no_grad():
            target = self.embed(hr)
        return F.cosine_similarity(self.embed(sr).float(), target.float(), dim=1)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.similarity(sr, hr)).mean()
