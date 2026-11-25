"""Inception-ResNet-v1 face recognition network of facenet, for the identity loss (PR 3.3).

A port of ``models/inception_resnet_v1.py`` of facenet-pytorch (https://github.com/timesler/
facenet-pytorch, MIT, Copyright (c) 2019 Timothy Esler; read on 2026-11-25), which ports the
TensorFlow models of David Sandberg's facenet (https://github.com/davidsandberg/facenet, MIT).
The layer layout and the parameter names are the same, so the published state dicts load as they
are. Only the embedding path is kept: the classification layer (``logits``) and dropout, which
does nothing in eval mode, are left out.

The network expects 160×160 RGB faces cropped by MTCNN and standardized as
``(255 x - 127.5) / 128`` (facenet's "fixed image standardization"); it returns L2-normalized
512-d embeddings. Weights: ``scripts/download_weights.py facenet_vggface2`` (licenses:
``docs/THIRD_PARTY_LICENSES.md`` §3).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

INPUT_SIZE = 160
EMBEDDING_SIZE = 512


class BasicConv2d(nn.Sequential):
    """Conv (no bias) → BN (TensorFlow eps 1e-3) → ReLU."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()


class _Residual(nn.Module):
    """Concatenated branches → 1×1 conv, scaled and added to the input, then ReLU (optional)."""

    n_branches = 0  # attributes branch0 … branch{n - 1}, set by the subclass

    def __init__(self, channels: int, mixed: int, scale: float, relu: bool) -> None:
        super().__init__()
        self.scale = scale
        self.conv2d = nn.Conv2d(mixed, channels, 1)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [getattr(self, f"branch{i}") for i in range(self.n_branches)]
        out: torch.Tensor = self.conv2d(torch.cat([branch(x) for branch in branches], 1))
        out = out * self.scale + x
        return self.relu(out) if self.relu is not None else out


class Block35(_Residual):
    n_branches = 3

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__(256, 96, scale, relu=True)
        self.branch0 = BasicConv2d(256, 32, 1)
        self.branch1 = nn.Sequential(BasicConv2d(256, 32, 1), BasicConv2d(32, 32, 3, padding=1))
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, 1),
            BasicConv2d(32, 32, 3, padding=1),
            BasicConv2d(32, 32, 3, padding=1),
        )


class Block17(_Residual):
    n_branches = 2

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__(896, 256, scale, relu=True)
        self.branch0 = BasicConv2d(896, 128, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, 1),
            BasicConv2d(128, 128, (1, 7), padding=(0, 3)),
            BasicConv2d(128, 128, (7, 1), padding=(3, 0)),
        )


class Block8(_Residual):
    n_branches = 2

    def __init__(self, scale: float = 1.0, relu: bool = True) -> None:
        super().__init__(1792, 384, scale, relu=relu)
        self.branch0 = BasicConv2d(1792, 192, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, 1),
            BasicConv2d(192, 192, (1, 3), padding=(0, 1)),
            BasicConv2d(192, 192, (3, 1), padding=(1, 0)),
        )


class Mixed6a(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(256, 384, 3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, 1),
            BasicConv2d(192, 192, 3, padding=1),
            BasicConv2d(192, 256, 3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)


class Mixed7a(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(BasicConv2d(896, 256, 1), BasicConv2d(256, 384, 3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(896, 256, 1), BasicConv2d(256, 256, 3, stride=2))
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, 1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (self.branch0, self.branch1, self.branch2, self.branch3)
        return torch.cat([branch(x) for branch in branches], 1)


class InceptionResnetV1(nn.Module):
    """Standardized faces (B, 3, 160, 160) → L2-normalized embeddings (B, 512)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv2d_1a = BasicConv2d(3, 32, 3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, 3)
        self.conv2d_2b = BasicConv2d(32, 64, 3, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, 1)
        self.conv2d_4a = BasicConv2d(80, 192, 3)
        self.conv2d_4b = BasicConv2d(192, 256, 3, stride=2)
        self.repeat_1 = nn.Sequential(*(Block35(scale=0.17) for _ in range(5)))
        self.mixed_6a = Mixed6a()
        self.repeat_2 = nn.Sequential(*(Block17(scale=0.10) for _ in range(10)))
        self.mixed_7a = Mixed7a()
        self.repeat_3 = nn.Sequential(*(Block8(scale=0.20) for _ in range(5)))
        self.block8 = Block8(relu=False)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(1792, EMBEDDING_SIZE, bias=False)
        self.last_bn = nn.BatchNorm1d(EMBEDDING_SIZE, eps=0.001, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = (
            self.conv2d_1a,
            self.conv2d_2a,
            self.conv2d_2b,
            self.maxpool_3a,
            self.conv2d_3b,
            self.conv2d_4a,
            self.conv2d_4b,
        )
        for layer in stem:
            x = layer(x)
        x = self.repeat_3(self.mixed_7a(self.repeat_2(self.mixed_6a(self.repeat_1(x)))))
        x = self.avgpool_1a(self.block8(x)).flatten(1)
        return F.normalize(self.last_bn(self.last_linear(x)), p=2, dim=1)


def load_facenet(weights: Path | str) -> InceptionResnetV1:
    """The network with a facenet-pytorch state dict from ``weights``, in eval mode."""
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run scripts/download_weights.py facenet_vggface2"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("logits.")}
    network = InceptionResnetV1()
    network.load_state_dict(state)
    return network.eval()
