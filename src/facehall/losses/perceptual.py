"""Perceptual loss: L1 between VGG19 features of SR and HR, before the activation (ESRGAN).

ESRGAN (arXiv:1809.00219, Sec. 3.3) uses "VGG19-54", the features of the 4th convolution before
the 5th max-pooling layer, taken before the activation. Inputs in [0, 1] are normalized with the
ImageNet mean and standard deviation of the torchvision weights. The network is frozen.

On a fixed pixel grid, this loss taught the generator a fine periodic hatching with periods of
2 and 4 px, which fits VGG19's pooling (PR 3.6). An optional ``view``
(``facehall.losses.views.RandomView``, PR 3.8) shows the network a randomly shifted, rotated and
scaled, low-passed copy of the whole image instead, the same for SR and HR.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from facehall.losses.views import RandomView

# Layer name -> number of torchvision ``vgg19().features`` modules up to that conv (inclusive),
# so the slice ends before its ReLU.
LAYERS = {"conv3_4": 17, "conv4_4": 26, "conv5_4": 35}
GENERATOR_STREAM = 0x9E  # SeedSequence entropy next to the run seed: a stream of its own
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def vgg19_features(weights: Path | str, layer: str = "conv5_4") -> nn.Sequential:
    """The torchvision VGG19 feature layers up to ``layer``, with weights from ``weights``."""
    from torchvision.models import vgg19

    if layer not in LAYERS:
        raise ValueError(f"unknown VGG19 layer {layer!r}; expected one of {sorted(LAYERS)}")
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/download_weights.py vgg19")
    state = torch.load(path, map_location="cpu", weights_only=True)
    features = vgg19().features
    prefix = "features."
    features.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    )
    return nn.Sequential(*list(features.children())[: LAYERS[layer]])


class PerceptualLoss(nn.Module):
    """Mean absolute difference of ``features`` of SR and HR (both in [0, 1])."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, features: nn.Module, view: RandomView | None = None) -> None:
        super().__init__()
        self.features = features.eval().requires_grad_(False)
        self.view = view
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "PerceptualLoss":
        super().train(mode)
        self.features.eval()  # always: the network is fixed
        return self

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.features((images - self.mean) / self.std)
        return result

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if self.view is not None:  # one random view per image, shared by SR and HR
            theta = self.view.transforms(sr.shape[0], jitter=True)
            sr, hr = self.view(sr, theta), self.view(hr, theta)
        with torch.no_grad():
            target = self.embed(hr)
        return F.l1_loss(self.embed(sr).float(), target.float())
