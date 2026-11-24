"""Metrics without an HR reference, for the real-world set (docs/PROJECT.md §9, PR 2.5).

- ``lr_psnr_y``: consistency with the input. SR is downscaled back to the LR size (torch bicubic
  with antialias, the degradation v0) and rounded to 8 bit; the score is PSNR-Y against the LR
  input. It shows whether the output still explains the input; it does not measure quality, and
  plain bicubic upsampling scores high on it.
- ``fid`` and ``kid``: distances between the InceptionV3 features of the SR outputs and those of
  real HR faces (FID: arXiv:1706.08500; KID: arXiv:1801.01401). They measure how close the
  outputs look to real faces as a set, not per image.

``InceptionFeatures`` uses the torchmetrics wrapper of the torch-fidelity InceptionV3 (the
2048-d pool features of the TensorFlow FID model), with the TensorFlow-compatible bilinear
resize to 299 px (``antialias=False``). FID and KID follow the torchmetrics formulas, in
float64, on stored features, so the reference features can be cached.
"""

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from facehall.eval.metrics import psnr_y

Features = npt.NDArray[np.float32]

KID_DEGREE = 3  # polynomial kernel (x . y / d + 1) ** 3, as in the KID paper and torchmetrics


class FeatureExtractor(Protocol):
    """Maps (B, 3, H, W) uint8 images to (B, D) features."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...


class InceptionFeatures:
    """2048-d InceptionV3 pool features (torch-fidelity weights, ``download_weights.py``)."""

    dims = 2048

    def __init__(self, weights: Path | str, device: torch.device | str = "cpu") -> None:
        from torchmetrics.image.fid import NoTrainInceptionV3

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"{self.weights} not found; run scripts/download_weights.py inception"
            )
        self.device = torch.device(device)
        self.net = NoTrainInceptionV3(
            name="inception-v3-compat",
            features_list=[str(self.dims)],
            feature_extractor_weights_path=str(self.weights),
            antialias=False,
        ).to(self.device)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features: torch.Tensor = self.net(images.to(self.device))
        return features.reshape(len(images), -1).float()


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) in [0, 1] -> uint8 levels."""
    return (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def downscale(sr: torch.Tensor, size: int) -> torch.Tensor:
    """Bicubic downscale with antialias to ``size`` px, rounded to 8-bit levels."""
    small = F.interpolate(
        sr, size=(size, size), mode="bicubic", antialias=True, align_corners=False
    )
    return torch.round(small.clamp(0.0, 1.0) * 255.0) / 255.0


def lr_psnr_y(sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
    """PSNR-Y of the downscaled SR against the LR input, per image."""
    return psnr_y(downscale(sr, lr.shape[-1]), lr)


def frechet_distance(
    mu1: npt.NDArray[np.float64],
    sigma1: npt.NDArray[np.float64],
    mu2: npt.NDArray[np.float64],
    sigma2: npt.NDArray[np.float64],
) -> float:
    """Fréchet distance of two Gaussians; tr(sqrt(S1 S2)) from the eigenvalues of S1 S2."""
    eigenvalues = torch.linalg.eigvals(torch.from_numpy(sigma1 @ sigma2))
    root_trace = float(eigenvalues.sqrt().real.sum())
    mean_term = float(np.square(mu1 - mu2).sum())
    return mean_term + float(np.trace(sigma1) + np.trace(sigma2)) - 2.0 * root_trace


def gaussian(features: Features) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mean and unbiased covariance of (N, D) features."""
    values = features.astype(np.float64)
    if len(values) < 2:
        raise ValueError("FID needs at least two samples per set")
    return values.mean(axis=0), np.cov(values, rowvar=False)


def fid(real: Features, fake: Features) -> float:
    return frechet_distance(*gaussian(real), *gaussian(fake))


def kid(
    real: Features,
    fake: Features,
    *,
    subsets: int,
    subset_size: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Mean and standard deviation of the unbiased polynomial-kernel MMD^2 over random subsets.

    Each subset draws ``subset_size`` samples without replacement from both sets; the size is
    capped by the smaller set. Deterministic for a given ``seed``.
    """
    size = min(subset_size, len(real), len(fake))
    if size < 2:
        raise ValueError("KID needs at least two samples per set")
    rng = np.random.default_rng(seed)
    real_t = torch.from_numpy(real.astype(np.float64)).to(device)
    fake_t = torch.from_numpy(fake.astype(np.float64)).to(device)
    values = []
    for _ in range(subsets):
        x = real_t[torch.from_numpy(rng.choice(len(real), size, replace=False)).to(device)]
        y = fake_t[torch.from_numpy(rng.choice(len(fake), size, replace=False)).to(device)]
        values.append(polynomial_mmd(x, y))
    array = np.asarray(values)
    return float(array.mean()), float(array.std())


def polynomial_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    """Unbiased MMD^2 with the kernel (x . y / d + 1) ** 3 for equally sized samples."""
    m, dims = x.shape
    k_xx = (x @ x.T / dims + 1.0) ** KID_DEGREE
    k_yy = (y @ y.T / dims + 1.0) ** KID_DEGREE
    k_xy = (x @ y.T / dims + 1.0) ** KID_DEGREE
    within = (k_xx.sum() - k_xx.diagonal().sum() + k_yy.sum() - k_yy.diagonal().sum()) / (
        m * (m - 1)
    )
    return float(within - 2.0 * k_xy.sum() / m**2)
