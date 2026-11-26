"""Periodic-pattern check: the power spectrum of SR against HR (D28).

A generator trained against a fixed network can learn a regular pixel pattern that the network
rewards: the stripe grid of the identity loss (PR 3.3) or the hatching of the VGG19 perceptual
loss (PR 3.6). Such a pattern shows as sharp peaks in the power spectrum at high frequencies,
where real faces have little power. FID and KID do not reliably see it (PR 3.6).

Measure, over all images of a protocol:

- the luma (BT.601 weights) of every image, its mean removed, times a 2-D Hann window;
- ``|FFT|²`` summed over the images, for SR and HR separately;
- the ratio SR / HR per frequency bin. The **peak ratio** is its maximum over the bins at a
  radius of at least ``min_radius`` cycles per pixel; the **band ratios** are the ratios of
  the power summed over radial bands.

A model that only smooths stays below 1 at high frequencies; a pattern gives a peak far above
it. The peak is a maximum over many bins, so it is noisy on a few images: compare it on full
protocols.
"""

import math
from typing import Any

import torch

LUMA = (0.299, 0.587, 0.114)
MIN_RADIUS = 0.25  # cycles per pixel
BAND_EDGES = (0.0, 0.125, 0.25, 0.375, 0.5, math.inf)  # cycles per pixel; > 0.5: the corners


class SpectrumRatio:
    """Accumulates the power spectra of SR and HR batches ((B, 3, H, W) in [0, 1])."""

    def __init__(self, height: int, width: int, device: torch.device | str = "cpu") -> None:
        self.shape = (height, width)
        self.device = torch.device(device)
        rows = torch.hann_window(height, periodic=False, dtype=torch.float64, device=self.device)
        cols = torch.hann_window(width, periodic=False, dtype=torch.float64, device=self.device)
        self.window = torch.outer(rows, cols)
        fy = torch.fft.fftfreq(height, dtype=torch.float64, device=self.device)
        fx = torch.fft.fftfreq(width, dtype=torch.float64, device=self.device)
        self.radius = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2)
        self.luma = torch.tensor(LUMA, dtype=torch.float64, device=self.device).view(1, 3, 1, 1)
        self.sr = torch.zeros(self.shape, dtype=torch.float64, device=self.device)
        self.hr = torch.zeros(self.shape, dtype=torch.float64, device=self.device)
        self.images = 0

    def power(self, images: torch.Tensor) -> torch.Tensor:
        """Summed power spectrum (H, W) of a batch, zero frequency at index (0, 0)."""
        if images.ndim != 4 or images.shape[1] != 3 or tuple(images.shape[2:]) != self.shape:
            raise ValueError(f"expected (B, 3, {self.shape[0]}, {self.shape[1]}) images")
        y = (images.to(self.device, torch.float64) * self.luma).sum(1)
        y = (y - y.mean((1, 2), keepdim=True)) * self.window
        power: torch.Tensor = (torch.fft.fft2(y).abs() ** 2).sum(0)
        return power

    def update(self, sr: torch.Tensor, hr: torch.Tensor) -> None:
        if sr.shape != hr.shape:
            raise ValueError(f"SR {tuple(sr.shape)} and HR {tuple(hr.shape)} differ")
        self.sr += self.power(sr)
        self.hr += self.power(hr)
        self.images += sr.shape[0]

    def ratio(self) -> torch.Tensor:
        """SR / HR power per frequency bin."""
        return self.sr / self.hr.clamp_min(torch.finfo(torch.float64).tiny)

    def peak_ratio(self, min_radius: float = MIN_RADIUS) -> float:
        mask = self.radius >= min_radius
        if not self.images or not mask.any():
            raise ValueError("no images, or no frequency bins at that radius")
        return float(self.ratio()[mask].max())

    def band_ratios(self) -> list[dict[str, Any]]:
        bands = []
        for low, high in zip(BAND_EDGES, BAND_EDGES[1:], strict=False):
            mask = (self.radius >= low) & (self.radius < high)
            ratio = float(self.sr[mask].sum() / self.hr[mask].sum()) if mask.any() else None
            bands.append({"from": low, "to": None if math.isinf(high) else high, "ratio": ratio})
        return bands

    def summary(self) -> dict[str, Any]:
        """What ``metrics.json`` records (``spectrum``)."""
        return {
            "images": self.images,
            "peak_ratio": self.peak_ratio(),
            "min_radius": MIN_RADIUS,
            "bands": self.band_ratios(),
        }
