"""Tests for the periodic-pattern check (facehall.eval.spectrum)."""

import pytest
import torch
import torch.nn.functional as F

from facehall.eval.spectrum import BAND_EDGES, MIN_RADIUS, SpectrumRatio

SIZE = 64


def faces(count: int = 6, seed: int = 0) -> torch.Tensor:
    """Smooth random images: little power at high frequencies, like downscaled photos."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(count, 3, SIZE // 8, SIZE // 8, generator=generator)
    return F.interpolate(coarse, size=SIZE, mode="bicubic", align_corners=False).clamp(0, 1)


def measure(sr: torch.Tensor, hr: torch.Tensor) -> SpectrumRatio:
    spectrum = SpectrumRatio(SIZE, SIZE)
    spectrum.update(sr, hr)
    return spectrum


def test_equal_images_give_ratio_one() -> None:
    hr = faces()
    spectrum = measure(hr, hr)
    assert spectrum.peak_ratio() == pytest.approx(1.0)
    assert all(band["ratio"] == pytest.approx(1.0) for band in spectrum.band_ratios())


def test_smoothing_stays_below_one() -> None:
    hr = faces() + 0.05 * torch.rand(6, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(1))
    sr = F.avg_pool2d(F.pad(hr, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    spectrum = measure(sr, hr)
    assert spectrum.peak_ratio() < 1.0
    high = [band["ratio"] for band in spectrum.band_ratios() if band["from"] >= MIN_RADIUS]
    assert max(high) < 0.5


def test_a_periodic_pattern_gives_a_peak() -> None:
    hr = faces()
    rows, cols = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    checkerboard = ((rows + cols) % 2).float() * 2 - 1  # period 2 px on both axes
    spectrum = measure(hr + 0.01 * checkerboard, hr)
    assert spectrum.peak_ratio() > 100
    bands = spectrum.band_ratios()
    assert bands[0]["ratio"] == pytest.approx(1.0, rel=0.01)  # low frequencies untouched
    assert bands[-1]["ratio"] > 100  # (0.5, 0.5): the corner band
    ratio = spectrum.ratio()
    assert ratio[SIZE // 2, SIZE // 2] == ratio.max()  # the checkerboard frequency


def test_batches_accumulate() -> None:
    sr, hr = faces(seed=2), faces(seed=3)
    whole = measure(sr, hr)
    parts = SpectrumRatio(SIZE, SIZE)
    parts.update(sr[:2], hr[:2])
    parts.update(sr[2:], hr[2:])
    assert parts.images == whole.images == 6
    assert torch.allclose(parts.sr, whole.sr) and torch.allclose(parts.hr, whole.hr)


def test_summary_and_errors() -> None:
    summary = measure(faces(seed=4), faces(seed=5)).summary()
    assert summary["images"] == 6 and summary["min_radius"] == MIN_RADIUS
    assert summary["peak_ratio"] > 0
    assert [band["from"] for band in summary["bands"]] == list(BAND_EDGES[:-1])
    assert summary["bands"][-1]["to"] is None
    spectrum = SpectrumRatio(SIZE, SIZE)
    with pytest.raises(ValueError, match="no images"):
        spectrum.peak_ratio()
    with pytest.raises(ValueError, match="expected"):
        spectrum.update(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))
    with pytest.raises(ValueError, match="differ"):
        spectrum.update(torch.zeros(1, 3, SIZE, SIZE), torch.zeros(2, 3, SIZE, SIZE))
