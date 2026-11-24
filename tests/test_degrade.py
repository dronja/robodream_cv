"""Tests for facehall.data.degrade."""

import numpy as np
import pytest
import torch
from PIL import Image

from facehall.data.config import DegradeConfig
from facehall.data.degrade import Bicubic, build_degradation

RNG = np.random.default_rng(0)


def smooth_image(size: int = 128) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    channels = [
        0.5 + 0.4 * torch.sin(x / 7.0) * torch.cos(y / 5.0),
        0.5 + 0.4 * torch.cos((x - y) / 9.0),
        0.5 + 0.45 * torch.sin(y / 3.0),
    ]
    image = torch.stack(channels).float()
    return torch.round(image * 255) / 255


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_bicubic_shape_range_and_levels(scale: int) -> None:
    lr = Bicubic()(smooth_image(), scale, RNG)
    assert lr.shape == (3, 128 // scale, 128 // scale)
    assert lr.dtype == torch.float32
    assert lr.min() >= 0.0 and lr.max() <= 1.0
    torch.testing.assert_close(lr * 255, torch.round(lr * 255), rtol=0, atol=1e-4)


def test_constant_image_stays_constant() -> None:
    hr = torch.full((3, 64, 64), 0.4)
    lr = Bicubic(quantize=False)(hr, 4, RNG)
    torch.testing.assert_close(lr, torch.full((3, 16, 16), 0.4))


def test_without_quantization_values_leave_the_8_bit_grid() -> None:
    lr = Bicubic(quantize=False)(smooth_image(), 4, RNG)
    assert not torch.allclose(lr * 255, torch.round(lr * 255), atol=1e-3)


@pytest.mark.parametrize("scale", [4, 8])
def test_bicubic_matches_pil_within_one_level(scale: int) -> None:
    hr = smooth_image()
    lr = Bicubic(quantize=False)(hr, scale, RNG)
    pixels = (hr * 255).round().byte().permute(1, 2, 0).numpy()
    size = 128 // scale
    pil = np.asarray(Image.fromarray(pixels).resize((size, size), Image.Resampling.BICUBIC))
    reference = torch.from_numpy(pil.copy()).permute(2, 0, 1).float()
    assert (lr * 255 - reference).abs().max() <= 1.0


def test_bad_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        Bicubic()(torch.zeros(3, 30, 30), 4, RNG)
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        Bicubic()(torch.zeros(1, 3, 32, 32), 4, RNG)
    with pytest.raises(ValueError, match="unknown degradation kind 'jpeg'"):
        build_degradation(DegradeConfig(kind="jpeg"))


def test_describe_records_the_function() -> None:
    description = build_degradation(DegradeConfig()).describe()
    assert description["kind"] == "bicubic"
    assert description["antialias"] is True
    assert "antialias=True" in str(description["function"])
    assert description["torch"] == torch.__version__
