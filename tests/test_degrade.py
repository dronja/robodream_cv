"""Tests for facehall.data.degrade."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from facehall.config import load_config
from facehall.data.config import (
    BlurConfig,
    ColorShiftConfig,
    DegradeConfig,
    JpegConfig,
    NoiseConfig,
    ResizeConfig,
    SecondOrderConfig,
    StageConfig,
)
from facehall.data.degrade import (
    RESIZE_MODES,
    Bicubic,
    Pipeline,
    build_degradation,
    choose,
    draw_blur,
    draw_resize,
    jpeg,
    resize,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

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
    lr = Bicubic()(smooth_image(), scale, RNG).lr
    assert lr.shape == (3, 128 // scale, 128 // scale)
    assert lr.dtype == torch.float32
    assert lr.min() >= 0.0 and lr.max() <= 1.0
    torch.testing.assert_close(lr * 255, torch.round(lr * 255), rtol=0, atol=1e-4)


def test_constant_image_stays_constant() -> None:
    hr = torch.full((3, 64, 64), 0.4)
    lr = Bicubic(quantize=False)(hr, 4, RNG).lr
    torch.testing.assert_close(lr, torch.full((3, 16, 16), 0.4))


def test_without_quantization_values_leave_the_8_bit_grid() -> None:
    lr = Bicubic(quantize=False)(smooth_image(), 4, RNG).lr
    assert not torch.allclose(lr * 255, torch.round(lr * 255), atol=1e-3)


@pytest.mark.parametrize("scale", [4, 8])
def test_bicubic_matches_pil_within_one_level(scale: int) -> None:
    hr = smooth_image()
    lr = Bicubic(quantize=False)(hr, scale, RNG).lr
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


def pipeline(**fields: Any) -> DegradeConfig:
    return DegradeConfig(kind="pipeline", **fields)


def constant(value: float = 0.5, size: int = 64) -> torch.Tensor:
    return torch.full((3, size, size), value)


def on_8_bit_grid(image: torch.Tensor) -> bool:
    return bool(torch.allclose(image * 255, torch.round(image * 255), rtol=0, atol=1e-4))


@pytest.mark.parametrize("scale", [4, 8])
def test_default_pipeline_is_the_bicubic_downscale(scale: int) -> None:
    hr = smooth_image()
    result = Pipeline(pipeline())(hr, scale, np.random.default_rng(1))
    torch.testing.assert_close(result.lr, Bicubic()(hr, scale, RNG).lr, rtol=0, atol=0)
    assert result.hr is hr
    assert result.params == {
        "gray": False,
        "stages": [
            {
                "size": [128 // scale, 128 // scale],
                "ratio": float(scale),
                "blur": None,
                "resize": {"mode": "bicubic", "antialias": True},
                "noise": None,
                "jpeg": None,
            }
        ],
        "color_shift": None,
    }


@pytest.mark.parametrize("name", ["realistic", "realistic_x8"])
@pytest.mark.parametrize("scale", [4, 8])
def test_realistic_config_runs(scale: int, name: str) -> None:
    config = load_config(CONFIGS / "degrade" / f"{name}.yaml", DegradeConfig)
    degradation = build_degradation(config)
    hr = smooth_image()
    outputs = [degradation(hr, scale, np.random.default_rng(seed)) for seed in range(30)]
    for out in outputs:
        assert out.lr.shape == (3, 128 // scale, 128 // scale) and out.lr.dtype == torch.float32
        assert out.lr.min() >= 0.0 and out.lr.max() <= 1.0
        assert on_8_bit_grid(out.lr)
        assert out.hr.shape == hr.shape and on_8_bit_grid(out.hr)
        json.dumps(out.params)
    assert {len(out.params["stages"]) for out in outputs} == {1, 2}
    assert len({out.lr.numpy().tobytes() for out in outputs}) == len(outputs)


def test_pipeline_is_deterministic() -> None:
    config = load_config(CONFIGS / "degrade" / "realistic.yaml", DegradeConfig)
    degradation = Pipeline(config)
    hr = smooth_image()
    first = degradation(hr, 4, np.random.default_rng([3, 7]))
    again = degradation(hr, 4, np.random.default_rng([3, 7]))
    other = degradation(hr, 4, np.random.default_rng([3, 8]))
    torch.testing.assert_close(first.lr, again.lr, rtol=0, atol=0)
    assert first.params == again.params
    assert not torch.equal(first.lr, other.lr)


def test_blur_sizes_are_in_lr_pixels() -> None:
    config = BlurConfig(prob=1.0, gaussian=1.0, sigma=[1.5, 1.5])
    for ratio in (1.0, 4.0, 8.0):
        kernel, record = draw_blur(config, ratio, np.random.default_rng(0))
        assert record == {"kind": "gaussian", "sigma": 1.5}
        radius = kernel.shape[0] // 2
        x = np.arange(-radius, radius + 1)
        variance = float((kernel.sum(axis=0) * x**2).sum())
        assert variance == pytest.approx((1.5 * ratio) ** 2, rel=0.03)
    blur = StageConfig(blur=config)
    for scale in (4, 8):
        out = Pipeline(pipeline(first=blur))(smooth_image(), scale, np.random.default_rng(0))
        assert out.params["stages"][0]["ratio"] == scale


@pytest.mark.parametrize("kind", ["anisotropic", "motion", "defocus"])
def test_every_blur_kind_runs(kind: str) -> None:
    config = BlurConfig(
        prob=1.0,
        sigma=[0.5, 1.0],
        motion_length=[1.0, 2.0],
        defocus_radius=[0.5, 1.0],
        **{kind: 1.0},
    )
    out = Pipeline(pipeline(first=StageConfig(blur=config)))(
        smooth_image(), 4, np.random.default_rng(0)
    )
    assert out.params["stages"][0]["blur"]["kind"] == kind
    assert out.lr.shape == (3, 32, 32)


def noisy(noise: NoiseConfig, seed: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
    """Noise alone on a constant 0.5 image (scale 1: no resize), without quantization."""
    config = pipeline(first=StageConfig(noise=noise), quantize=False)
    out = Pipeline(config)(constant(), 1, np.random.default_rng(seed))
    return out.lr - 0.5, out.params["stages"][0]["noise"]


def test_gaussian_noise() -> None:
    noise, record = noisy(NoiseConfig(prob=1.0, gaussian=1.0, sigma=[10.0, 10.0]))
    assert record == {"kind": "gaussian", "sigma": 10.0, "gray": False}
    assert float(noise.std()) * 255 == pytest.approx(10.0, rel=0.05)
    assert not torch.equal(noise[0], noise[1])
    gray, record = noisy(NoiseConfig(prob=1.0, gaussian=1.0, sigma=[10.0, 10.0], gray=1.0))
    assert record["gray"] is True
    torch.testing.assert_close(gray[0], gray[1], rtol=0, atol=0)
    torch.testing.assert_close(gray[0], gray[2], rtol=0, atol=0)


def test_poisson_noise() -> None:
    noise, record = noisy(NoiseConfig(prob=1.0, poisson=1.0, peak=[200.0, 200.0]))
    assert record["kind"] == "poisson" and record["peak"] == pytest.approx(200.0)
    assert float(noise.std()) == pytest.approx((0.5 / 200.0) ** 0.5, rel=0.05)
    assert abs(float(noise.mean())) < 0.003


def test_jpeg_stage() -> None:
    hr = smooth_image(64)
    config = pipeline(first=StageConfig(jpeg=JpegConfig(prob=1.0, quality=[10, 10])))
    out = Pipeline(config)(hr, 1, np.random.default_rng(0))
    assert out.params["stages"][0]["jpeg"] == {"quality": 10}
    assert on_8_bit_grid(out.lr)
    assert (out.lr - hr).abs().mean() > 1 / 255
    image = hr.permute(1, 2, 0).numpy()
    expected = torch.from_numpy(jpeg(image, 10, gray=False)).permute(2, 0, 1)
    torch.testing.assert_close(out.lr, expected, rtol=0, atol=0)
    better = torch.from_numpy(jpeg(image, 95, gray=False)).permute(2, 0, 1)
    assert (better - hr).abs().mean() < (out.lr - hr).abs().mean()


def test_grayscale_changes_the_target() -> None:
    noise = NoiseConfig(prob=1.0, gaussian=1.0, sigma=[8.0, 8.0])
    config = pipeline(
        grayscale=1.0,
        first=StageConfig(noise=noise, jpeg=JpegConfig(prob=1.0, quality=[50, 50])),
        color_shift=ColorShiftConfig(prob=1.0, shift=20.0),
    )
    degradation = Pipeline(config)
    out = degradation(smooth_image(), 4, np.random.default_rng(0))
    assert out.params["gray"] is True and out.params["color_shift"] is None
    assert out.params["stages"][0]["noise"]["gray"] is True
    for image in (out.hr, out.lr):
        assert on_8_bit_grid(image)
        torch.testing.assert_close(image[0], image[1], rtol=0, atol=0)
        torch.testing.assert_close(image[0], image[2], rtol=0, atol=0)
    assert degradation.describe()["changes_target"] is True
    assert Pipeline(pipeline()).describe()["changes_target"] is False


def test_second_order() -> None:
    second = SecondOrderConfig(prob=1.0, size=[1.5, 1.5])
    out = Pipeline(pipeline(second=second))(smooth_image(), 4, np.random.default_rng(0))
    first, last = out.params["stages"]
    assert (first["size"], first["ratio"]) == ([48, 48], 4.0)
    assert (last["size"], last["ratio"]) == ([32, 32], 1.5)
    assert out.lr.shape == (3, 32, 32)


def test_color_shift() -> None:
    config = pipeline(color_shift=ColorShiftConfig(prob=1.0, shift=20.0), quantize=False)
    out = Pipeline(config)(constant(), 1, np.random.default_rng(0))
    shift = out.params["color_shift"]
    assert len(shift) == 3 and all(abs(v) <= 20.0 for v in shift)
    for channel, value in enumerate(shift):
        torch.testing.assert_close(
            out.lr[channel], torch.full((64, 64), 0.5 + value / 255), rtol=0, atol=1e-6
        )
    torch.testing.assert_close(out.hr, constant())  # the target keeps its colours


@pytest.mark.parametrize("mode", RESIZE_MODES)
def test_resize_modes(mode: str) -> None:
    image = np.full((64, 64, 3), 0.25, dtype=np.float32)
    for size in ((16, 16), (24, 24), (96, 96)):
        out = resize(image, size, mode, antialias=mode in ("bicubic", "bilinear"))
        assert out.shape == (*size, 3)
        np.testing.assert_allclose(out, 0.25, atol=1e-6)
    weights = {name: float(name == mode) for name in RESIZE_MODES}
    drawn, antialias = draw_resize(ResizeConfig(**weights, antialias=1.0), RNG)
    assert drawn == mode
    assert antialias is (mode in ("bicubic", "bilinear"))


def test_choose_follows_the_weights() -> None:
    rng = np.random.default_rng(0)
    assert {choose(rng, {"a": 1.0, "b": 0.0}) for _ in range(50)} == {"a"}
    draws = [choose(rng, {"a": 1.0, "b": 3.0}) for _ in range(4000)]
    assert draws.count("b") / len(draws) == pytest.approx(0.75, abs=0.03)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"grayscale": 1.5}, "grayscale must be a probability"),
        ({"first": StageConfig(blur=BlurConfig(prob=0.5))}, "first.blur needs a positive weight"),
        (
            {"first": StageConfig(resize=ResizeConfig(bicubic=0.0))},
            "first.resize needs a positive weight",
        ),
        (
            {"first": StageConfig(blur=BlurConfig(sigma=[2.0, 1.0]))},
            r"first.blur.sigma must be \[low, high\]",
        ),
        (
            {"first": StageConfig(noise=NoiseConfig(gaussian=-1.0))},
            "first.noise weights must be non-negative",
        ),
        (
            {"second": SecondOrderConfig(stage=StageConfig(jpeg=JpegConfig(quality=[0, 50])))},
            r"second.stage.jpeg.quality must lie in \[1, 100\]",
        ),
        ({"second": SecondOrderConfig(size=[0.0, 1.0])}, "second.size must be positive"),
        ({"color_shift": ColorShiftConfig(shift=-1.0)}, "color_shift.shift must be non-negative"),
    ],
)
def test_invalid_pipeline_settings(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_degradation(pipeline(**fields))


def test_repository_realistic_config_is_described() -> None:
    config = load_config(CONFIGS / "degrade" / "realistic.yaml", DegradeConfig)
    description = build_degradation(config).describe()
    assert description["kind"] == "pipeline"
    assert description["config"] == asdict(config)
    assert description["changes_target"] is True  # grayscale > 0
    assert {"torch", "opencv", "numpy", "implementation"} <= set(description)
