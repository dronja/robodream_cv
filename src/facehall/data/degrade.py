"""HR -> LR degradations.

- ``Bicubic``: degradation v0 (PR 1.4), a plain bicubic downscale with antialias.
- ``Pipeline`` (PR 2.1): a random degradation drawn per sample from a config
  (``configs/degrade/*.yaml``, ``facehall.data.config.DegradeConfig``):

  1. with probability ``grayscale``, the HR image is turned grey, and so is the HR target;
  2. stage ``first``: blur -> resize -> noise -> JPEG;
  3. with probability ``second.prob``, stage ``second.stage`` (Real-ESRGAN-style second order,
     arXiv:2107.10833): the first stage then ends at ``second.size`` times the LR size, and the
     second one resizes to the LR size;
  4. with probability ``color_shift.prob``, a per-channel offset of the LR image;
  5. clamp to [0, 1] and, with ``quantize``, round to 8-bit levels.

  Blur sizes are in LR pixels and are scaled to the working resolution of each stage, so one
  config gives the same blur in the LR image at x4 and x8. Noise is in 8-bit levels at the
  working resolution. JPEG uses OpenCV's encoder (4:2:0 chroma subsampling, its default); grey
  images are encoded as single-channel JPEG.

A degradation draws only from the per-sample generator it receives, so validation and test stay
reproducible, and it returns the parameters it drew (``Degraded.params``).
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from facehall.config import require_choice
from facehall.data import kernels
from facehall.data.config import (
    BlurConfig,
    DegradeConfig,
    JpegConfig,
    NoiseConfig,
    ResizeConfig,
    StageConfig,
)

KINDS = ("bicubic", "pipeline")
BLUR_KINDS = ("gaussian", "anisotropic", "motion", "defocus")
RESIZE_MODES = ("bicubic", "bilinear", "area", "nearest")
NOISE_KINDS = ("gaussian", "poisson")
# torch modes; "nearest-exact" samples pixel centres, like PIL and OpenCV.
TORCH_MODES = {
    "bicubic": "bicubic",
    "bilinear": "bilinear",
    "area": "area",
    "nearest": "nearest-exact",
}
ANTIALIASED = ("bicubic", "bilinear")  # modes with an antialias option in torch
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601

Image = npt.NDArray[np.float32]  # (H, W, 3) in [0, 1]


@dataclass(frozen=True)
class Degraded:
    """The output of a degradation.

    ``hr`` is the training target: the input HR image, unless the degradation also changes it
    (grayscale). ``params`` holds the drawn parameters and is JSON-serialisable.
    """

    lr: torch.Tensor
    hr: torch.Tensor
    params: dict[str, Any] = field(default_factory=dict)


class Degradation(Protocol):
    def __call__(self, hr: torch.Tensor, scale: int, rng: np.random.Generator) -> Degraded:
        """Degrade a (C, H, W) float image in [0, 1] to (C, H / scale, W / scale)."""
        ...

    def describe(self) -> dict[str, object]:
        """Exact settings and implementation, recorded with every run."""
        ...


def lr_size(hr: torch.Tensor, scale: int) -> tuple[int, int]:
    if hr.ndim != 3:
        raise ValueError(f"expected a (C, H, W) image, got shape {tuple(hr.shape)}")
    _, height, width = hr.shape
    if scale < 1 or height % scale or width % scale:
        raise ValueError(f"scale {scale} does not divide the image size {height}x{width}")
    return height // scale, width // scale


@dataclass(frozen=True)
class Bicubic:
    """Bicubic downscale; deterministic, so ``rng`` is not used."""

    antialias: bool = True
    quantize: bool = True

    def __call__(self, hr: torch.Tensor, scale: int, rng: np.random.Generator) -> Degraded:
        lr = F.interpolate(
            hr[None],
            size=lr_size(hr, scale),
            mode="bicubic",
            antialias=self.antialias,
            align_corners=False,
        )[0].clamp(0.0, 1.0)
        if self.quantize:
            lr = torch.round(lr * 255.0) / 255.0
        return Degraded(lr=lr, hr=hr)

    def describe(self) -> dict[str, object]:
        return {
            "kind": "bicubic",
            "antialias": self.antialias,
            "quantize": self.quantize,
            "function": (
                "torch.nn.functional.interpolate(mode='bicubic', "
                f"antialias={self.antialias}, align_corners=False), clamped to [0, 1]"
            ),
            "torch": torch.__version__,
        }


class Pipeline:
    """Random blur, resize, noise and JPEG per sample (see the module docstring)."""

    def __init__(self, config: DegradeConfig) -> None:
        validate_degrade(config)
        self.config = config

    def __call__(self, hr: torch.Tensor, scale: int, rng: np.random.Generator) -> Degraded:
        config = self.config
        size = lr_size(hr, scale)
        image: Image = np.array(hr.permute(1, 2, 0).numpy(), dtype=np.float32)
        target = hr
        gray = bool(rng.random() < config.grayscale)
        if gray:
            image = quantized(to_gray(image))
            target = to_tensor(image)

        stages: list[tuple[StageConfig, tuple[int, int]]] = [(config.first, size)]
        if rng.random() < config.second.prob:
            factor = uniform(rng, config.second.size)
            middle = (max(1, round(size[0] * factor)), max(1, round(size[1] * factor)))
            stages = [(config.first, middle), (config.second.stage, size)]
        records = []
        for stage, output in stages:
            ratio = image.shape[0] / size[0]  # working pixels per LR pixel
            image, record = run_stage(image, stage, output, ratio, gray, rng)
            records.append(record)

        shift = None
        if rng.random() < config.color_shift.prob and not gray:
            limit = config.color_shift.shift
            shift = [float(v) for v in rng.uniform(-limit, limit, 3)]
            image = image + np.asarray(shift, dtype=np.float32) / 255.0
        image = np.clip(image, 0.0, 1.0)
        if gray:
            image = to_gray(image)
        if config.quantize:
            image = quantized(image)
        params = {"gray": gray, "stages": records, "color_shift": shift}
        return Degraded(lr=to_tensor(image), hr=target, params=params)

    def describe(self) -> dict[str, object]:
        return {
            "kind": "pipeline",
            "config": asdict(self.config),
            "changes_target": self.config.grayscale > 0.0,
            "implementation": {
                "blur": (
                    "facehall.data.kernels at the working resolution (sizes in LR px times "
                    "working px per LR px), cv2.filter2D with BORDER_REFLECT_101, float32"
                ),
                "resize": (
                    "torch.nn.functional.interpolate, align_corners=False for bicubic and "
                    "bilinear, 'nearest-exact' for nearest; clamped to [0, 1]"
                ),
                "noise": "numpy Generator normal / poisson; grey noise is computed on BT.601 luma",
                "jpeg": "cv2.imencode quality q (default 4:2:0 subsampling), grey as 1 channel",
            },
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        }


def run_stage(
    image: Image,
    stage: StageConfig,
    size: tuple[int, int],
    ratio: float,
    gray: bool,
    rng: np.random.Generator,
) -> tuple[Image, dict[str, Any]]:
    """One stage at a working resolution of ``ratio`` pixels per LR pixel, ending at ``size``."""
    record: dict[str, Any] = {"size": list(size), "ratio": ratio}
    record["blur"] = None
    if rng.random() < stage.blur.prob:
        kernel, record["blur"] = draw_blur(stage.blur, ratio, rng)
        image = np.asarray(
            cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101), dtype=np.float32
        )

    mode, antialias = draw_resize(stage.resize, rng)
    record["resize"] = {"mode": mode, "antialias": antialias}
    image = resize(image, size, mode, antialias)

    record["noise"] = None
    if rng.random() < stage.noise.prob:
        image, record["noise"] = add_noise(image, stage.noise, gray, rng)

    record["jpeg"] = None
    if rng.random() < stage.jpeg.prob:
        low, high = stage.jpeg.quality
        quality = int(rng.integers(low, high, endpoint=True))
        image = jpeg(image, quality, gray)
        record["jpeg"] = {"quality": quality}
    return image, record


def draw_blur(
    config: BlurConfig, ratio: float, rng: np.random.Generator
) -> tuple[kernels.Kernel, dict[str, Any]]:
    """A blur kernel for a working resolution of ``ratio`` pixels per LR pixel; the record
    gives the sizes in LR pixels."""
    kind = choose(rng, {name: getattr(config, name) for name in BLUR_KINDS})
    if kind == "gaussian":
        sigma = uniform(rng, config.sigma)
        return kernels.gaussian(sigma * ratio), {"kind": kind, "sigma": sigma}
    if kind == "anisotropic":
        sigma_x, sigma_y = uniform(rng, config.sigma), uniform(rng, config.sigma)
        angle = float(rng.uniform(0.0, math.pi))
        kernel = kernels.gaussian(sigma_x * ratio, sigma_y * ratio, angle)
        return kernel, {"kind": kind, "sigma": [sigma_x, sigma_y], "angle": angle}
    if kind == "motion":
        length = uniform(rng, config.motion_length)
        angle = float(rng.uniform(0.0, math.pi))
        return kernels.motion(length * ratio, angle), {
            "kind": kind,
            "length": length,
            "angle": angle,
        }
    radius = uniform(rng, config.defocus_radius)
    return kernels.defocus(radius * ratio), {"kind": kind, "radius": radius}


def draw_resize(config: ResizeConfig, rng: np.random.Generator) -> tuple[str, bool]:
    mode = choose(rng, {name: getattr(config, name) for name in RESIZE_MODES})
    antialias = bool(rng.random() < config.antialias) and mode in ANTIALIASED
    return mode, antialias


def resize(image: Image, size: tuple[int, int], mode: str, antialias: bool) -> Image:
    """Resize (H, W, 3) to ``size`` (height, width) with torch, clamped to [0, 1]."""
    if image.shape[:2] == size:
        return image
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]
    output = F.interpolate(
        tensor,
        size=size,
        mode=TORCH_MODES[mode],
        antialias=antialias,
        align_corners=False if mode in ANTIALIASED else None,
    )
    result: Image = np.ascontiguousarray(output[0].clamp(0.0, 1.0).permute(1, 2, 0).numpy())
    return result


def add_noise(
    image: Image, config: NoiseConfig, gray: bool, rng: np.random.Generator
) -> tuple[Image, dict[str, Any]]:
    kind = choose(rng, {name: getattr(config, name) for name in NOISE_KINDS})
    same = bool(rng.random() < config.gray) or gray
    height, width = image.shape[:2]
    if kind == "gaussian":
        sigma = uniform(rng, config.sigma)
        noise = rng.normal(0.0, sigma / 255.0, (height, width, 1 if same else 3))
        record: dict[str, Any] = {"kind": kind, "sigma": sigma, "gray": same}
    else:
        low, high = config.peak
        peak = math.exp(uniform(rng, [math.log(low), math.log(high)]))
        base = np.clip(luma(image)[..., None] if same else image, 0.0, 1.0)
        noise = rng.poisson(base * peak) / peak - base
        record = {"kind": kind, "peak": peak, "gray": same}
    noisy: Image = np.clip(image + noise.astype(np.float32), 0.0, 1.0)
    return noisy, record


def jpeg(image: Image, quality: int, gray: bool) -> Image:
    """JPEG round trip at ``quality``; grey images are encoded as one channel."""
    pixels = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    settings = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if gray:
        ok, data = cv2.imencode(".jpg", np.ascontiguousarray(pixels[..., 0]), settings)
        decoded = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if ok else None
        if decoded is not None:
            decoded = np.repeat(decoded[..., None], 3, axis=2)
    else:
        ok, data = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), settings)
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR) if ok else None
        if decoded is not None:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if decoded is None:
        raise RuntimeError(f"JPEG round trip failed for an image of shape {image.shape}")
    result: Image = decoded.astype(np.float32) / 255.0
    return result


def luma(image: Image) -> npt.NDArray[np.float32]:
    result: npt.NDArray[np.float32] = image @ LUMA
    return result


def to_gray(image: Image) -> Image:
    result: Image = np.repeat(luma(image)[..., None], 3, axis=2)
    return result


def quantized(image: Image) -> Image:
    result: Image = (np.round(image * 255.0) / 255.0).astype(np.float32)
    return result


def to_tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1)


def uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    low, high = bounds
    return float(rng.uniform(low, high))


def choose(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """One name, drawn with probability proportional to its weight."""
    names = list(weights)
    values = np.asarray([weights[name] for name in names], dtype=np.float64)
    return names[int(rng.choice(len(names), p=values / values.sum()))]


def validate_degrade(config: DegradeConfig) -> None:
    """Raise ``ValueError`` for settings the pipeline cannot use."""
    require_choice(config.kind, KINDS, "degradation kind")
    if config.kind != "pipeline":
        return
    probability("grayscale", config.grayscale)
    probability("color_shift.prob", config.color_shift.prob)
    if config.color_shift.shift < 0:
        raise ValueError(f"color_shift.shift must be non-negative, got {config.color_shift.shift}")
    probability("second.prob", config.second.prob)
    bounds("second.size", config.second.size, positive=True)
    for name, stage in (("first", config.first), ("second.stage", config.second.stage)):
        validate_stage(name, stage)


def validate_stage(name: str, stage: StageConfig) -> None:
    blur, noise, jpeg_config = stage.blur, stage.noise, stage.jpeg
    probability(f"{name}.blur.prob", blur.prob)
    weights(f"{name}.blur", {k: getattr(blur, k) for k in BLUR_KINDS}, needed=blur.prob > 0)
    for key in ("sigma", "motion_length", "defocus_radius"):
        bounds(f"{name}.blur.{key}", getattr(blur, key))
    weights(f"{name}.resize", {k: getattr(stage.resize, k) for k in RESIZE_MODES}, needed=True)
    probability(f"{name}.resize.antialias", stage.resize.antialias)
    probability(f"{name}.noise.prob", noise.prob)
    weights(f"{name}.noise", {k: getattr(noise, k) for k in NOISE_KINDS}, needed=noise.prob > 0)
    bounds(f"{name}.noise.sigma", noise.sigma)
    bounds(f"{name}.noise.peak", noise.peak, positive=True)
    probability(f"{name}.noise.gray", noise.gray)
    validate_jpeg(f"{name}.jpeg", jpeg_config)


def validate_jpeg(name: str, config: JpegConfig) -> None:
    probability(f"{name}.prob", config.prob)
    bounds(f"{name}.quality", [float(q) for q in config.quality])
    low, high = config.quality
    if low < 1 or high > 100:
        raise ValueError(f"{name}.quality must lie in [1, 100], got {list(config.quality)}")


def probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value}")


def bounds(name: str, values: list[float], *, positive: bool = False) -> None:
    if len(values) != 2 or not 0.0 <= values[0] <= values[1]:
        raise ValueError(f"{name} must be [low, high] with 0 <= low <= high, got {list(values)}")
    if positive and values[0] <= 0.0:
        raise ValueError(f"{name} must be positive, got {list(values)}")


def weights(name: str, values: dict[str, float], *, needed: bool) -> None:
    if any(v < 0 for v in values.values()):
        raise ValueError(f"{name} weights must be non-negative, got {values}")
    if needed and sum(values.values()) <= 0:
        raise ValueError(f"{name} needs a positive weight for at least one of {sorted(values)}")


def build_degradation(config: DegradeConfig) -> Degradation:
    validate_degrade(config)
    if config.kind == "bicubic":
        return Bicubic(antialias=config.antialias, quantize=config.quantize)
    return Pipeline(config)
