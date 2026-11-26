"""Temporal consistency of the pipeline on synthetic CelebA clips (PR 4.5).

A clip shows an aligned CelebA image as a small LR scene over a number of frames. The face has
the model's input size, as in ``facehall.eval.detection.make_scene``. Conditions:

- ``static_degraded``: no motion, with each frame degraded anew (fresh noise and JPEG), like a
  camera filming a still face;
- ``drift_clean`` and ``drift_degraded``: the face moves by ``DRIFT`` LR px per frame, so its
  sub-pixel phase changes from frame to frame;
- ``sway_degraded``: the face sways sideways, ``SWAY`` LR px of amplitude over a period of
  ``SWAY_PERIOD`` frames (up to 0.79 LR px per frame), so its speed keeps changing; this shows
  the lag of a smoothing filter;
- ``fast_degraded``: the face moves by ``FAST`` LR px per frame, like a small face crossing the
  frame; a filter without a velocity lags behind it in proportion to the speed. Its scene is
  padded by three times the image size instead of once, so the face stays inside.

Every frame is rendered from the source image: the source is padded by its own size with its
mean colour, shifted (bicubic) at the source resolution and downscaled (area). So frames differ
only by the motion and the degradation. The HR reference of each frame is rendered the same way
at ×scale and aligned to the template by its own landmarks.

- **Flicker:** the face crops of consecutive frames, each aligned by the frame's true landmarks
  so that the motion cancels, are compared inside the paste-back mask. The measure is the mean
  absolute luma difference in 8-bit levels, averaged over frame pairs. The HR clip's flicker is
  the floor that rendering and cropping leave.
- **Accuracy:** PSNR-Y of each frame's face crop against the frame's HR crop.
- **Restored:** the fraction of frames in which the face (landmark NME below 0.5, as in D31) was
  restored.
- **Variants:** ``bicubic``; ``annotated`` (the pipeline with the true landmarks of every
  frame); ``detected`` (YuNet in every frame); ``track:<alpha>,<beta>,<hold>`` (YuNet and
  ``facehall.video.FaceTracker``).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from facehall.align.template import CELEBA_TEMPLATE
from facehall.align.warp import alignment_matrix, warp
from facehall.eval.detection import (
    SceneDegradation,
    degrade,
    interocular,
    landmark_error,
    load_rgb,
)
from facehall.eval.endtoend import face_crop
from facehall.pipeline import Enhanced, Enhancer, MaskConfig, ellipse_mask, upscale
from facehall.video import FaceTracker

Image = npt.NDArray[np.uint8]
DRIFT = (0.37, 0.13)  # LR px per frame: the sub-pixel phase keeps changing
SWAY, SWAY_PERIOD = 2.0, 16  # LR px amplitude, frames
FAST = (1.5, 0.5)  # LR px per frame
CONDITIONS = ("static_degraded", "drift_clean", "drift_degraded", "sway_degraded", "fast_degraded")


def motion(condition: str, t: int) -> tuple[float, float]:
    """The face's offset in frame ``t``, LR px."""
    if condition.startswith("drift"):
        return DRIFT[0] * t, DRIFT[1] * t
    if condition.startswith("fast"):
        return FAST[0] * t, FAST[1] * t
    if condition.startswith("sway"):
        return SWAY * float(np.sin(2.0 * np.pi * t / SWAY_PERIOD)), 0.0
    return 0.0, 0.0


@dataclass(frozen=True)
class Tracking:
    """A ``FaceTracker`` setting: alpha-beta filter and the frames a missed face is held."""

    alpha: float
    beta: float
    hold: int

    @property
    def name(self) -> str:
        return f"track:{self.alpha:g},{self.beta:g},{self.hold}"

    def tracker(self) -> FaceTracker:
        return FaceTracker(self.alpha, self.beta, hold=self.hold, keep=max(2, self.hold))

    @classmethod
    def parse(cls, text: str) -> "Tracking":
        """``alpha,beta,hold``, e.g. ``0.5,0.1,2``."""
        try:
            alpha, beta, hold = text.split(",")
            return cls(float(alpha), float(beta), int(hold))
        except ValueError:
            raise ValueError(f"expected alpha,beta,hold, got {text!r}") from None


@dataclass(frozen=True)
class Clip:
    frames: list[Image]  # LR scenes
    truth: list[npt.NDArray[np.float64]]  # (5, 2) true landmarks per frame, LR scene px
    hr: list[Image]  # the HR face crop of every frame (hr_size, CELEBA_TEMPLATE)


def render(
    image: Image,
    points: npt.NDArray[np.float64],
    factor: float,
    offset: tuple[float, float],
    pad: int = 1,
) -> tuple[Image, npt.NDArray[np.float64]]:
    """A scene of ``image`` padded by ``pad`` times its own size on each side (mean colour),
    scaled by ``factor`` (area) and with the content moved by ``offset`` px of the scene;
    returns the scene and the landmarks in it."""
    height, width = image.shape[:2]
    fill = tuple(float(v) for v in image.reshape(-1, 3).mean(axis=0))
    px, py = pad * width, pad * height
    padded = cv2.copyMakeBorder(image, py, py, px, px, cv2.BORDER_CONSTANT, value=fill)
    full = ((2 * pad + 1) * width, (2 * pad + 1) * height)
    size = (max(1, round(full[0] * factor)), max(1, round(full[1] * factor)))
    scale = np.array([size[0] / full[0], size[1] / full[1]])
    shift = np.asarray(offset, dtype=np.float64) / scale  # source px
    moved = cv2.warpAffine(
        padded,
        np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]]),
        full,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    scene = np.asarray(cv2.resize(moved, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
    truth = (points + np.array([px, py]) + shift + 0.5) * scale - 0.5
    return scene, truth


def make_clip(
    image: Image,
    points: npt.ArrayLike,
    *,
    scale: int,
    hr_size: int,
    frames: int,
    condition: str,
    rng: np.random.Generator,
) -> Clip:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    source = np.asarray(points, dtype=np.float64)
    eyes = float(CELEBA_TEMPLATE[1, 0] - CELEBA_TEMPLATE[0, 0]) * hr_size / scale
    factor = eyes / interocular(source)
    degradation = SceneDegradation() if condition.endswith("degraded") else None
    pad = 3 if condition.startswith("fast") else 1  # room for the face to travel
    lr: list[Image] = []
    truth: list[npt.NDArray[np.float64]] = []
    hr: list[Image] = []
    for t in range(frames):
        offset = motion(condition, t)
        scene, landmarks = render(image, source, factor, offset, pad)
        if degradation is not None:
            scene = degrade(scene, degradation, rng)
        hr_offset = (offset[0] * scale, offset[1] * scale)
        hr_scene, hr_points = render(image, source, factor * scale, hr_offset, pad)
        matrix = alignment_matrix(hr_points, CELEBA_TEMPLATE, hr_size)
        hr.append(np.asarray(warp(hr_scene, matrix, hr_size, border="reflect"), dtype=np.uint8))
        lr.append(scene)
        truth.append(landmarks)
    return Clip(lr, truth, hr)


def luma(image: Image) -> npt.NDArray[np.float64]:
    rgb = image.astype(np.float64)
    out: npt.NDArray[np.float64] = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return out


def flicker(crops: Sequence[Image], weights: npt.NDArray[np.float32]) -> float:
    """Mean absolute luma difference of consecutive crops, weighted by ``weights``."""
    if len(crops) < 2:
        raise ValueError("flicker needs at least two frames")
    values = [
        float((np.abs(luma(b) - luma(a)) * weights).sum() / weights.sum())
        for a, b in zip(crops, crops[1:], strict=False)
    ]
    return float(np.mean(values))


def psnr_luma(a: Image, b: Image) -> float:
    mse = float(np.mean((luma(a) - luma(b)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def face_restored(result: Enhanced, truth: npt.NDArray[np.float64]) -> bool:
    return any(f.restored and landmark_error(f.landmarks, truth) < 0.5 for f in result.faces)


def evaluate_clip(
    enhancer: Enhancer,
    clip: Clip,
    tracking: Sequence[Tracking] = (),
    mask: MaskConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Flicker, PSNR-Y and the restored fraction per variant for one clip."""
    scale, size = enhancer.scale, enhancer.hr_size
    weights = ellipse_mask(size, mask or MaskConfig())
    trackers = {t.name: t.tracker() for t in tracking}
    names = ["bicubic", "annotated", "detected", *trackers]
    crops: dict[str, list[Image]] = {"hr": list(clip.hr), **{n: [] for n in names}}
    restored = dict.fromkeys(["annotated", "detected", *trackers], 0)
    for frame, truth in zip(clip.frames, clip.truth, strict=True):
        crops["bicubic"].append(face_crop(upscale(frame, scale), truth, scale, size))
        results: dict[str, Enhanced] = {"annotated": enhancer.enhance(frame, [truth])}
        faces = enhancer.detector.detect(frame)
        results["detected"] = enhancer.enhance(frame, detections=faces)
        for name, tracker in trackers.items():
            results[name] = enhancer.enhance(frame, detections=tracker.update(faces))
        for name, result in results.items():
            crops[name].append(face_crop(result.image, truth, scale, size))
            restored[name] += face_restored(result, truth)
    out: dict[str, dict[str, float]] = {}
    for name, series in crops.items():
        out[name] = {"flicker": flicker(series, weights)}
        if name != "hr":
            out[name]["psnr_y"] = float(
                np.mean([psnr_luma(c, h) for c, h in zip(series, clip.hr, strict=True)])
            )
        if name in restored:
            out[name]["restored"] = restored[name] / len(clip.frames)
    return out


def clips(
    rows: Sequence[tuple[Any, npt.ArrayLike]],
    *,
    scale: int,
    hr_size: int,
    frames: int,
    condition: str,
    seed: int,
) -> Iterator[Clip]:
    rng = np.random.default_rng(seed)
    for path, points in rows:
        yield make_clip(
            load_rgb(path),
            points,
            scale=scale,
            hr_size=hr_size,
            frames=frames,
            condition=condition,
            rng=rng,
        )


def mean_results(results: Sequence[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """The mean of every value over clips."""
    if not results:
        raise ValueError("no clips")
    return {
        variant: {
            key: float(np.mean([r[variant][key] for r in results])) for key in results[0][variant]
        }
        for variant in results[0]
    }
