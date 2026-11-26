"""End-to-end face restoration in a photo (Phase 4, PR 4.2).

    image → bicubic background ×scale → faces (YuNet on a ×scale upscale, D31, or landmarks given
    by the user) → aligned LR crop per face → restoration model → feathered-ellipse mask → inverse
    warp and paste into the background → output ×scale

- **Alignment.** Detected landmarks are aligned to ``YUNET_TEMPLATES[scale]``, given landmarks
  (CelebA convention: eye centres, nose tip, mouth corners) to ``CELEBA_TEMPLATE``. Both give the
  crops the models were trained on, up to the landmark error.
- **Face size.** The LR crop has the model's input size (32 px at ×4, 16 px at ×8, the face at
  8 or 4 px between the eyes). A face larger than that is downscaled into it with area
  averaging; one more than ``max_downscale`` times larger is left to the background, because its
  own pixels hold more detail than the model could restore from the downscaled crop.
- **Paste-back.** The restored crop and its mask are warped back into a window around the face
  only, and blended there: ``out = mask * face + (1 - mask) * background``. Faces are pasted in
  detection order.
- **Colour correction** (optional, off by default) shifts the per-channel mean of the restored
  face, inside the mask, to that of the bicubic upscale of its LR crop.

The output is a plausible reconstruction, not a recovery of the true details (``CLAUDE.md``
§11); marking the output is PR 4.3.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from facehall.align.detect import Face, YuNet
from facehall.align.similarity import (
    Matrix,
    Points,
    compose,
    invert,
    scale_of,
    transform_points,
    translation,
)
from facehall.align.template import CELEBA_TEMPLATE, YUNET_TEMPLATES
from facehall.align.warp import alignment_matrix, resample_crop, unwarp
from facehall.weights import yunet_weights

Image = npt.NDArray[np.uint8]  # (H, W, 3) RGB


class Detector(Protocol):
    """Faces of an RGB image, with five landmarks in its pixel-centre coordinates (``YuNet``)."""

    def detect(self, image: Image) -> list[Face]: ...


@dataclass
class DetectorConfig:
    weights: str = ""  # empty: facehall.weights.yunet_weights() ($FACEHALL_WEIGHTS)
    score_threshold: float = 0.6  # D31
    nms_threshold: float = 0.3
    top_k: int = 5000


@dataclass
class MaskConfig:
    """Feathered ellipse in normalized crop coordinates (the crop is [0, 1] x [0, 1])."""

    center_x: float = 0.5
    center_y: float = 0.55
    radius_x: float = 0.42
    radius_y: float = 0.44
    feather: float = 0.25  # the outer fraction of the elliptic radius that fades to 0


@dataclass
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    max_downscale: float = 2.0  # larger faces are left to the background
    color_correction: bool = False
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.max_downscale < 1.0:
            raise ValueError("pipeline max_downscale must be at least 1")
        if self.batch_size < 1:
            raise ValueError("pipeline batch_size must be at least 1")
        m = self.mask
        if min(m.radius_x, m.radius_y) <= 0 or not 0.0 < m.feather <= 1.0:
            raise ValueError("pipeline mask radii must be positive and feather in (0, 1]")


@dataclass(frozen=True)
class FaceResult:
    """One face of the input: where it is, and whether it was restored."""

    landmarks: Points  # (5, 2), input pixel-centre coordinates
    source: str  # "detected" or "given"
    score: float | None  # detector score
    lr_scale: float  # LR crop pixels per input pixel (1: the face has the model's input size)
    restored: bool
    reason: str | None = None  # why it was not restored

    def to_dict(self) -> dict[str, Any]:
        return {
            "landmarks": np.round(self.landmarks, 2).tolist(),
            "source": self.source,
            "score": None if self.score is None else round(self.score, 4),
            "lr_scale": round(self.lr_scale, 4),
            "restored": self.restored,
            "reason": self.reason,
        }


@dataclass
class Enhanced:
    image: Image  # (scale * H, scale * W, 3)
    faces: list[FaceResult]
    scale: int


def ellipse_mask(size: int, config: MaskConfig) -> npt.NDArray[np.float32]:
    """(size, size) weights in [0, 1]: 1 inside, a smoothstep fade over the outer ``feather``."""
    coords = (np.arange(size, dtype=np.float64) + 0.5) / size
    x = (coords[None, :] - config.center_x) / config.radius_x
    y = (coords[:, None] - config.center_y) / config.radius_y
    radius = np.sqrt(x**2 + y**2)
    t = np.clip((1.0 - radius) / config.feather, 0.0, 1.0)
    return np.asarray(t * t * (3.0 - 2.0 * t), dtype=np.float32)


def upscale(image: Image, scale: int) -> Image:
    """Bicubic background: exactly ``scale`` times the input size, pixel centres aligned."""
    height, width = image.shape[:2]
    out = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    return np.asarray(out, dtype=np.uint8)


def given_points(points: npt.ArrayLike) -> Points:
    """Five (x, y) landmarks given by the user, as a (5, 2) float array."""
    array = np.asarray(points, dtype=np.float64)
    if array.size != 10:
        raise ValueError(f"expected five (x, y) landmarks, got shape {array.shape}")
    out: Points = array.reshape(5, 2)
    return out


def to_frame(points: npt.ArrayLike, scale: int) -> Points:
    """Input pixel-centre coordinates to those of the ×``scale`` frame."""
    out: Points = (np.asarray(points, dtype=np.float64) + 0.5) * scale - 0.5
    return out


def lr_crop(image: Image, landmarks: Points, template: Points, size: int) -> tuple[Image, float]:
    """The aligned ``size`` crop of ``image`` and its scale (crop px per input px), cut by
    ``facehall.align.resample_crop``: bicubic for a face at or below the crop scale, an integer
    supersampling and box average for a larger one."""
    matrix = alignment_matrix(landmarks, template, size)
    crop = resample_crop(image, matrix, size, border="reflect")
    return np.asarray(crop, dtype=np.uint8), scale_of(matrix)


def paste(
    frame: npt.NDArray[np.float32],
    face: npt.NDArray[np.float32],
    mask: npt.NDArray[np.float32],
    matrix: Matrix,
) -> None:
    """Blend ``face`` (crop, float RGB) into ``frame`` in place; ``matrix`` maps frame to crop."""
    height, width = frame.shape[:2]
    side = face.shape[0]
    corners = np.array(
        [[-0.5, -0.5], [side - 0.5, -0.5], [-0.5, side - 0.5], [side - 0.5, side - 0.5]]
    )
    spread = transform_points(invert(matrix), corners)
    x0, y0 = np.floor(spread.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(spread.max(axis=0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x0 >= x1 or y0 >= y1:
        return
    window = compose(matrix, translation(float(x0), float(y0)))  # window pixel -> crop pixel
    size = (int(x1 - x0), int(y1 - y0))
    pixels = unwarp(face, window, size, interpolation="bicubic", border="constant")
    alpha = unwarp(mask, window, size, interpolation="bilinear", border="constant")
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    region = frame[y0:y1, x0:x1]
    region += alpha * (np.clip(pixels, 0.0, 255.0) - region)


class Enhancer:
    """Restores the faces of RGB uint8 photos with one restoration model."""

    def __init__(
        self,
        model: nn.Module,
        scale: int,
        config: PipelineConfig,
        *,
        hr_size: int = 128,
        device: torch.device | str = "cpu",
        detector: Detector | None = None,
    ) -> None:
        if hr_size % scale:
            raise ValueError(f"HR size {hr_size} is not a multiple of scale {scale}")
        self.model = model.to(device).eval()
        self.scale, self.hr_size, self.lr_size = scale, hr_size, hr_size // scale
        self.config = config
        self.device = torch.device(device)
        self.mask = ellipse_mask(hr_size, config.mask)
        self._detector = detector

    @classmethod
    def from_checkpoint(
        cls, path: Path, config: PipelineConfig, device: torch.device | str = "cpu"
    ) -> "Enhancer":
        from facehall.train.checkpoint import load_checkpoint
        from facehall.train.config import experiment_from_dict

        model, info = load_checkpoint(path, device)
        experiment = experiment_from_dict(info["config"])
        return cls(model, experiment.scale, config, hr_size=experiment.data.hr_size, device=device)

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            d = self.config.detector
            self._detector = YuNet(
                d.weights or yunet_weights(),
                upsample=self.scale,
                score_threshold=d.score_threshold,
                nms_threshold=d.nms_threshold,
                top_k=d.top_k,
            )
        return self._detector

    def enhance(self, image: Image, landmarks: Sequence[npt.ArrayLike] | None = None) -> Enhanced:
        """Restore the faces of ``image``: detected, or at the given landmarks (input pixels)."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"expected an RGB uint8 image (H, W, 3), got {image.shape} {image.dtype}"
            )
        found: list[tuple[Points, str, float | None]]
        if landmarks is None:
            if self.scale not in YUNET_TEMPLATES:
                raise ValueError(f"no YuNet template for scale {self.scale}; give landmarks")
            found = [(f.landmarks, "detected", f.score) for f in self.detector.detect(image)]
            template = YUNET_TEMPLATES[self.scale]
        else:
            found = [(given_points(p), "given", None) for p in landmarks]
            template = CELEBA_TEMPLATE

        faces: list[FaceResult] = []
        crops: list[Image] = []
        for points, source, score in found:
            crop, factor = lr_crop(image, points, template, self.lr_size)
            if factor * self.config.max_downscale < 1.0:
                reason = (
                    f"the face is {1.0 / factor:.1f} times the model input size "
                    f"(limit {self.config.max_downscale:g}); left to the background"
                )
                faces.append(FaceResult(points, source, score, factor, False, reason))
                continue
            faces.append(FaceResult(points, source, score, factor, True))
            crops.append(crop)

        frame = upscale(image, self.scale).astype(np.float32)
        restored = self.restore(crops)
        index = 0
        for face in faces:
            if not face.restored:
                continue
            matrix = alignment_matrix(to_frame(face.landmarks, self.scale), template, self.hr_size)
            paste(frame, restored[index], self.mask, matrix)
            index += 1
        out = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
        return Enhanced(out, faces, self.scale)

    @torch.no_grad()
    def restore(self, crops: Sequence[Image]) -> list[npt.NDArray[np.float32]]:
        """Restored HR crops (float RGB in [0, 255]) of LR crops."""
        out: list[npt.NDArray[np.float32]] = []
        for start in range(0, len(crops), self.config.batch_size):
            batch = np.stack(crops[start : start + self.config.batch_size])
            lr = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
            sr = self.model(lr).float().clamp(0.0, 1.0)
            if self.config.color_correction:
                sr = self.correct_color(sr, lr)
            pixels = (sr * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
            out.extend(pixels)
        return out

    def correct_color(self, sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """Shift the mean colour of ``sr`` inside the mask to that of the bicubic LR upscale."""
        reference = torch.nn.functional.interpolate(
            lr, size=sr.shape[-2:], mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        weights = torch.from_numpy(self.mask).to(sr.device)[None, None]
        total = weights.sum()
        shift = ((reference - sr) * weights).sum(dim=(2, 3), keepdim=True) / total
        return (sr + shift).clamp(0.0, 1.0)
