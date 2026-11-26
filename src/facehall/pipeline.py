"""End-to-end face restoration in a photo (Phase 4, PR 4.2).

    image → bicubic background ×scale → faces (YuNet on a ×scale upscale, D31, or landmarks or
    face boxes given by the user) → aligned LR crop per face → restoration model →
    feathered-ellipse mask → inverse warp and paste into the background → output ×scale

- **Alignment.** Detected landmarks are aligned to ``YUNET_TEMPLATES[scale]``, given landmarks
  (CelebA convention: eye centres, nose tip, mouth corners) to ``CELEBA_TEMPLATE``. Both give the
  crops the models were trained on, up to the landmark error.
- **Face boxes** (PR 4.4) are a fallback for faces the detector misses. In each box, the
  highest-scoring face that YuNet finds at the lower ``box_score_threshold`` is used; without
  one, the landmarks come from the box itself (``YUNET_BOX_LANDMARKS``), so the face's rotation
  and proportions are guessed.
- **Face size.** The LR crop has the model's input size (32 px at ×4, 16 px at ×8, the face at
  8 or 4 px between the eyes). A face larger than that is downscaled into it with area
  averaging; one more than ``max_downscale`` times larger is left to the background, because its
  own pixels hold more detail than the model could restore from the downscaled crop. A face
  more than ``max_upscale`` times smaller is left too: the models were trained on faces of
  0.8–1.25 times their input size, and on real photos such tiny detections were false ones
  (PR 4.4).
- **Paste-back.** The restored crop and its mask are warped back into a window around the face
  only, and blended there: ``out = mask * face + (1 - mask) * background``. Faces are pasted in
  detection order.
- **Colour correction** (optional, off by default) shifts the per-channel mean of the restored
  face, inside the mask, to that of the bicubic upscale of its LR crop.

The output is a plausible reconstruction, not a recovery of the true details (``CLAUDE.md``
§11); ``facehall.marking`` marks it (PR 4.4).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from facehall.align.detect import Face, YuNet, box_landmarks
from facehall.align.similarity import (
    Matrix,
    Points,
    compose,
    invert,
    scale_of,
    transform_points,
    translation,
)
from facehall.align.template import CELEBA_TEMPLATE, YUNET_BOX_LANDMARKS, YUNET_TEMPLATES
from facehall.align.warp import alignment_matrix, resample_crop, unwarp
from facehall.weights import yunet_weights

Image = npt.NDArray[np.uint8]  # (H, W, 3) RGB


class Detector(Protocol):
    """Faces of an RGB image, with five landmarks in its pixel-centre coordinates (``YuNet``)."""

    def detect(self, image: Image) -> list[Face]: ...


@dataclass
class DetectorConfig:
    weights: str = ""  # empty: facehall.weights.yunet_weights() ($FACEHALL_WEIGHTS)
    score_threshold: float = 0.8  # D33 (D31: 0.6, too many false faces on real photos)
    box_score_threshold: float = 0.3  # inside a face box given by the user
    nms_threshold: float = 0.3
    top_k: int = 5000
    max_pixels: int = 16_000_000  # upscaled detector input per tile (OpenCV crashed at 49 MP)
    tile_overlap: int = 128  # input pixels shared by neighbouring tiles


@dataclass
class MaskConfig:
    """Feathered ellipse in normalized crop coordinates (the crop is [0, 1] x [0, 1])."""

    center_x: float = 0.5
    center_y: float = 0.55
    radius_x: float = 0.42
    radius_y: float = 0.44
    feather: float = 0.25  # the outer fraction of the elliptic radius that fades to 0


@dataclass
class ModelsConfig:
    """Restoration checkpoints per scale, for ``facehall enhance --scale`` without ``--ckpt``."""

    x4: str = ""
    x8: str = ""

    def checkpoint(self, scale: int) -> str:
        path = {4: self.x4, 8: self.x8}.get(scale, "")
        if not path:
            raise ValueError(f"no restoration model configured for scale {scale}")
        return path


@dataclass
class VideoConfig:
    """Video (``facehall.video``, PR 4.5): face tracks and encoding."""

    # Face tracks (chosen on synthetic clips, PR 4.5): an alpha-beta filter of each track's
    # landmarks and box, and the frames a missed face stays restored at its predicted place.
    smooth: bool = True
    alpha: float = 0.5
    beta: float = 0.1
    hold: int = 2
    track_iou: float = 0.3  # box IoU that continues a track
    # Detector score at which a detection continues a track; a new track needs
    # detector.score_threshold (D33). Equal to it: no hysteresis.
    track_threshold: float = 0.3
    codec: str = "libx264"  # libx264 or libx265 (system ffmpeg)
    crf: int = 18
    preset: str = "medium"
    audio: bool = True  # keep the input's audio (re-encoded to AAC in MP4 and MOV)

    def __post_init__(self) -> None:
        if self.codec not in ("libx264", "libx265"):
            raise ValueError(f"video codec must be libx264 or libx265, got {self.codec!r}")
        if not 0 <= self.crf <= 51:
            raise ValueError(f"video crf must be in [0, 51], got {self.crf}")
        if not 0 <= self.hold <= 25:
            raise ValueError(f"video hold must be in [0, 25] frames, got {self.hold}")
        if not 0.0 < self.track_threshold <= 1.0:
            raise ValueError(f"video track_threshold must be in (0, 1], got {self.track_threshold}")


@dataclass
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    max_downscale: float = 2.0  # larger faces are left to the background
    max_upscale: float = 2.0  # so are faces this many times smaller than the model input
    max_output_pixels: int = 200_000_000  # larger outputs are refused (memory: 3 bytes each)
    color_correction: bool = False
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.max_downscale < 1.0 or self.max_upscale < 1.0:
            raise ValueError("pipeline max_downscale and max_upscale must be at least 1")
        if self.batch_size < 1 or self.max_output_pixels < 1:
            raise ValueError("pipeline batch_size and max_output_pixels must be positive")
        m = self.mask
        if min(m.radius_x, m.radius_y) <= 0 or not 0.0 < m.feather <= 1.0:
            raise ValueError("pipeline mask radii must be positive and feather in (0, 1]")


@dataclass(frozen=True)
class FaceResult:
    """One face of the input: where it is, and whether it was restored."""

    landmarks: Points  # (5, 2), input pixel-centre coordinates
    source: str  # "detected", "given" (landmarks), "box-detected" or "box" (a face box)
    score: float | None  # detector score
    lr_scale: float  # LR crop pixels per input pixel (1: the face has the model's input size)
    restored: bool
    reason: str | None = None  # why it was not restored
    region: tuple[int, int, int, int] | None = None  # changed output pixels: x, y, width, height

    def to_dict(self) -> dict[str, Any]:
        return {
            "landmarks": np.round(self.landmarks, 2).tolist(),
            "source": self.source,
            "score": None if self.score is None else round(self.score, 4),
            "lr_scale": round(self.lr_scale, 4),
            "restored": self.restored,
            "reason": self.reason,
            "region": None if self.region is None else list(self.region),
        }


@dataclass
class Enhanced:
    image: Image  # (scale * H, scale * W, 3)
    faces: list[FaceResult]
    scale: int


Box = tuple[float, float, float, float]  # left, top, width, height (input pixels, box edges)
# A detection belongs to a given face box when its centre lies in the box and its size is within
# this factor of the box's: a default for hand-drawn boxes, not tuned.
BOX_SIZE_RATIO = 3.0


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
    frame: Image,
    face: npt.NDArray[np.float32],
    mask: npt.NDArray[np.float32],
    matrix: Matrix,
) -> tuple[int, int, int, int] | None:
    """Blend ``face`` (crop, float RGB) into ``frame`` (uint8) in place; ``matrix`` maps frame to
    crop. Only a window around the face is converted to float. Returns the window (x, y, width,
    height), or None when the face lies outside the frame."""
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
        return None
    window = compose(matrix, translation(float(x0), float(y0)))  # window pixel -> crop pixel
    size = (int(x1 - x0), int(y1 - y0))
    pixels = unwarp(face, window, size, interpolation="bicubic", border="constant")
    alpha = unwarp(mask, window, size, interpolation="bilinear", border="constant")
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    region = frame[y0:y1, x0:x1].astype(np.float32)
    region += alpha * (np.clip(pixels, 0.0, 255.0) - region)
    frame[y0:y1, x0:x1] = np.clip(np.rint(region), 0, 255).astype(np.uint8)
    return int(x0), int(y0), size[0], size[1]


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
        box_detector: Detector | None = None,
    ) -> None:
        if hr_size % scale:
            raise ValueError(f"HR size {hr_size} is not a multiple of scale {scale}")
        self.model = model.to(device).eval()
        self.scale, self.hr_size, self.lr_size = scale, hr_size, hr_size // scale
        self.config = config
        self.device = torch.device(device)
        self.mask = ellipse_mask(hr_size, config.mask)
        self._detector = detector
        self._box_detector = box_detector if box_detector is not None else detector
        self._track_detector = detector
        self.info: dict[str, Any] = {"scale": scale}  # model provenance for output marking

    @classmethod
    def from_checkpoint(
        cls, path: Path, config: PipelineConfig, device: torch.device | str = "cpu"
    ) -> "Enhancer":
        from facehall.train.checkpoint import load_checkpoint
        from facehall.train.config import experiment_from_dict

        model, info = load_checkpoint(path, device)
        experiment = experiment_from_dict(info["config"])
        enhancer = cls(
            model, experiment.scale, config, hr_size=experiment.data.hr_size, device=device
        )
        enhancer.info = {
            "name": info.get("run") or path.parent.name,
            "checkpoint": path.name,
            "sha256": info.get("sha256"),
            "architecture": experiment.model.name,
            "scale": experiment.scale,
        }
        return enhancer

    def _yunet(self, threshold: float) -> YuNet:
        d = self.config.detector
        return YuNet(
            d.weights or yunet_weights(),
            upsample=self.scale,
            score_threshold=threshold,
            nms_threshold=d.nms_threshold,
            top_k=d.top_k,
            max_pixels=d.max_pixels,
            overlap=d.tile_overlap,
        )

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = self._yunet(self.config.detector.score_threshold)
        return self._detector

    @property
    def box_detector(self) -> Detector:
        """The detector for face boxes, at the lower ``box_score_threshold``."""
        if self._box_detector is None:
            self._box_detector = self._yunet(self.config.detector.box_score_threshold)
        return self._box_detector

    @property
    def track_detector(self) -> Detector:
        """The detector for video face tracks, at ``video.track_threshold`` (or the pipeline's
        threshold, if that is lower)."""
        if self._track_detector is None:
            threshold = min(self.config.video.track_threshold, self.config.detector.score_threshold)
            self._track_detector = self._yunet(threshold)
        return self._track_detector

    def enhance(
        self,
        image: Image,
        landmarks: Sequence[npt.ArrayLike] | None = None,
        boxes: Sequence[Box] | None = None,
        detections: Sequence[Face] | None = None,
    ) -> Enhanced:
        """Restore the faces of ``image``: detected, or at the given landmarks, face boxes or
        detections (YuNet's convention, e.g. smoothed over video frames); input pixels."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"expected an RGB uint8 image (H, W, 3), got {image.shape} {image.dtype}"
            )
        height, width = image.shape[:2]
        pixels = height * width * self.scale**2
        if pixels > self.config.max_output_pixels:
            raise ValueError(
                f"the output would have {pixels / 1e6:.0f} MP at x{self.scale}, more than the "
                f"limit of {self.config.max_output_pixels / 1e6:.0f} MP (max_output_pixels)"
            )
        if sum(v is not None for v in (landmarks, boxes, detections)) > 1:
            raise ValueError("give landmarks, face boxes or detections, not more than one")
        found: list[tuple[Points, str, float | None, Points]] = []
        if detections is not None:
            template = self.yunet_template()
            found = [(f.landmarks, "detected", f.score, template) for f in detections]
        elif landmarks is not None:
            found = [(given_points(p), "given", None, CELEBA_TEMPLATE) for p in landmarks]
        elif boxes is not None:
            found = self.faces_in_boxes(image, boxes)
        else:
            template = self.yunet_template()
            found = [
                (f.landmarks, "detected", f.score, template) for f in self.detector.detect(image)
            ]

        faces: list[FaceResult] = []
        crops: list[Image] = []
        templates: list[Points] = []
        for points, source, score, template in found:
            crop, factor = lr_crop(image, points, template, self.lr_size)
            reason = None
            if factor * self.config.max_downscale < 1.0:
                reason = (
                    f"too large for the model: {1.0 / factor:.1f} times its input size "
                    f"(limit {self.config.max_downscale:g}), left to the background"
                )
            elif factor > self.config.max_upscale:
                reason = (
                    f"too small for the model: {factor:.1f} times smaller than its input "
                    f"(limit {self.config.max_upscale:g}), left to the background"
                )
            if reason is not None:
                faces.append(FaceResult(points, source, score, factor, False, reason))
                continue
            faces.append(FaceResult(points, source, score, factor, True))
            crops.append(crop)
            templates.append(template)

        frame = upscale(image, self.scale)
        restored = iter(zip(self.restore(crops), templates, strict=True))
        results: list[FaceResult] = []
        for face in faces:
            if face.restored:
                pixels_hr, template = next(restored)
                landmarks_hr = to_frame(face.landmarks, self.scale)
                matrix = alignment_matrix(landmarks_hr, template, self.hr_size)
                region = paste(frame, pixels_hr, self.mask, matrix)
                if region is None:
                    face = replace(face, restored=False, reason="outside the image")
                else:
                    face = replace(face, region=region)
            results.append(face)
        return Enhanced(frame, results, self.scale)

    def yunet_template(self) -> Points:
        if self.scale not in YUNET_TEMPLATES:
            raise ValueError(f"no YuNet template for scale {self.scale}; give landmarks")
        return YUNET_TEMPLATES[self.scale]

    def faces_in_boxes(
        self, image: Image, boxes: Sequence[Box]
    ) -> list[tuple[Points, str, float | None, Points]]:
        """Landmarks for each face box: the best face the detector finds in it at the lower box
        threshold (its centre inside the box, its size within ``BOX_SIZE_RATIO`` of the box's),
        or the box's own mean landmarks."""
        template = self.yunet_template()
        detected = self.box_detector.detect(image)
        found: list[tuple[Points, str, float | None, Points]] = []
        for box in boxes:
            left, top, width, height = (float(v) for v in box)
            if width <= 0 or height <= 0:
                raise ValueError(f"face box must have a positive size, got {tuple(box)}")
            inside = [
                f
                for f in detected
                if left <= f.box[0] + f.box[2] / 2 <= left + width
                and top <= f.box[1] + f.box[3] / 2 <= top + height
                and 1.0 / BOX_SIZE_RATIO
                <= np.sqrt(f.box[2] * f.box[3] / (width * height))
                <= BOX_SIZE_RATIO
            ]
            if inside:
                best = max(inside, key=lambda f: f.score)
                found.append((best.landmarks, "box-detected", best.score, template))
            else:
                points = box_landmarks(box, YUNET_BOX_LANDMARKS[self.scale])
                found.append((points, "box", None, template))
        return found

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
