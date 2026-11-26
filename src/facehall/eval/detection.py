"""Detection and alignment accuracy on synthetic scenes from CelebA (Phase 4, D31).

A scene is an aligned CelebA image, downscaled so that the annotated eyes are ``interocular``
pixels apart, optionally degraded, and padded on every side by its own size with its mean
colour, so that the face does not fill the frame. The annotated landmarks give the truth.

- ``match``: the detection the pipeline would use (highest score), or ``None`` if it misses
  the face;
- ``fit_template``: the mean detected landmarks in the crop that the annotated landmarks align
  to ``CELEBA_TEMPLATE``, mirrored and averaged. Aligning detected points to this template gives
  the crops the models were trained on, up to the per-image error;
- ``alignment_error``: that per-image error, as the similarity between the crop from the detected
  points and the crop from the annotated points, in the units of the training jitter
  (``JitterConfig``).

The scenes are optimistic: CelebA faces are near-frontal and centred, the backgrounds are flat,
and the degradation is one fixed light mix. YuNet was trained on WIDER FACE, so the WIDER
real-world set cannot measure it either (D31).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from facehall.align.detect import Face, box_iou
from facehall.align.similarity import Points, angle_of, compose, invert, scale_of, transform_points
from facehall.align.template import CELEBA_TEMPLATE, symmetrize
from facehall.align.warp import alignment_matrix
from facehall.data.config import JitterConfig
from facehall.data.manifest import read_manifest

CROP = 128  # HR crop side; errors in pixels are HR pixels


@dataclass(frozen=True)
class SceneDegradation:
    """A fixed light degradation of the downscaled scene, in its own pixels and 8-bit levels."""

    blur: float = 0.5  # Gaussian sigma
    noise: float = 3.0  # Gaussian sigma
    jpeg: int = 70  # quality


@dataclass(frozen=True)
class AlignmentError:
    shift: float  # largest per-axis shift of the crop centre, as a fraction of the crop side
    rotation: float  # absolute angle, degrees
    scale: float  # factor >= 1 (larger of the scale and its inverse)
    points: float  # mean distance of the annotated landmarks from the template, HR pixels

    def within(self, jitter: JitterConfig) -> bool:
        """Inside the training jitter on all three axes."""
        return (
            self.shift <= jitter.shift
            and self.rotation <= jitter.rotation
            and self.scale <= 1.0 + jitter.scale
        )


Row = tuple[Path, Points]  # image path and its five annotated landmarks


def sample_rows(root: Path, split: str, count: int) -> list[Row]:
    """``count`` rows of ``split``, evenly spaced in manifest order (deterministic)."""
    table = read_manifest(root).select(["path", "split", "landmarks"]).to_pydict()
    rows = [
        (root / path, np.asarray(points, dtype=np.float64).reshape(5, 2))
        for path, part, points in zip(
            table["path"], table["split"], table["landmarks"], strict=True
        )
        if part == split
    ]
    step = max(1, len(rows) // count)
    return rows[::step][:count]


def load_rgb(path: Path) -> npt.NDArray[np.uint8]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(image[:, :, ::-1], dtype=np.uint8)


def interocular(landmarks: npt.ArrayLike) -> float:
    points = np.asarray(landmarks, dtype=np.float64)
    return float(np.linalg.norm(points[0] - points[1]))


def make_scene(
    image: npt.NDArray[np.uint8],
    landmarks: npt.ArrayLike,
    eyes: float,
    *,
    degradation: SceneDegradation | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.uint8], Points]:
    """A padded scene (RGB uint8) with ``eyes`` px between the eyes, and the landmarks in it."""
    truth = np.asarray(landmarks, dtype=np.float64)
    height, width = image.shape[:2]
    factor = eyes / interocular(truth)
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    small = np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
    if degradation is not None:
        small = degrade(small, degradation, rng if rng is not None else np.random.default_rng())
    fill = [int(round(v)) for v in small.reshape(-1, 3).mean(axis=0)]
    pad_x, pad_y = size
    scene = cv2.copyMakeBorder(small, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=fill)
    scale = np.array([size[0] / width, size[1] / height])
    points: Points = (truth + 0.5) * scale - 0.5 + np.array([pad_x, pad_y])
    return np.asarray(scene, dtype=np.uint8), points


def make_scenes(
    rows: Sequence[Row], eyes: float, degradation: SceneDegradation | None, seed: int
) -> Iterator[tuple[npt.NDArray[np.uint8], Points]]:
    """``make_scene`` for every row, with one seeded generator for the degradation."""
    rng = np.random.default_rng(seed)
    for path, points in rows:
        yield make_scene(load_rgb(path), points, eyes, degradation=degradation, rng=rng)


def degrade(
    image: npt.NDArray[np.uint8], settings: SceneDegradation, rng: np.random.Generator
) -> npt.NDArray[np.uint8]:
    out = image
    if settings.blur > 0:
        out = np.asarray(cv2.GaussianBlur(out, (0, 0), settings.blur), dtype=np.uint8)
    if settings.noise > 0:
        noisy = out.astype(np.float32) + rng.normal(0.0, settings.noise, out.shape)
        out = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".jpg", out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return np.ascontiguousarray(decoded[:, :, ::-1], dtype=np.uint8)


def landmark_error(points: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """Mean landmark distance over the true interocular distance (NME)."""
    distance = np.linalg.norm(np.asarray(points) - np.asarray(truth), axis=1)
    return float(distance.mean() / interocular(truth))


def match(faces: Sequence[Face], truth: npt.ArrayLike, max_error: float = 0.5) -> Face | None:
    """The highest-scoring face, or ``None`` when it is farther than ``max_error`` (NME)."""
    if not faces:
        return None
    best = max(faces, key=lambda face: face.score)
    return best if landmark_error(best.landmarks, truth) <= max_error else None


def fit_template(pairs: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]) -> Points:
    """Template for detected landmarks, from (detected, annotated) pairs; see the module doc."""
    if not pairs:
        raise ValueError("no pairs to fit a template on")
    normalized = []
    for detected, truth in pairs:
        matrix = alignment_matrix(truth, CELEBA_TEMPLATE, CROP)
        normalized.append((transform_points(matrix, detected) + 0.5) / CROP)
    return symmetrize(np.mean(normalized, axis=0))


def fit_box_landmarks(faces: Sequence[Face]) -> Points:
    """Mean landmarks of detected faces in box-normalized coordinates (``box_landmarks``):
    (0, 0) is the box's top-left corner, (1, 1) its bottom-right corner; mirror-averaged."""
    if not faces:
        raise ValueError("no faces to fit box landmarks on")
    normalized = [
        (np.asarray(f.landmarks) + 0.5 - np.array(f.box[:2])) / np.array(f.box[2:]) for f in faces
    ]
    return symmetrize(np.mean(normalized, axis=0))


def alignment_error(
    detected: npt.ArrayLike, truth: npt.ArrayLike, template: npt.ArrayLike
) -> AlignmentError:
    """How far the crop aligned from ``detected`` (to ``template``) is from the true crop."""
    found = alignment_matrix(detected, template, CROP)
    true = alignment_matrix(truth, CELEBA_TEMPLATE, CROP)
    relative = compose(found, invert(true))  # true crop -> detected crop
    centre = (CROP - 1) / 2.0
    moved = transform_points(relative, np.array([[centre, centre]]))[0] - centre
    factor = scale_of(relative)
    target = CELEBA_TEMPLATE * CROP - 0.5
    points = np.linalg.norm(transform_points(found, truth) - target, axis=1).mean()
    return AlignmentError(
        shift=float(np.abs(moved).max() / CROP),
        rotation=abs(angle_of(relative)),
        scale=float(max(factor, 1.0 / factor)),
        points=float(points),
    )


def summarize(errors: Sequence[AlignmentError], jitter: JitterConfig) -> dict[str, float]:
    """Medians, 95th percentiles and the fraction inside the training jitter."""
    if not errors:
        raise ValueError("no alignment errors to summarize")
    table = {
        "shift": np.array([e.shift for e in errors]),
        "rotation": np.array([e.rotation for e in errors]),
        "scale": np.array([e.scale for e in errors]),
        "points": np.array([e.points for e in errors]),
    }
    out = {"faces": float(len(errors))}
    for name, values in table.items():
        out[f"{name}_p50"] = float(np.median(values))
        out[f"{name}_p95"] = float(np.percentile(values, 95))
    out["points_mean"] = float(table["points"].mean())
    out["within_jitter"] = float(np.mean([e.within(jitter) for e in errors]))
    return out


# Real photos (PR 4.4): how many detections of a size the pipeline restores are faces.

BOX_EYE_SPREAD = 0.448  # eye distance / box width of YuNet's boxes (YUNET_BOX_LANDMARKS)


@dataclass(frozen=True)
class Detection:
    """A detection on a real photo, scored against the photo's annotated face boxes."""

    score: float
    lr_scale: float  # LR crop px per photo px, as in the pipeline
    true: bool  # overlaps an annotated face box (IoU at least ``iou``)


def score_detections(
    faces: Sequence[Face],
    truth: Sequence[Sequence[float]],
    template: npt.ArrayLike,
    lr_size: int,
    *,
    iou: float = 0.3,
) -> tuple[list[Detection], list[float | None]]:
    """The detections scored against the annotated boxes ``truth`` (left, top, width, height),
    and for each annotated box the best score of a detection that overlaps it (None if none)."""
    detections = []
    found: list[float | None] = [None] * len(truth)
    for face in faces:
        factor = scale_of(alignment_matrix(face.landmarks, template, lr_size))
        overlaps = [box_iou(face.box, box) >= iou for box in truth]
        detections.append(Detection(face.score, factor, any(overlaps)))
        for k, hit in enumerate(overlaps):
            if hit:
                found[k] = max(face.score, found[k] or 0.0)
    return detections, found


def box_lr_scale(width: float, lr_size: int) -> float:
    """The pipeline's crop scale for a face whose box is ``width`` px wide (YuNet-like box)."""
    return 0.25 * lr_size / (BOX_EYE_SPREAD * width)  # the template eyes are 0.25 crop apart


def precision_table(
    detections: Sequence[Detection],
    recalled: Sequence[float | None],
    thresholds: Sequence[float],
    photos: int,
    limits: tuple[float, float],
) -> list[dict[str, Any]]:
    """Per score threshold: the detections the pipeline would restore (crop scale within
    ``limits``), the fraction of them that are faces, false ones per photo, and the fraction of
    annotated faces of restorable size found (``recalled``: best overlapping score per face)."""
    low, high = limits
    sized = [d for d in detections if low <= d.lr_scale <= high]
    rows = []
    for t in thresholds:
        kept = [d for d in sized if d.score >= t]
        true = sum(d.true for d in kept)
        rows.append(
            {
                "threshold": t,
                "restorable": len(kept),
                "precision": true / len(kept) if kept else None,
                "false_per_photo": (len(kept) - true) / photos,
                "recall": float(np.mean([s is not None and s >= t for s in recalled]))
                if recalled
                else None,
            }
        )
    return rows
