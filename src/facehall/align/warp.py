"""Alignment matrices, jitter, warp to the crop and inverse warp back to the frame.

All matrices map source-image pixel-centre coordinates to crop pixel-centre coordinates
(``facehall.align.similarity``). A crop of side ``size`` covers the normalized square
``[0, 1] x [0, 1]`` of the template: the normalized point ``u`` is at pixel-centre coordinate
``u * size - 0.5``. The same template therefore gives consistent crops at every size.
"""

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from facehall.align.similarity import Matrix, compose, scale_of, umeyama_batch

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}
BORDERS = {
    "constant": cv2.BORDER_CONSTANT,  # zeros
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
}

# Numeric image array (H, W) or (H, W, C); the same type OpenCV accepts and returns.
Image = npt.NDArray[np.integer[Any] | np.floating[Any]]


def alignment_matrix(landmarks: npt.ArrayLike, template: npt.ArrayLike, size: int) -> Matrix:
    """Similarity from the source image to a ``size`` x ``size`` crop.

    Fitted by least squares so that ``landmarks`` (pixel-centre coordinates in the source) land
    as close as possible to the normalized ``template`` points.
    """
    batch = np.asarray(landmarks, dtype=np.float64)[None]
    matrix: Matrix = alignment_matrices(batch, template, size)[0]
    return matrix


def alignment_matrices(
    landmarks: npt.ArrayLike, template: npt.ArrayLike, size: int
) -> npt.NDArray[np.float64]:
    """``alignment_matrix`` for a (B, N, 2) batch of landmark sets; returns (B, 2, 3)."""
    if size <= 0:
        raise ValueError(f"crop size must be positive, got {size}")
    # Fit in continuous coordinates (pixel centre at +0.5), then convert back:
    # out = A (p + 0.5) + t - 0.5.
    points = np.asarray(landmarks, dtype=np.float64)
    fitted = umeyama_batch(points + 0.5, np.asarray(template, dtype=np.float64) * size)
    fitted[:, :, 2] += fitted[:, :, :2].sum(axis=2) * 0.5 - 0.5
    return fitted


def hflip_matrix(size: int) -> Matrix:
    """Horizontal flip of a ``size`` x ``size`` crop, in crop pixel-centre coordinates."""
    return np.array([[-1.0, 0.0, size - 1.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class Jitter:
    """Random perturbation of the alignment, to make models robust to imprecise landmarks.

    Each sample is a similarity about the crop centre, drawn uniformly:
    - shift: up to ``shift`` x crop side along each axis;
    - rotation: up to ``rotation`` degrees either way;
    - scale: a factor in ``[1 / (1 + scale), 1 + scale]``, uniform in log space.

    A sample always draws four numbers from the generator, so the random stream does not depend
    on the settings. All zeros gives the identity.
    """

    shift: float = 0.0
    rotation: float = 0.0
    scale: float = 0.0

    def __post_init__(self) -> None:
        if min(self.shift, self.rotation, self.scale) < 0:
            raise ValueError(f"jitter limits must be non-negative: {self}")

    def sample(self, rng: np.random.Generator, size: int) -> Matrix:
        """A random similarity in the coordinates of a ``size`` x ``size`` crop."""
        dx, dy, angle_unit, scale_unit = rng.uniform(-1.0, 1.0, size=4)
        angle = np.radians(self.rotation * angle_unit)
        factor = (1.0 + self.scale) ** scale_unit
        cos, sin = np.cos(angle), np.sin(angle)
        linear = factor * np.array([[cos, -sin], [sin, cos]])
        centre = np.full(2, (size - 1) / 2)
        offset = centre - linear @ centre + np.array([dx, dy]) * self.shift * size
        return np.hstack([linear, offset[:, None]])


def warp(
    image: Image,
    matrix: Matrix,
    size: int | tuple[int, int],
    *,
    interpolation: str = "bicubic",
    border: str = "reflect",
) -> Image:
    """Warp ``image`` with ``matrix`` (source to output) into an output of ``size``.

    ``size`` is a side length, or ``(width, height)``. Output pixels that sample outside the
    source are filled according to ``border``; ``valid_mask`` tells which ones.
    """
    width, height = (size, size) if isinstance(size, int) else size
    result = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=_lookup(INTERPOLATIONS, interpolation, "interpolation"),
        borderMode=_lookup(BORDERS, border, "border"),
    )
    return _keep_channel_axis(image, result)


def rescale_matrix(factor: float) -> Matrix:
    """Pixel-centre coordinates of an image to those of the same image ``factor`` times larger.

    A crop of side ``size`` maps to one of side ``factor * size``: ``(p + 0.5) * factor - 0.5``.
    """
    offset = 0.5 * factor - 0.5
    return np.array([[factor, 0.0, offset], [0.0, factor, offset]])


def resample_crop(image: Image, matrix: Matrix, size: int, *, border: str = "reflect") -> Image:
    """A ``size`` crop of ``image`` through ``matrix`` that is not aliased (PR 4.2).

    At or above the image's own resolution (``scale_of(matrix) >= 1``) this is the bicubic warp.
    Below it, the image is warped to an integer multiple ``k`` of ``size``, at least its own
    resolution, and averaged over k x k blocks: the crop is filtered by an exact box, not by a
    non-integer area resize (such as 33 -> 32 px), which would blur it. The inference pipeline
    cuts its LR crops this way, and so do training samples made like them (``data.scene``).
    """
    factor = scale_of(matrix)
    if factor >= 1.0:
        return warp(image, matrix, size, interpolation="bicubic", border=border)
    k = math.ceil(1.0 / factor - 1e-9)
    full = warp(image, compose(rescale_matrix(k), matrix), size * k, border=border)
    result = cv2.resize(full, (size, size), interpolation=cv2.INTER_AREA)
    return _keep_channel_axis(full, result)


def unwarp(
    crop: Image,
    matrix: Matrix,
    frame_size: tuple[int, int],
    *,
    interpolation: str = "bicubic",
    border: str = "constant",
) -> Image:
    """Map ``crop`` back into a frame of ``frame_size`` = ``(width, height)``.

    ``matrix`` is the same source-to-crop matrix that was used for ``warp``. Frame pixels that
    fall outside the crop are filled according to ``border`` (zeros by default); paste-back
    blends with a mask, so they are never used.
    """
    result = cv2.warpAffine(
        crop,
        matrix,
        frame_size,
        flags=_lookup(INTERPOLATIONS, interpolation, "interpolation") | cv2.WARP_INVERSE_MAP,
        borderMode=_lookup(BORDERS, border, "border"),
    )
    return _keep_channel_axis(crop, result)


def valid_mask(matrix: Matrix, source_size: tuple[int, int], size: int) -> npt.NDArray[np.bool_]:
    """Boolean ``size`` x ``size`` mask: True where the crop samples inside the source image.

    ``source_size`` is ``(width, height)``. False marks padding added by the border mode.
    """
    width, height = source_size
    inside = np.ones((height, width), dtype=np.uint8)
    sampled = warp(inside, matrix, size, interpolation="nearest", border="constant")
    return np.asarray(sampled > 0, dtype=np.bool_)


def _lookup(table: dict[str, int], name: str, kind: str) -> int:
    try:
        return table[name]
    except KeyError:
        raise ValueError(f"unknown {kind} {name!r}; expected one of {sorted(table)}") from None


def _keep_channel_axis(source: Image, result: Image) -> Image:
    """OpenCV drops a trailing channel axis of size 1; restore it."""
    if source.ndim == 3 and result.ndim == 2:
        return result[:, :, None]
    return result
