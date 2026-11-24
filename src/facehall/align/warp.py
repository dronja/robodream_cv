"""Alignment matrices, jitter, warp to the crop and inverse warp back to the frame.

All matrices map source-image pixel-centre coordinates to crop pixel-centre coordinates
(``facehall.align.similarity``). A crop of side ``size`` covers the normalized square
``[0, 1] x [0, 1]`` of the template: the normalized point ``u`` is at pixel-centre coordinate
``u * size - 0.5``. The same template therefore gives consistent crops at every size.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from facehall.align.similarity import Matrix, compose, translation, umeyama

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
    if size <= 0:
        raise ValueError(f"crop size must be positive, got {size}")
    # Fit in continuous coordinates (pixel centre at +0.5), then convert back.
    fitted = umeyama(np.asarray(landmarks, dtype=np.float64) + 0.5, np.asarray(template) * size)
    return compose(translation(-0.5, -0.5), compose(fitted, translation(0.5, 0.5)))


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
