"""Face detection with five landmarks: the YuNet detector of OpenCV Zoo (Phase 4, D31).

Faces as small as the model inputs (4 px between the eyes at ×8, 8 px at ×4) are found reliably
only on a bicubic upscale of the image (``upsample``). On synthetic CelebA scenes, YuNet finds
16% of the faces with 4 px between the eyes at the input size, and 99% after a ×8 upscale
(``docs/PROJECT.md`` §9, Phase 4). The pipeline therefore detects at ``upsample`` = the model
scale, where the face has about the 32 px between the eyes of the HR template.

Coordinates are returned in pixel-centre coordinates of the input image (OpenCV convention,
``facehall.align.warp``). The landmarks follow ``LANDMARK_NAMES``: image-left eye, image-right
eye, nose, image-left and image-right mouth corner, which is also YuNet's order (right eye of the
person first). YuNet's points follow its own annotation convention, not CelebA's: align them with
``YUNET_TEMPLATES``, not with ``CELEBA_TEMPLATE``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from facehall.align.similarity import Points

NET_INPUT = (320, 320)  # placeholder; the input size is set per image


@dataclass(frozen=True)
class Face:
    """One detection, in pixel coordinates of the input image."""

    box: tuple[float, float, float, float]  # left, top, width, height
    landmarks: Points  # (5, 2), pixel-centre coordinates, LANDMARK_NAMES order
    score: float


class YuNet:
    """YuNet through ``cv2.FaceDetectorYN``; detects in RGB uint8 images of shape (H, W, 3)."""

    def __init__(
        self,
        weights: str | Path,
        *,
        upsample: float = 1.0,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if upsample <= 0:
            raise ValueError(f"upsample must be positive, got {upsample}")
        if not (0.0 <= score_threshold <= 1.0 and 0.0 <= nms_threshold <= 1.0):
            raise ValueError("score_threshold and nms_threshold must be in [0, 1]")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if not Path(weights).is_file():
            raise FileNotFoundError(
                f"{weights}: no YuNet weights; run `scripts/download_weights.py yunet`"
            )
        self.upsample = upsample
        self.score_threshold = score_threshold
        self._net = cv2.FaceDetectorYN.create(
            str(weights), "", NET_INPUT, score_threshold, nms_threshold, top_k
        )

    def detect(self, image: npt.NDArray[np.uint8]) -> list[Face]:
        """Faces in ``image``, highest score first."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"expected an RGB uint8 image (H, W, 3), got {image.shape} {image.dtype}"
            )
        height, width = image.shape[:2]
        bgr = np.ascontiguousarray(image[:, :, ::-1])
        size = (max(1, round(width * self.upsample)), max(1, round(height * self.upsample)))
        if size != (width, height):
            bgr = np.asarray(cv2.resize(bgr, size, interpolation=cv2.INTER_CUBIC), dtype=np.uint8)
        self._net.setInputSize(size)
        _, rows = self._net.detect(bgr)
        if rows is None or len(rows) == 0:
            return []
        rows = np.asarray(rows, dtype=np.float64)
        factors = np.array([size[0] / width, size[1] / height])  # exact per axis
        faces = []
        for row in rows[np.argsort(-rows[:, 14], kind="stable")]:
            left, top = row[:2] / factors
            box_width, box_height = row[2:4] / factors
            landmarks: Points = (row[4:14].reshape(5, 2) + 0.5) / factors - 0.5
            faces.append(
                Face(
                    box=(float(left), float(top), float(box_width), float(box_height)),
                    landmarks=landmarks,
                    score=float(row[14]),
                )
            )
        return faces


def box_landmarks(box: Sequence[float], normalized: npt.ArrayLike) -> Points:
    """Five landmarks for a face box ``(left, top, width, height)`` (input pixels, box edges),
    from ``normalized`` landmark positions in the box (``YUNET_BOX_LANDMARKS``). A fallback
    for a face the detector does not find: the face's rotation and proportions are unknown."""
    left, top, width, height = (float(v) for v in box)
    if width <= 0 or height <= 0:
        raise ValueError(f"face box must have a positive size, got {tuple(box)}")
    points: Points = np.array([left, top]) + np.asarray(normalized) * np.array([width, height])
    return points - 0.5
