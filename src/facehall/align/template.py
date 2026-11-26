"""Canonical 5-point face template.

Landmark order (image coordinates, so the "left" eye is the one on the left of the image):
left eye, right eye, nose tip, left mouth corner, right mouth corner. This is the CelebA order
and the usual order of 5-point face detectors.

Template points are in normalized crop coordinates: ``(0, 0)`` is the top-left corner of the
crop and ``(1, 1)`` its bottom-right corner, so one template serves every crop size.
"""

import numpy as np
import numpy.typing as npt

from facehall.align.similarity import Points

LANDMARK_NAMES = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")

# Index permutation that swaps left and right landmarks (for mirroring).
MIRROR_ORDER = (1, 0, 2, 4, 3)

#: Mean landmarks of the CelebA train split (aligned images, pixel coordinates), computed from the
#: manifest on 2026-11-24 (docs/PROJECT.md §6.3).
CELEBA_TRAIN_MEAN_LANDMARKS: Points = np.array(
    [
        [69.3632, 111.1967],
        [107.6367, 111.1549],
        [88.0740, 135.0955],
        [71.2684, 152.1190],
        [105.5621, 152.2041],
    ]
)

#: ``symmetrize(template_from_landmarks(CELEBA_TRAIN_MEAN_LANDMARKS))``, rounded to 5 decimals.
CELEBA_TEMPLATE: Points = np.array(
    [
        [0.375, 0.47323],
        [0.625, 0.47323],
        [0.5, 0.62947],
        [0.388, 0.74094],
        [0.612, 0.74094],
    ]
)


# YuNet's landmarks (facehall.align.detect) follow another annotation convention than CelebA's.
# Templates for them, fitted by scripts/measure_detection.py (commit f45cafa, 2026-11-26) on
# 1,500 CelebA train scenes per scale (faces with 32 / scale px between the eyes, detected on a
# bicubic ×scale upscale; 1,495 and 1,489 detected) and mirror-averaged (D31). Aligning YuNet
# points to the template of the scale gives the crops of CELEBA_TEMPLATE up to the per-image
# error.
YUNET_TEMPLATES: dict[int, Points] = {
    4: np.array(
        [
            [0.37713, 0.4798],
            [0.62287, 0.4798],
            [0.5, 0.62562],
            [0.39685, 0.73847],
            [0.60315, 0.73847],
        ]
    ),
    8: np.array(
        [
            [0.38138, 0.48258],
            [0.61862, 0.48258],
            [0.5, 0.62594],
            [0.40397, 0.74255],
            [0.59603, 0.74255],
        ]
    ),
}


# Mean YuNet landmarks in its face box, normalized to the box ((0, 0) its top-left corner, (1, 1)
# its bottom-right corner): the fallback for a face given as a box (facehall.align.box_landmarks).
# Placeholder until fitted by scripts/measure_detection.py --parts box.
YUNET_BOX_LANDMARKS: Points = np.array(
    [[0.3, 0.4], [0.7, 0.4], [0.5, 0.6], [0.35, 0.75], [0.65, 0.75]]
)


def template_from_landmarks(landmarks: npt.ArrayLike) -> Points:
    """Normalized template from one set of 5 landmarks (typically a dataset mean).

    Uses the FFHQ alignment rule, reduced to 5 points: a square centred slightly below the eyes,
    oriented along the eye-to-eye and eye-to-mouth vectors, with side
    ``max(4 * eye distance, 3.6 * eye-to-mouth distance)``. The landmarks are expressed in the
    coordinates of that square, divided by its side.
    """
    points = _check_landmarks(landmarks)
    left_eye, right_eye, _, mouth_left, mouth_right = points
    eye_avg = (left_eye + right_eye) / 2
    eye_to_eye = right_eye - left_eye
    eye_to_mouth = (mouth_left + mouth_right) / 2 - eye_avg

    x_axis = eye_to_eye - np.array([-eye_to_mouth[1], eye_to_mouth[0]])
    x_axis /= np.hypot(*x_axis)
    y_axis = np.array([-x_axis[1], x_axis[0]])
    half = max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    centre = eye_avg + eye_to_mouth * 0.1
    corner = centre - (x_axis + y_axis) * half

    result: Points = (points - corner) @ np.stack([x_axis, y_axis]).T / (2 * half)
    return result


def mirror(template: npt.ArrayLike) -> Points:
    """Template of the horizontally flipped face."""
    points = _check_landmarks(template)
    flipped = np.stack([1.0 - points[:, 0], points[:, 1]], axis=1)
    return flipped[list(MIRROR_ORDER)]


def symmetrize(template: npt.ArrayLike) -> Points:
    """Average of a template and its mirror image, so that flipping commutes with alignment."""
    points = _check_landmarks(template)
    result: Points = (points + mirror(points)) / 2
    return result


def _check_landmarks(landmarks: npt.ArrayLike) -> Points:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2):
        raise ValueError(f"expected 5 landmarks as a (5, 2) array, got shape {points.shape}")
    return points
