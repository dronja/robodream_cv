"""Tests for facehall.align.warp on synthetic images and points."""

import numpy as np
import numpy.typing as npt
import pytest

from facehall.align.similarity import angle_of, invert, scale_of, transform_points
from facehall.align.template import CELEBA_TEMPLATE
from facehall.align.warp import Image, Jitter, alignment_matrix, unwarp, valid_mask, warp

SOURCE_SIZE = (178, 218)  # (width, height), as CelebA aligned
LANDMARKS = np.array([[69.0, 111.0], [107.0, 112.0], [88.0, 135.0], [71.0, 152.0], [105.0, 153.0]])


def blob(
    center: tuple[float, float], size: tuple[int, int], sigma: float = 2.0
) -> npt.NDArray[np.float32]:
    """Gaussian blob at a sub-pixel position (pixel-centre coordinates)."""
    width, height = size
    y, x = np.mgrid[0:height, 0:width]
    image = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2))
    return np.asarray(image, dtype=np.float32)


def centroid(image: Image) -> npt.NDArray[np.float64]:
    y, x = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    weight = image.astype(np.float64)
    return np.array([(x * weight).sum(), (y * weight).sum()]) / weight.sum()


def smooth_image(size: tuple[int, int]) -> npt.NDArray[np.float32]:
    width, height = size
    y, x = np.mgrid[0:height, 0:width]
    channels = [
        0.5 + 0.4 * np.sin(x / 9.0) * np.cos(y / 13.0),
        0.5 + 0.4 * np.cos((x + y) / 11.0),
        0.5 + 0.3 * np.sin(y / 7.0),
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def test_landmarks_land_on_the_template() -> None:
    # Landmarks that are an exact similarity image of the template land exactly on it.
    target = CELEBA_TEMPLATE * 154.0 + [12.0, 38.0]
    m = alignment_matrix(target, CELEBA_TEMPLATE, 128)
    np.testing.assert_allclose(transform_points(m, target), CELEBA_TEMPLATE * 128 - 0.5, atol=1e-9)


def test_crop_sizes_are_consistent() -> None:
    m32 = alignment_matrix(LANDMARKS, CELEBA_TEMPLATE, 32)
    m128 = alignment_matrix(LANDMARKS, CELEBA_TEMPLATE, 128)
    points = np.array([[0.0, 0.0], [100.0, 50.0], [177.0, 217.0]])
    p32 = transform_points(m32, points)
    np.testing.assert_allclose(transform_points(m128, points), (p32 + 0.5) * 4 - 0.5, atol=1e-9)


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic", "lanczos"])
def test_warp_moves_content_where_the_matrix_says(interpolation: str) -> None:
    m = alignment_matrix(LANDMARKS, CELEBA_TEMPLATE, 128)
    for point in [(88.3, 135.6), (60.0, 100.25), (120.7, 160.0)]:
        crop = warp(blob(point, SOURCE_SIZE), m, 128, interpolation=interpolation)
        expected = transform_points(m, np.array([point]))[0]
        np.testing.assert_allclose(centroid(crop), expected, atol=0.05)


def test_unwarp_inverts_warp() -> None:
    image = smooth_image(SOURCE_SIZE)
    m = alignment_matrix(LANDMARKS, CELEBA_TEMPLATE, 128)
    crop = warp(image, m, 128)
    restored = unwarp(crop, m, SOURCE_SIZE)
    assert restored.shape == image.shape

    # Compare where the frame maps well inside the crop (away from interpolation borders).
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    in_crop = transform_points(m, np.stack([x.ravel(), y.ravel()], axis=1)).reshape(
        height, width, 2
    )
    inside = ((in_crop > 2) & (in_crop < 125)).all(axis=-1)
    assert inside.mean() > 0.4
    assert np.abs(restored[inside] - image[inside]).max() < 0.02


def test_points_round_trip() -> None:
    m = alignment_matrix(LANDMARKS, CELEBA_TEMPLATE, 128)
    np.testing.assert_allclose(
        transform_points(invert(m), transform_points(m, LANDMARKS)), LANDMARKS
    )


def test_channel_axis_and_dtype_are_kept() -> None:
    m = alignment_matrix(LANDMARKS, CELEBA_TEMPLATE, 64)
    gray = np.zeros((218, 178, 1), dtype=np.uint8)
    assert warp(gray, m, 64).shape == (64, 64, 1)
    assert unwarp(np.zeros((64, 64, 1), np.uint8), m, SOURCE_SIZE).shape == (218, 178, 1)
    rgb = np.zeros((218, 178, 3), dtype=np.uint8)
    assert warp(rgb, m, (40, 30)).shape == (30, 40, 3)
    assert warp(rgb, m, 64).dtype == np.uint8


def test_valid_mask_marks_padding() -> None:
    inside = alignment_matrix(LANDMARKS, CELEBA_TEMPLATE, 128)
    assert valid_mask(inside, SOURCE_SIZE, 128).all()
    # Same face moved to the image corner: the crop now reaches outside the source.
    shifted = alignment_matrix(LANDMARKS - [60.0, 100.0], CELEBA_TEMPLATE, 128)
    mask = valid_mask(shifted, SOURCE_SIZE, 128)
    assert 0.2 < mask.mean() < 0.9
    assert not mask[0, 0] and mask[-1, -1]


def test_unknown_mode_names_are_rejected() -> None:
    m = alignment_matrix(LANDMARKS, CELEBA_TEMPLATE, 32)
    image = np.zeros((218, 178, 3), np.uint8)
    with pytest.raises(ValueError, match="unknown interpolation 'area'"):
        warp(image, m, 32, interpolation="area")
    with pytest.raises(ValueError, match="unknown border 'wrap'"):
        warp(image, m, 32, border="wrap")


def test_alignment_rejects_bad_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        alignment_matrix(LANDMARKS, CELEBA_TEMPLATE, 0)


def test_zero_jitter_is_identity_and_keeps_the_random_stream() -> None:
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    identity = Jitter().sample(a, 128)
    Jitter(shift=0.1, rotation=10.0, scale=0.2).sample(b, 128)
    np.testing.assert_allclose(identity, [[1, 0, 0], [0, 1, 0]], atol=1e-12)
    assert a.random() == b.random()


def test_jitter_stays_within_limits() -> None:
    jitter = Jitter(shift=0.05, rotation=8.0, scale=0.1)
    rng = np.random.default_rng(0)
    size = 128
    centre = np.array([[(size - 1) / 2] * 2])
    shifts, angles, scales = [], [], []
    for _ in range(2000):
        m = jitter.sample(rng, size)
        shifts.append(transform_points(m, centre)[0] - centre[0])
        angles.append(angle_of(m))
        scales.append(scale_of(m))
    shift = np.array(shifts)
    assert np.abs(shift).max() <= 0.05 * size + 1e-9
    assert np.abs(angles).max() <= 8.0 + 1e-9
    assert min(scales) >= 1 / 1.1 - 1e-9 and max(scales) <= 1.1 + 1e-9
    # Both directions occur, close to the limits.
    assert shift.min() < -0.045 * size and shift.max() > 0.045 * size
    assert min(angles) < -7.5 and max(angles) > 7.5
    assert min(scales) < 1 / 1.09 and max(scales) > 1.09


def test_negative_jitter_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Jitter(rotation=-1.0)
