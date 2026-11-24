"""Tests for facehall.align.similarity."""

import numpy as np
import pytest

from facehall.align.similarity import (
    Matrix,
    angle_of,
    compose,
    invert,
    scale_of,
    transform_points,
    translation,
    umeyama,
    umeyama_batch,
)


def similarity(scale: float, angle_deg: float, dx: float, dy: float) -> Matrix:
    a = np.radians(angle_deg)
    linear = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.hstack([linear, [[dx], [dy]]])


POINTS = np.array([[69.0, 111.0], [107.0, 111.0], [88.0, 135.0], [71.0, 152.0], [105.0, 152.0]])


@pytest.mark.parametrize(
    ("scale", "angle", "dx", "dy"),
    [
        (1.0, 0.0, 0.0, 0.0),
        (0.82, 3.0, -10.0, 5.0),
        (2.5, -170.0, 40.0, -7.5),
        (0.1, 90.0, 1.0, 2.0),
    ],
)
def test_umeyama_recovers_an_exact_similarity(
    scale: float, angle: float, dx: float, dy: float
) -> None:
    truth = similarity(scale, angle, dx, dy)
    fitted = umeyama(POINTS, transform_points(truth, POINTS))
    np.testing.assert_allclose(fitted, truth, atol=1e-9)
    assert scale_of(fitted) == pytest.approx(scale)
    assert angle_of(fitted) == pytest.approx(angle)


def test_umeyama_is_the_least_squares_optimum() -> None:
    rng = np.random.default_rng(0)
    truth = similarity(0.8, 5.0, 3.0, -2.0)
    noisy = transform_points(truth, POINTS) + rng.normal(0.0, 2.0, POINTS.shape)
    fitted = umeyama(POINTS, noisy)

    def error(m: Matrix) -> float:
        return float(((transform_points(m, POINTS) - noisy) ** 2).sum())

    assert error(fitted) <= error(truth)
    for _ in range(50):  # no nearby similarity fits better
        nudged = similarity(
            scale_of(fitted) * (1 + rng.normal(0, 1e-3)),
            angle_of(fitted) + rng.normal(0, 0.05),
            fitted[0, 2] + rng.normal(0, 0.05),
            fitted[1, 2] + rng.normal(0, 0.05),
        )
        assert error(fitted) <= error(nudged)


def test_umeyama_never_reflects() -> None:
    mirrored = POINTS * np.array([-1.0, 1.0])
    fitted = umeyama(POINTS, mirrored)
    assert np.linalg.det(fitted[:, :2]) > 0


def test_umeyama_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        umeyama(np.ones((5, 2)), POINTS)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        umeyama(POINTS, POINTS[:4])
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        umeyama(POINTS[:1], POINTS[:1])


def test_invert_and_compose() -> None:
    m = similarity(1.7, 30.0, 12.0, -4.0)
    np.testing.assert_allclose(compose(invert(m), m), translation(0.0, 0.0), atol=1e-12)
    shift = translation(1.0, 2.0)
    np.testing.assert_allclose(
        transform_points(compose(shift, m), POINTS), transform_points(m, POINTS) + [1.0, 2.0]
    )


def test_umeyama_batch_matches_umeyama() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(0.0, 30.0, (20, 5, 2))
    dst = rng.normal(0.0, 30.0, (5, 2))
    batch = umeyama_batch(src, dst)
    assert batch.shape == (20, 2, 3)
    for i in range(20):
        np.testing.assert_allclose(batch[i], umeyama(src[i], dst), atol=1e-9)
    with pytest.raises(ValueError, match="degenerate"):
        umeyama_batch(np.ones((2, 5, 2)), dst)
