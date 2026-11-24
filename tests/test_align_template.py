"""Tests for facehall.align.template."""

import numpy as np
import pytest

from facehall.align.similarity import transform_points
from facehall.align.template import (
    CELEBA_TEMPLATE,
    CELEBA_TRAIN_MEAN_LANDMARKS,
    mirror,
    symmetrize,
    template_from_landmarks,
)

# Symmetric face: eye distance 20 and eye-to-mouth distance 20, so the square side is
# max(4 * 20, 3.6 * 20) = 80, centred at the eye midpoint + 0.1 * eye-to-mouth = (50, 52).
FACE = np.array([[40.0, 50.0], [60.0, 50.0], [50.0, 60.0], [42.0, 70.0], [58.0, 70.0]])


def test_template_follows_the_ffhq_square() -> None:
    expected = (FACE - [10.0, 12.0]) / 80.0
    np.testing.assert_allclose(template_from_landmarks(FACE), expected, atol=1e-12)


def test_template_is_invariant_to_similarity_transforms() -> None:
    a = np.radians(25.0)
    m = np.array([[2 * np.cos(a), -2 * np.sin(a), 13.0], [2 * np.sin(a), 2 * np.cos(a), -7.0]])
    np.testing.assert_allclose(
        template_from_landmarks(transform_points(m, FACE)),
        template_from_landmarks(FACE),
        atol=1e-12,
    )


def test_mirror_and_symmetrize() -> None:
    template = template_from_landmarks(
        FACE + [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    np.testing.assert_allclose(mirror(mirror(template)), template)
    symmetric = symmetrize(template)
    np.testing.assert_allclose(mirror(symmetric), symmetric, atol=1e-12)
    assert symmetric[2, 0] == pytest.approx(0.5)


def test_celeba_template_matches_its_derivation() -> None:
    derived = symmetrize(template_from_landmarks(CELEBA_TRAIN_MEAN_LANDMARKS))
    np.testing.assert_allclose(CELEBA_TEMPLATE, derived, atol=5e-6)
    np.testing.assert_allclose(mirror(CELEBA_TEMPLATE), CELEBA_TEMPLATE, atol=1e-12)
    # Eyes a quarter of the crop apart and centred: the eye-distance rule of the square wins.
    assert CELEBA_TEMPLATE[1, 0] - CELEBA_TEMPLATE[0, 0] == pytest.approx(0.25)


def test_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(5, 2\)"):
        template_from_landmarks(np.zeros((4, 2)))
