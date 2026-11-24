"""Tests for facehall.data.kernels."""

import math

import numpy as np
import numpy.typing as npt
import pytest

from facehall.data import kernels


def moments(kernel: npt.NDArray[np.float32]) -> tuple[float, float, float, float, float]:
    """Mean x, mean y, variance x, variance y, covariance of a kernel as a distribution."""
    radius = kernel.shape[0] // 2
    y, x = np.meshgrid(
        np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij"
    )
    w = kernel.astype(np.float64)
    mx, my = float((w * x).sum()), float((w * y).sum())
    vx = float((w * (x - mx) ** 2).sum())
    vy = float((w * (y - my) ** 2).sum())
    cxy = float((w * (x - mx) * (y - my)).sum())
    return mx, my, vx, vy, cxy


def check_kernel(kernel: npt.NDArray[np.float32]) -> None:
    assert kernel.dtype == np.float32
    assert kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1
    assert kernel.min() >= 0.0
    assert kernel.sum() == pytest.approx(1.0, abs=1e-5)
    mx, my, *_ = moments(kernel)
    assert abs(mx) < 1e-5 and abs(my) < 1e-5  # centred


@pytest.mark.parametrize(
    "kernel",
    [
        kernels.gaussian(0.0),
        kernels.motion(0.0, 1.0),
        kernels.defocus(0.0),
    ],
)
def test_tiny_sizes_give_the_identity(kernel: npt.NDArray[np.float32]) -> None:
    np.testing.assert_array_equal(kernel, np.ones((1, 1), dtype=np.float32))


@pytest.mark.parametrize("sigma", [1.0, 2.5, 6.0])
def test_gaussian_variance(sigma: float) -> None:
    # Point sampling makes the variance of a narrower Gaussian smaller than sigma^2.
    kernel = kernels.gaussian(sigma)
    check_kernel(kernel)
    _, _, vx, vy, cxy = moments(kernel)
    assert vx == pytest.approx(sigma**2, rel=0.03)
    assert vy == pytest.approx(vx, rel=1e-6)
    assert abs(cxy) < 1e-6


def test_anisotropic_gaussian_follows_its_angle() -> None:
    along_x = kernels.gaussian(3.0, 1.0, 0.0)
    check_kernel(along_x)
    _, _, vx, vy, _ = moments(along_x)
    assert vx == pytest.approx(9.0, rel=0.03) and vy == pytest.approx(1.0, rel=0.03)
    rotated = kernels.gaussian(3.0, 1.0, math.pi / 2)
    np.testing.assert_allclose(rotated, along_x.T, atol=1e-6)
    diagonal = kernels.gaussian(3.0, 1.0, math.pi / 4)
    _, _, vx, vy, cxy = moments(diagonal)
    assert vx == pytest.approx(vy, rel=1e-3)
    assert cxy == pytest.approx((9.0 - 1.0) / 2, rel=0.03)  # y points down: +x and +y together


def test_motion_kernel_is_a_line_of_the_given_length() -> None:
    length = 6.0
    kernel = kernels.motion(length, 0.0)
    check_kernel(kernel)
    centre = kernel.shape[0] // 2
    assert kernel[centre].sum() == pytest.approx(1.0, abs=1e-6)  # only the middle row
    _, _, vx, vy, _ = moments(kernel)
    assert vx == pytest.approx(length**2 / 12, rel=0.1)  # uniform segment of this length
    assert vy == pytest.approx(0.0, abs=1e-9)
    vertical = kernels.motion(length, math.pi / 2)
    np.testing.assert_allclose(vertical, kernel.T, atol=1e-6)
    check_kernel(kernels.motion(4.3, 0.7))


def test_defocus_kernel_is_a_disk() -> None:
    radius = 3.0
    kernel = kernels.defocus(radius)
    check_kernel(kernel)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-7)
    np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-7)
    centre = kernel.shape[0] // 2
    assert kernel[centre, centre] == pytest.approx(1.0 / (math.pi * radius**2), rel=0.02)
    _, _, vx, _, _ = moments(kernel)
    assert vx == pytest.approx(radius**2 / 4, rel=0.05)  # variance of a uniform disk per axis
    assert kernel[0, 0] == 0.0
