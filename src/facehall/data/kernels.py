"""Blur kernels for the degradation pipeline (``facehall.data.degrade``).

Every kernel is a 2-D float32 array of odd side, non-negative, summing to 1, centred on its
middle pixel. Sizes are in pixels of the image the kernel is applied to. A size below
``MIN_SIZE`` gives the identity kernel ``[[1]]``.
"""

import math

import numpy as np
import numpy.typing as npt

Kernel = npt.NDArray[np.float32]

MIN_SIZE = 1e-3  # smaller sigmas, lengths and radii give the identity kernel
GAUSSIAN_RADIUS = 3.0  # Gaussian kernels are cut at 3 sigma
MOTION_SAMPLES_PER_PX = 8  # points per pixel of path length, splatted bilinearly
DEFOCUS_SUPERSAMPLING = 8  # subpixels per axis for the disk coverage


def identity() -> Kernel:
    return np.ones((1, 1), dtype=np.float32)


def normalized(kernel: npt.NDArray[np.floating]) -> Kernel:
    total = float(kernel.sum())
    if total <= 0.0:
        return identity()
    result: Kernel = (kernel / total).astype(np.float32)
    return result


def offsets(radius: int) -> npt.NDArray[np.float64]:
    return np.arange(-radius, radius + 1, dtype=np.float64)


def gaussian(sigma_x: float, sigma_y: float | None = None, angle: float = 0.0) -> Kernel:
    """Gaussian with standard deviations ``sigma_x`` and ``sigma_y`` along axes rotated by
    ``angle`` (radians, counter-clockwise from the image x axis, y pointing down)."""
    sigma_y = sigma_x if sigma_y is None else sigma_y
    largest = max(sigma_x, sigma_y)
    if largest < MIN_SIZE:
        return identity()
    radius = math.ceil(GAUSSIAN_RADIUS * largest)
    sx, sy = max(sigma_x, MIN_SIZE), max(sigma_y, MIN_SIZE)
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]])
    inverse = rotation @ np.diag([1.0 / sx**2, 1.0 / sy**2]) @ rotation.T
    y, x = np.meshgrid(offsets(radius), offsets(radius), indexing="ij")
    exponent = inverse[0, 0] * x**2 + 2.0 * inverse[0, 1] * x * y + inverse[1, 1] * y**2
    return normalized(np.exp(-0.5 * exponent))


def motion(length: float, angle: float) -> Kernel:
    """A straight motion path of ``length`` pixels through the centre at ``angle`` (radians),
    rendered with bilinear splatting."""
    if length < MIN_SIZE:
        return identity()
    radius = math.ceil(length / 2.0) + 1
    size = 2 * radius + 1
    count = max(2, math.ceil(length * MOTION_SAMPLES_PER_PX)) + 1
    t = np.linspace(-length / 2.0, length / 2.0, count)
    xs = radius + t * math.cos(angle)
    ys = radius + t * math.sin(angle)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    kernel = np.zeros((size, size), dtype=np.float64)
    for dx, dy, weight in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        np.add.at(kernel, (y0 + dy, x0 + dx), weight)
    return normalized(kernel)


def defocus(radius: float) -> Kernel:
    """A uniform disk of ``radius`` pixels; edge pixels are weighted by their covered area."""
    if radius < MIN_SIZE:
        return identity()
    half = math.ceil(radius)
    steps = (np.arange(DEFOCUS_SUPERSAMPLING) + 0.5) / DEFOCUS_SUPERSAMPLING - 0.5
    fine = (offsets(half)[:, None] + steps[None, :]).reshape(-1)
    y, x = np.meshgrid(fine, fine, indexing="ij")
    inside = (x**2 + y**2 <= radius**2).astype(np.float64)
    size = 2 * half + 1
    coverage = inside.reshape(size, DEFOCUS_SUPERSAMPLING, size, DEFOCUS_SUPERSAMPLING)
    return normalized(coverage.mean(axis=(1, 3)))
