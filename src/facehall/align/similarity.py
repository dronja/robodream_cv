"""2-D similarity transforms stored as 2x3 affine matrices (OpenCV layout).

A matrix ``m`` maps a point ``p`` to ``m[:, :2] @ p + m[:, 2]``. Image coordinates are
pixel-centre coordinates, as in OpenCV: the centre of the pixel in row ``i`` and column ``j`` is
at ``(x, y) = (j, i)``.
"""

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float64]  # (2, 3)
Points = npt.NDArray[np.float64]  # (N, 2)

_EPS = 1e-12


def umeyama(src: npt.ArrayLike, dst: npt.ArrayLike) -> Matrix:
    """Least-squares similarity (rotation, uniform scale, translation) that maps ``src`` to ``dst``.

    Umeyama (1991), restricted to proper rotations: the result never contains a reflection.
    Raises ValueError for mismatched shapes or degenerate ``src`` (all points equal).
    """
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape or len(a) < 2:
        raise ValueError(f"expected two (N, 2) arrays with N >= 2, got {a.shape} and {b.shape}")
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    a, b = a - mean_a, b - mean_b
    var_a = (a**2).sum() / len(a)
    if var_a < _EPS:
        raise ValueError("source points are degenerate (all equal)")
    u, sigma, vt = np.linalg.svd(b.T @ a / len(a))
    signs = np.array([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt)) or 1.0])
    rotation = u @ np.diag(signs) @ vt
    scale = float((sigma * signs).sum() / var_a)
    translation = mean_b - scale * rotation @ mean_a
    return np.hstack([scale * rotation, translation[:, None]])


def umeyama_batch(src: npt.ArrayLike, dst: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """``umeyama`` for many point sets at once: (B, N, 2) to (N, 2) or (B, N, 2) -> (B, 2, 3).

    In 2-D the least-squares proper similarity is the complex least-squares fit
    ``z -> k * z + t``, which this computes in closed form.
    """
    a = np.asarray(src, dtype=np.float64)
    b = np.broadcast_to(np.asarray(dst, dtype=np.float64), a.shape)
    if a.ndim != 3 or a.shape[2] != 2 or a.shape[1] < 2:
        raise ValueError(f"expected (B, N, 2) source points with N >= 2, got {a.shape}")
    za = a[..., 0] + 1j * a[..., 1]
    zb = b[..., 0] + 1j * b[..., 1]
    mean_a, mean_b = za.mean(axis=1), zb.mean(axis=1)
    da, db = za - mean_a[:, None], zb - mean_b[:, None]
    var = (np.abs(da) ** 2).sum(axis=1)
    if (var < _EPS).any():
        raise ValueError("source points are degenerate (all equal)")
    k = (np.conj(da) * db).sum(axis=1) / var
    t = mean_b - k * mean_a
    rows = [[k.real, -k.imag, t.real], [k.imag, k.real, t.imag]]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=1)


def transform_points(matrix: Matrix, points: npt.ArrayLike) -> Points:
    """Apply ``matrix`` to an (N, 2) array of points."""
    p = np.asarray(points, dtype=np.float64)
    result: Points = p @ matrix[:, :2].T + matrix[:, 2]
    return result


def invert(matrix: Matrix) -> Matrix:
    """Inverse transform."""
    linear = np.linalg.inv(matrix[:, :2])
    return np.hstack([linear, -(linear @ matrix[:, 2])[:, None]])


def compose(outer: Matrix, inner: Matrix) -> Matrix:
    """The transform that applies ``inner`` first, then ``outer``."""
    return np.hstack(
        [outer[:, :2] @ inner[:, :2], (outer[:, :2] @ inner[:, 2] + outer[:, 2])[:, None]]
    )


def translation(dx: float, dy: float) -> Matrix:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])


def scale_of(matrix: Matrix) -> float:
    """Uniform scale factor: output pixels per input pixel."""
    return float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))


def angle_of(matrix: Matrix) -> float:
    """Rotation angle in degrees (positive from +x towards +y, i.e. clockwise on screen)."""
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
