"""Face detection (facehall.align.detect) and its measures (facehall.eval.detection).

The detector network is replaced by a stub: tests need no weights.
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from facehall.align import (
    CELEBA_TEMPLATE,
    YUNET_BOX_LANDMARKS,
    YUNET_TEMPLATES,
    Face,
    YuNet,
    box_landmarks,
    mirror,
)
from facehall.align.detect import box_iou, detect_tiled, suppress, tile_starts
from facehall.align.warp import alignment_matrix
from facehall.data.config import JitterConfig
from facehall.eval.detection import (
    AlignmentError,
    SceneDegradation,
    alignment_error,
    degrade,
    fit_box_landmarks,
    fit_template,
    interocular,
    landmark_error,
    make_scene,
    match,
    summarize,
)

CELEBA_POINTS = np.array(
    [[69.0, 111.0], [108.0, 112.0], [84.0, 134.0], [73.0, 152.0], [105.0, 152.0]]
)


class StubNet:
    """Stands in for cv2.FaceDetectorYN: records its input and returns fixed rows."""

    def __init__(self, rows: npt.NDArray[np.float32] | None) -> None:
        self.rows = rows
        self.size: tuple[int, int] | None = None
        self.image: npt.NDArray[np.uint8] | None = None

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802 (OpenCV name)
        self.size = size

    def detect(self, image: npt.NDArray[np.uint8]) -> tuple[int, npt.NDArray[np.float32] | None]:
        self.image = image
        return 1, self.rows


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "yunet.onnx"
    path.write_bytes(b"stub")
    return path


def stub(monkeypatch: pytest.MonkeyPatch, rows: npt.NDArray[np.float32] | None) -> StubNet:
    net = StubNet(rows)

    def create(*args: Any) -> StubNet:
        return net

    monkeypatch.setattr(cv2.FaceDetectorYN, "create", create)
    return net


def yunet_row(landmarks: npt.ArrayLike, box: tuple[float, ...], score: float) -> list[float]:
    return [*box, *np.asarray(landmarks, dtype=np.float64).ravel().tolist(), score]


def test_yunet_maps_coordinates_back_from_the_upscale(
    monkeypatch: pytest.MonkeyPatch, weights: Path
) -> None:
    source = np.array([[10.0, 12.0], [20.0, 12.0], [15.0, 17.0], [11.0, 21.0], [19.0, 21.0]])
    upscaled = (source + 0.5) * 4 - 0.5  # the same points, pixel-centre, in the ×4 image
    rows = np.array(
        [
            yunet_row(upscaled, (8.0, 16.0, 80.0, 96.0), 0.7),
            yunet_row(upscaled + 4, (0.0, 0.0, 4.0, 4.0), 0.9),
        ],
        dtype=np.float32,
    )
    net = stub(monkeypatch, rows)
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[:, :, 2] = 200  # blue in RGB
    faces = YuNet(weights, upsample=4).detect(image)

    assert net.size == (160, 120)
    assert net.image is not None and net.image.shape == (120, 160, 3)
    assert net.image[:, :, 0].min() == 200  # the network gets BGR
    assert [f.score for f in faces] == pytest.approx([0.9, 0.7])  # highest score first
    np.testing.assert_allclose(faces[1].landmarks, source, atol=1e-4)
    assert faces[1].box == pytest.approx((2.0, 4.0, 20.0, 24.0))


def test_yunet_without_faces_and_invalid_input(
    monkeypatch: pytest.MonkeyPatch, weights: Path, tmp_path: Path
) -> None:
    stub(monkeypatch, None)
    detector = YuNet(weights)
    assert detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == []
    with pytest.raises(ValueError, match="RGB uint8"):
        detector.detect(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB uint8"):
        detector.detect(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="download_weights.py yunet"):
        YuNet(tmp_path / "missing.onnx")
    for kwargs in (
        {"upsample": 0.0},
        {"score_threshold": 1.5},
        {"top_k": 0},
        {"upsample": 8.0, "max_pixels": 1000**2, "overlap": 125},  # tiles of 125 px
    ):
        with pytest.raises(ValueError):
            YuNet(weights, **kwargs)


def test_tile_starts_cover_with_overlap() -> None:
    assert tile_starts(100, 120, 16) == [0]
    for length, tile, overlap in ((1000, 300, 64), (1001, 500, 128), (4321, 1000, 128)):
        starts = tile_starts(length, tile, overlap)
        assert starts[0] == 0 and starts[-1] + tile == length
        assert all(b - a <= tile - overlap + 1 for a, b in zip(starts, starts[1:], strict=False))
    with pytest.raises(ValueError):
        tile_starts(100, 16, 16)


def test_box_iou_and_suppression() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    points = np.zeros((5, 2))
    faces = [
        Face((0, 0, 10, 10), points, 0.5),
        Face((1, 0, 10, 10), points, 0.9),
        Face((30, 30, 10, 10), points, 0.6),
    ]
    kept = suppress(faces, 0.3)
    assert [f.score for f in kept] == [0.9, 0.6]


def squares(image: npt.NDArray[np.uint8]) -> list[Face]:
    """A stand-in detector: white squares that do not touch the image border are faces."""
    count, _, stats, _ = cv2.connectedComponentsWithStats((image[:, :, 0] == 255).astype(np.uint8))
    height, width = image.shape[:2]
    faces = []
    for left, top, w, h, area in stats[1:count]:
        if left == 0 or top == 0 or left + w == width or top + h == height:
            continue
        corners = np.array([[left, top], [left + w - 1, top], [left, top + h - 1]], float)
        landmarks = np.vstack([corners, corners[:2] + 1.0])
        faces.append(Face((float(left), float(top), float(w), float(h)), landmarks, float(area)))
    return sorted(faces, key=lambda f: -f.score)


def test_detect_tiled_finds_the_same_faces_as_the_whole_image() -> None:
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for k in range(12):
        size = 10 + k
        x, y = 20 + (k % 6) * 75 + int(rng.integers(0, 20)), 30 + (k // 6) * 140
        image[y : y + size, x : x + size] = 255
    whole = detect_tiled(
        squares, image, upsample=4, max_pixels=10**9, overlap=40, nms_threshold=0.3
    )
    tiled = detect_tiled(
        squares, image, upsample=4, max_pixels=(120 * 4) ** 2, overlap=40, nms_threshold=0.3
    )
    assert len(whole) == 12
    assert sorted(f.box for f in tiled) == sorted(f.box for f in whole)
    for a, b in zip(
        sorted(tiled, key=lambda f: f.box), sorted(whole, key=lambda f: f.box), strict=True
    ):
        np.testing.assert_allclose(a.landmarks, b.landmarks)


def test_box_landmarks_and_their_fit() -> None:
    assert set(YUNET_BOX_LANDMARKS) == {4, 8}
    for normalized in YUNET_BOX_LANDMARKS.values():
        np.testing.assert_allclose(mirror(normalized), normalized, atol=1e-12)
    normalized = YUNET_BOX_LANDMARKS[4]
    points = box_landmarks((10.0, 20.0, 40.0, 50.0), normalized)
    np.testing.assert_allclose(points, np.array([10.0, 20.0]) + normalized * [40, 50] - 0.5)
    with pytest.raises(ValueError, match="positive"):
        box_landmarks((0, 0, 0, 10), normalized)
    # faces whose landmarks sit at the same box-relative positions give those positions back
    faces = [
        Face((x, y, s, s), box_landmarks((x, y, s, s), normalized), 0.9)
        for x, y, s in ((0, 0, 10), (50, 20, 30), (7, 3, 64))
    ]
    np.testing.assert_allclose(fit_box_landmarks(faces), normalized, atol=1e-9)
    with pytest.raises(ValueError):
        fit_box_landmarks([])


def test_yunet_templates_are_symmetric() -> None:
    assert set(YUNET_TEMPLATES) == {4, 8}
    for template in YUNET_TEMPLATES.values():
        assert template.shape == (5, 2)
        np.testing.assert_allclose(mirror(template), template, atol=1e-12)
        # close to the CelebA template: the conventions differ by a few hundredths
        assert np.abs(template - CELEBA_TEMPLATE).max() < 0.05


def test_make_scene_places_the_landmarks() -> None:
    image = np.full((218, 178, 3), 40, dtype=np.uint8)
    x, y = CELEBA_POINTS[0].astype(int)
    image[y - 4 : y + 5, x - 4 : x + 5] = 255  # a bright square on the image-left eye
    scene, points = make_scene(image, CELEBA_POINTS, 8.0)

    assert interocular(points) == pytest.approx(8.0, rel=0.02)
    assert scene.shape[0] == 3 * round(218 * 8.0 / interocular(CELEBA_POINTS))
    brightest = np.unravel_index(np.argmax(scene[:, :, 0]), scene.shape[:2])
    assert np.abs(np.array(brightest[::-1]) - points[0]).max() <= 1.0
    # the face image sits in the middle third, the padding has its mean colour
    third_y, third_x = scene.shape[0] // 3, scene.shape[1] // 3
    face = scene[third_y : 2 * third_y, third_x : 2 * third_x]
    assert abs(int(scene[0, 0, 0]) - float(face[:, :, 0].mean())) <= 0.5


def test_degrade_is_seeded_and_keeps_shape() -> None:
    image = np.random.default_rng(1).integers(0, 256, (20, 16, 3), dtype=np.uint8)
    settings = SceneDegradation(blur=0.5, noise=3.0, jpeg=70)
    a = degrade(image, settings, np.random.default_rng(0))
    b = degrade(image, settings, np.random.default_rng(0))
    assert a.shape == image.shape and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)


def test_match_and_landmark_error() -> None:
    truth = CELEBA_POINTS
    near = Face((0, 0, 1, 1), truth + 2.0, 0.8)
    far = Face((0, 0, 1, 1), truth + 30.0, 0.95)
    assert landmark_error(truth + np.array([3.0, 4.0]), truth) == pytest.approx(
        5.0 / interocular(truth)
    )
    assert match([near], truth) is near
    assert match([near, far], truth) is None  # the highest score is the one used
    assert match([], truth) is None


def similarity(
    points: npt.NDArray[np.float64], angle: float, scale: float, shift: tuple[float, float]
) -> npt.NDArray[np.float64]:
    a = np.radians(angle)
    rotation = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.asarray(points @ rotation.T + np.array(shift), dtype=np.float64)


def test_fit_template_recovers_a_detector_convention() -> None:
    other = CELEBA_TEMPLATE + np.array(
        [[0.01, 0.0], [-0.01, 0.0], [0.0, 0.02], [0.02, 0.01], [-0.02, 0.01]]
    )
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(5):
        angle, scale = rng.uniform(-10, 10), rng.uniform(0.5, 2.0)
        shift = (rng.uniform(0, 50), rng.uniform(0, 50))
        truth = similarity(CELEBA_TEMPLATE * 128 - 0.5, angle, scale, shift)
        detected = similarity(other * 128 - 0.5, angle, scale, shift)
        pairs.append((detected, truth))
    np.testing.assert_allclose(fit_template(pairs), other, atol=1e-9)
    with pytest.raises(ValueError, match="no pairs"):
        fit_template([])


def test_alignment_error_units() -> None:
    truth = CELEBA_POINTS
    exact = alignment_error(truth, truth, CELEBA_TEMPLATE)
    assert exact.shift == pytest.approx(0.0, abs=1e-9)
    assert exact.rotation == pytest.approx(0.0, abs=1e-9)
    assert exact.scale == pytest.approx(1.0)
    # a source shift of (2, 0) moves the crop by the matrix's first column times 2 (per axis)
    crop_per_source = np.abs(alignment_matrix(truth, CELEBA_TEMPLATE, 128)[:, 0]).max()
    shifted = alignment_error(truth + np.array([2.0, 0.0]), truth, CELEBA_TEMPLATE)
    assert shifted.shift == pytest.approx(2.0 * crop_per_source / 128, rel=1e-6)
    centre = truth.mean(axis=0)
    turned = alignment_error(
        similarity(truth - centre, 3.0, 1.0, tuple(centre)), truth, CELEBA_TEMPLATE
    )
    assert turned.rotation == pytest.approx(3.0, rel=1e-6)
    grown = alignment_error(
        similarity(truth - centre, 0.0, 1.1, tuple(centre)), truth, CELEBA_TEMPLATE
    )
    assert grown.scale == pytest.approx(1.1, rel=1e-6)
    # a larger crop error moves the annotated points farther from the template
    assert grown.points > exact.points


def test_within_jitter_and_summary() -> None:
    jitter = JitterConfig(shift=0.03, rotation=5.0, scale=0.05)
    inside = AlignmentError(shift=0.02, rotation=1.0, scale=1.04, points=1.0)
    outside = AlignmentError(shift=0.02, rotation=1.0, scale=1.08, points=3.0)
    assert inside.within(jitter) and not outside.within(jitter)
    summary = summarize([inside, outside], jitter)
    assert summary["faces"] == 2 and summary["within_jitter"] == 0.5
    assert summary["points_mean"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="no alignment errors"):
        summarize([], jitter)


def test_measure_detection_report(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("measure_detection")
    summary = {
        "points_mean": 3.4,
        "shift_p50": 0.013,
        "shift_p95": 0.031,
        "rotation_p50": 1.4,
        "rotation_p95": 4.1,
        "scale_p50": 1.033,
        "scale_p95": 1.106,
        "within_jitter": 0.63,
    }
    grid = [
        {"eyes": 4, "upsample": 8, "degraded": degraded, "detected": rate, "nme_mean": nme}
        for degraded, rate, nme in ((False, 0.99, 0.13), (True, 0.94, None))
    ]
    template = [
        {
            "scale": 4,
            "template": YUNET_TEMPLATES[4].tolist(),
            "val": {
                "clean": {"detected": 1.0, "celeba_template": summary, "fitted_template": summary}
            },
        }
    ]
    text = script.render({"grid": grid, "template": template})
    assert "| 4 | ×8 | 0.990 / 0.940 | 0.130 / — |" in text
    assert (
        "| ×4 | clean | fitted | 1.000 | 3.40 | 0.013 / 0.031 | 1.40 / 4.10 | "
        "1.033 / 1.106 | 0.630 |" in text
    )
    args = script.parse_args([])
    assert args.scales == [4, 8] and args.upsample == [1, 2, 4, 8, 16]
