"""The inference pipeline (facehall.pipeline) and its end-to-end evaluation (eval.endtoend).

Models and the detector are stand-ins: no weights, no dataset.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from facehall.align import CELEBA_TEMPLATE, YUNET_BOX_LANDMARKS, YUNET_TEMPLATES, Face
from facehall.align.detect import box_landmarks
from facehall.align.similarity import scale_of
from facehall.align.warp import alignment_matrix, warp
from facehall.config import load_config
from facehall.data.manifest import read_manifest
from facehall.eval.detection import SceneDegradation, sample_rows
from facehall.eval.endtoend import VARIANTS, evaluate, samples
from facehall.pipeline import (
    Enhancer,
    MaskConfig,
    ModelsConfig,
    PipelineConfig,
    ellipse_mask,
    given_points,
    lr_crop,
    to_frame,
    upscale,
)
from facehall.restorer import TorchRestorer

# A face with 8 px between the eyes (the x4 input size) in a 64 x 64 image.
FACE = CELEBA_TEMPLATE * 32 - 0.5 + np.array([16.0, 16.0])


class Bicubic(nn.Module):
    """Upsamples like the background, so the pipeline should change little."""

    def __init__(self, scale: int, offset: float = 0.0) -> None:
        super().__init__()
        self.scale, self.offset = scale, offset

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return out + self.offset


class Constant(nn.Module):
    def __init__(self, scale: int, value: float) -> None:
        super().__init__()
        self.scale, self.value = scale, value

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        b, c, h, w = lr.shape
        return torch.full((b, c, h * self.scale, w * self.scale), self.value)


class StubDetector:
    def __init__(self, faces: Sequence[Face]) -> None:
        self.faces = list(faces)
        self.calls = 0

    def detect(self, image: npt.NDArray[np.uint8]) -> list[Face]:
        self.calls += 1
        return self.faces


def photo(size: int = 64, seed: int = 0) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 128 + 60 * np.sin(x / 5.0)[..., None] * np.cos(y / 7.0)[..., None]
    pixels: npt.NDArray[np.uint8] = np.clip(
        base + rng.normal(0, 5, (size, size, 3)), 0, 255
    ).astype(np.uint8)
    return pixels


def enhancer(model: nn.Module, **kwargs: object) -> Enhancer:
    config = PipelineConfig(**kwargs)  # type: ignore[arg-type]
    return Enhancer(TorchRestorer(model, 4), config, detector=StubDetector([]))


def test_default_config_file_matches_the_schema_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACEHALL_RUNS", "/runs")
    config = load_config("configs/pipeline/default.yaml", PipelineConfig)
    assert replace(config, models=ModelsConfig()) == PipelineConfig()
    # the models are the gate references (configs/gate.yaml)
    gate = Path("configs/gate.yaml").read_text()
    for scale in (4, 8):
        path = Path(config.models.checkpoint(scale))
        assert path.parent.parent == Path("/runs") and path.name == "best.safetensors"
        assert f"{path.parent.name}-best" in gate
    with pytest.raises(ValueError, match="scale 2"):
        config.models.checkpoint(2)


def test_config_validation() -> None:
    for kwargs in (
        {"max_downscale": 0.5},
        {"max_upscale": 0.9},
        {"batch_size": 0},
        {"max_output_pixels": 0},
        {"mask": MaskConfig(feather=0.0)},
    ):
        with pytest.raises(ValueError):
            PipelineConfig(**kwargs)


def test_ellipse_mask() -> None:
    mask = ellipse_mask(128, MaskConfig())
    assert mask.shape == (128, 128) and mask.dtype == np.float32
    assert mask.min() == 0.0 and mask.max() == 1.0
    assert mask[70, 64] == 1.0  # the centre (0.55 of the height)
    assert mask[0, 0] == mask[0, 127] == mask[127, 0] == 0.0
    np.testing.assert_array_equal(mask, mask[:, ::-1])  # left-right symmetric


def test_upscale_and_frame_coordinates() -> None:
    image = photo(16)
    frame = upscale(image, 4)
    assert frame.shape == (64, 64, 3)
    # pixel centre x of the input lands at (x + 0.5) * 4 - 0.5 of the frame
    np.testing.assert_allclose(to_frame([[0.0, 0.0], [15.0, 15.0]], 4), [[1.5, 1.5], [61.5, 61.5]])
    with pytest.raises(ValueError, match="five"):
        given_points([1.0, 2.0])


def test_lr_crop_scales() -> None:
    image = photo(128)
    crop, factor = lr_crop(image, FACE, CELEBA_TEMPLATE, 32)
    assert crop.shape == (32, 32, 3) and factor == pytest.approx(1.0)
    big = (FACE - 16 + 0.5) * 3 - 0.5 + 10  # the same face three times larger
    crop, factor = lr_crop(image, big, CELEBA_TEMPLATE, 32)
    assert crop.shape == (32, 32, 3) and factor == pytest.approx(1 / 3)
    # slightly larger than the crop scale: a 2x supersampled warp, close to the direct warp
    near = (FACE - 16 + 0.5) * 1.02 - 0.5 + 16
    crop, factor = lr_crop(image, near, CELEBA_TEMPLATE, 32)
    direct = warp(image, alignment_matrix(near, CELEBA_TEMPLATE, 32), 32)
    assert factor < 1.0 and np.abs(crop.astype(int) - direct.astype(int)).mean() < 6.0


def test_paste_back_leaves_the_background_outside_the_face() -> None:
    image = photo()
    result = enhancer(Constant(4, 1.0)).enhance(image, [FACE])
    background = upscale(image, 4)
    assert result.image.shape == (256, 256, 3) and result.scale == 4
    assert [f.restored for f in result.faces] == [True]
    assert result.faces[0].source == "given" and result.faces[0].score is None
    # the face centre is the model output (white), far corners are the background
    centre = to_frame(FACE, 4).mean(axis=0).round().astype(int)
    assert (result.image[centre[1], centre[0]] == 255).all()
    np.testing.assert_array_equal(result.image[:20, :20], background[:20, :20])
    np.testing.assert_array_equal(result.image[-20:, -20:], background[-20:, -20:])


def test_a_bicubic_model_barely_changes_the_frame() -> None:
    image = photo()
    result = enhancer(Bicubic(4)).enhance(image, [FACE])
    difference = np.abs(result.image.astype(int) - upscale(image, 4).astype(int))
    assert difference.mean() < 2.0


def test_too_large_faces_are_left_to_the_background() -> None:
    image = photo(128)
    big = (FACE - 16 + 0.5) * 3 - 0.5 + 10
    result = enhancer(Constant(4, 1.0), max_downscale=2.0).enhance(image, [big])
    assert result.faces[0].restored is False and "too large for the model" in str(
        result.faces[0].reason
    )
    assert result.faces[0].lr_scale == pytest.approx(1 / 3)
    np.testing.assert_array_equal(result.image, upscale(image, 4))


def test_too_small_faces_are_left_to_the_background() -> None:
    image = photo()
    small = (FACE - 32 + 0.5) / 2.5 - 0.5 + 32  # the face 2.5 times smaller, same centre
    result = enhancer(Constant(4, 1.0)).enhance(image, [small, FACE])
    tiny, normal = result.faces
    assert tiny.restored is False and "too small for the model" in str(tiny.reason)
    assert tiny.lr_scale == pytest.approx(2.5) and normal.restored
    lenient = enhancer(Constant(4, 1.0), max_upscale=3.0).enhance(image, [small])
    assert lenient.faces[0].restored


def test_detected_faces_use_the_yunet_template() -> None:
    image = photo()
    detected = alignment_matrix(FACE, CELEBA_TEMPLATE, 32)  # FACE in CelebA convention
    stub = StubDetector([Face((10, 10, 40, 40), FACE, 0.9)])
    result = Enhancer(TorchRestorer(Constant(4, 1.0), 4), PipelineConfig(), detector=stub).enhance(
        image
    )
    assert stub.calls == 1
    face = result.faces[0]
    assert face.source == "detected" and face.score == pytest.approx(0.9) and face.restored
    # the YuNet template differs from CelebA's, so the crop scale differs slightly from 1
    yunet = alignment_matrix(FACE, YUNET_TEMPLATES[4], 32)
    assert face.lr_scale == pytest.approx(scale_of(yunet)) != pytest.approx(scale_of(detected))
    assert face.to_dict()["source"] == "detected"


def test_restored_faces_report_their_region_and_outside_faces_are_skipped() -> None:
    image = photo()
    outside = FACE + 400.0  # far right of and below the 64 x 64 image
    result = enhancer(Constant(4, 1.0)).enhance(image, [FACE, outside])
    inside, away = result.faces
    assert inside.restored and inside.region is not None
    x, y, w, h = inside.region
    assert 0 <= x < x + w <= 256 and 0 <= y < y + h <= 256
    assert away.restored is False and "outside" in str(away.reason) and away.region is None
    assert inside.to_dict()["region"] == list(inside.region)


def test_face_boxes_use_a_detection_inside_or_the_box_landmarks() -> None:
    image = photo()
    box = (14.0, 14.0, 36.0, 40.0)
    inside = Face((15.0, 16.0, 34.0, 38.0), FACE, 0.4)
    elsewhere = Face((50.0, 50.0, 20.0, 20.0), FACE + 30, 0.9)  # centre outside the box
    tiny = Face((30.0, 30.0, 4.0, 4.0), FACE, 0.95)  # centre inside, far too small
    boxes = StubDetector([tiny, elsewhere, inside])
    model = Enhancer(
        TorchRestorer(Constant(4, 1.0), 4),
        PipelineConfig(),
        detector=StubDetector([]),
        box_detector=boxes,
    )
    face = model.enhance(image, boxes=[box]).faces[0]
    assert face.source == "box-detected" and face.score == pytest.approx(0.4)
    np.testing.assert_allclose(face.landmarks, FACE)
    assert boxes.calls == 1

    model = Enhancer(
        TorchRestorer(Constant(4, 1.0), 4),
        PipelineConfig(),
        detector=StubDetector([]),
        box_detector=StubDetector([elsewhere]),
    )
    face = model.enhance(image, boxes=[box]).faces[0]
    assert face.source == "box" and face.score is None and face.restored
    np.testing.assert_allclose(face.landmarks, box_landmarks(box, YUNET_BOX_LANDMARKS[4]))


def test_enhance_input_checks() -> None:
    model = enhancer(Bicubic(4), max_output_pixels=256 * 256 - 1)
    with pytest.raises(ValueError, match="MP"):
        model.enhance(photo())  # 64 x 64 at x4 is 256 x 256
    with pytest.raises(ValueError, match="not more than one"):
        enhancer(Bicubic(4)).enhance(photo(), landmarks=[FACE], boxes=[(0, 0, 10, 10)])
    with pytest.raises(ValueError, match="positive size"):
        enhancer(Bicubic(4)).enhance(photo(), boxes=[(0, 0, 0, 10)])


def test_detection_needs_a_template_for_the_scale() -> None:
    model = Constant(2, 1.0)
    with pytest.raises(ValueError, match="no YuNet template for scale 2"):
        Enhancer(TorchRestorer(model, 2), PipelineConfig(), detector=StubDetector([])).enhance(
            photo(32)
        )


def test_invalid_input() -> None:
    with pytest.raises(ValueError, match="RGB uint8"):
        enhancer(Bicubic(4)).enhance(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="multiple"):
        TorchRestorer(Bicubic(3), 3)


def test_color_correction_moves_the_mean_back() -> None:
    image = photo()
    plain = enhancer(Bicubic(4, offset=0.2)).enhance(image, [FACE]).image
    corrected = enhancer(Bicubic(4, offset=0.2), color_correction=True).enhance(image, [FACE]).image
    centre = to_frame(FACE, 4).mean(axis=0).round().astype(int)
    window = (slice(centre[1] - 8, centre[1] + 8), slice(centre[0] - 8, centre[0] + 8))
    background = upscale(image, 4)[window].astype(float).mean()
    assert abs(corrected[window].astype(float).mean() - background) < 3.0
    assert plain[window].astype(float).mean() - background > 30.0


def test_end_to_end_evaluation(
    tmp_path: Path,
    make_dataset: Callable[[Path, Sequence[tuple[str, bool, tuple[int, int] | None]]], Path],
) -> None:
    root = make_dataset(
        tmp_path, [("val", False, (0, 0)), ("val", False, (2, -1)), ("val", False, (-1, 1))]
    )
    rows = sample_rows(root, "val", 3)
    assert len(rows) == 3 and len(read_manifest(root)) == 3
    points = rows[0][1]
    stub = StubDetector([])  # never finds the face: "detected" stays bicubic
    model = Enhancer(TorchRestorer(Bicubic(4), 4), PipelineConfig(), detector=stub)
    results = evaluate(model, samples(rows, 4, 128, SceneDegradation(), seed=0))
    assert results["scenes"] == 3 and results["detected"] == 0.0 and results["restored"] == 0.0
    variants = results["variants"]
    assert set(variants) == set(VARIANTS)
    assert variants["detected"]["psnr_y"] == pytest.approx(variants["bicubic"]["psnr_y"])
    assert abs(variants["annotated"]["psnr_y"] - variants["bicubic"]["psnr_y"]) < 1.0
    assert points.shape == (5, 2)
    with pytest.raises(ValueError, match="no scenes"):
        evaluate(model, iter([]))
