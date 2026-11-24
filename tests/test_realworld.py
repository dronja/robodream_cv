"""Tests for the real-world evaluation: no-reference metrics, dataset, protocol run, report.

No third-party weights: the Inception features are replaced by small image statistics.
"""

import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from facehall.cli import main
from facehall.data.config import DataConfig
from facehall.data.manifest import UNKNOWN_IDENTITY, write_manifest
from facehall.data.realworld import RealFaces
from facehall.data.wider import LICENSE, SOURCE, manifest_schema
from facehall.eval.noref import (
    downscale,
    fid,
    frechet_distance,
    kid,
    lr_psnr_y,
    polynomial_mmd,
)
from facehall.eval.realworld import (
    NoRefMetricsConfig,
    RealWorldConfig,
    evaluate_realworld,
    realworld_digest,
    validate_realworld,
)
from facehall.eval.report import collect, comparison_grid, render
from facehall.eval.runner import CACHE, InputsChangedError
from facehall.models import ModelConfig, build_model

HR = 32
# (scale, split, blur, occlusion) of every synthetic face
FACES = (
    [(4, "test", i % 3, i % 2) for i in range(7)]
    + [(4, "val", 1, 0) for _ in range(3)]
    + [(8, "test", 2, 0) for _ in range(4)]
)
CELEBA_ROWS = [("val", False, (i, -i)) for i in range(4)] + [("test", False, (0, 0))]


class ImageStats:
    """A stand-in for InceptionV3: per-channel means and standard deviations, 2x2 block means."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert images.dtype == torch.uint8
        self.calls += 1
        x = images.float() / 255.0
        blocks = F.adaptive_avg_pool2d(x, 2).flatten(1)
        return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3)), blocks], dim=1)


def write_realworld(root: Path) -> Path:
    """PNG crops and a manifest in the layout of ``scripts/prepare_wider.py``."""
    rng = np.random.default_rng(0)
    rows = []
    counters: dict[tuple[int, str], int] = {}
    for i, (scale, split, blur, occlusion) in enumerate(FACES):
        size = HR // scale
        index = counters.get((scale, split), 0)
        counters[(scale, split)] = index + 1
        path = f"x{scale}/{split}/{index:06d}.png"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        base = rng.integers(40, 216, size=(1, 1, 3))
        pixels = np.clip(base + rng.integers(-30, 30, size=(size, size, 3)), 0, 255)
        Image.fromarray(pixels.astype(np.uint8)).save(root / path)
        rows.append(
            {
                "path": path,
                "sha256": f"{i:064x}",
                "split": split,
                "identity_id": UNKNOWN_IDENTITY,
                "width": size,
                "height": size,
                "source": SOURCE,
                "license": LICENSE,
                "scale": scale,
                "source_image": f"0--Event/{i // 2}.jpg",
                "event": "0--Event",
                "face_index": i % 2,
                "bbox": [0, 0, size, size],
                "landmarks": [0.0] * 10,
                "align_scale": 1.0,
                "align_angle": 0.0,
                "align_residual": 0.01,
                "padded": 0.1 if i == 0 else 0.0,
                "attr_blur": blur,
                "attr_expression": 0,
                "attr_illumination": 0,
                "attr_occlusion": occlusion,
                "attr_pose": int(i == 1),
            }
        )
    write_manifest(pa.Table.from_pylist(rows, schema=manifest_schema()), root)
    return root


@pytest.fixture(scope="module")
def real_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_realworld(tmp_path_factory.mktemp("real"))


@pytest.fixture(scope="module")
def celeba_root(
    tmp_path_factory: pytest.TempPathFactory, make_dataset: Callable[..., Path]
) -> Path:
    return make_dataset(tmp_path_factory.mktemp("celeba"), CELEBA_ROWS)


def protocol(real_root: Path, celeba_root: Path, output: Path, **values: Any) -> RealWorldConfig:
    config = RealWorldConfig(
        name="tiny_real_x4",
        scale=4,
        root=str(real_root),
        hr_size=HR,
        device="cpu",
        workers=0,
        batch_size=3,
        output=str(output),
        grid_samples=2,
        reference=DataConfig(root=str(celeba_root), hr_size=HR),
        metrics=NoRefMetricsConfig(kid_subsets=4, kid_subset_size=3, kid_min_faces=3),
    )
    return replace(config, **values)


def bicubic(scale: int) -> torch.nn.Module:
    return build_model(ModelConfig(name="bicubic"), scale, HR // scale)


def test_lr_psnr_of_a_bicubic_upscale_is_high() -> None:
    lr = torch.linspace(0.2, 0.8, 8).view(1, 1, 1, 8).expand(2, 3, 8, 8).contiguous()
    sr = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
    assert downscale(sr, 8).shape == lr.shape
    assert (lr_psnr_y(sr, lr) > 35).all()
    assert (lr_psnr_y(torch.zeros_like(sr), lr) < 15).all()


def test_frechet_distance_of_diagonal_gaussians() -> None:
    mu1, mu2 = np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 0.0])
    var1, var2 = np.array([1.0, 4.0, 0.25]), np.array([9.0, 1.0, 0.25])
    expected = np.square(mu1 - mu2).sum() + (var1 + var2 - 2.0 * np.sqrt(var1 * var2)).sum()
    value = frechet_distance(mu1, np.diag(var1), mu2, np.diag(var2))
    assert value == pytest.approx(expected)
    assert frechet_distance(mu1, np.diag(var1), mu1, np.diag(var1)) == pytest.approx(0.0, abs=1e-9)


def test_fid_and_kid_on_samples() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(size=(400, 4)).astype(np.float32)
    same = rng.normal(size=(400, 4)).astype(np.float32)
    shifted = (rng.normal(size=(400, 4)) + 1.0).astype(np.float32)
    assert fid(real, real) == pytest.approx(0.0, abs=1e-6)
    assert fid(real, same) < 0.2 < 3.0 < fid(real, shifted)
    near, _ = kid(real, same, subsets=10, subset_size=100, seed=0)
    far, _ = kid(real, shifted, subsets=10, subset_size=100, seed=0)
    assert abs(near) < 0.05 < far
    assert kid(real, shifted, subsets=10, subset_size=100, seed=0)[0] == far  # deterministic


def test_polynomial_mmd_matches_the_definition() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))

    def k(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
        return float((a @ b / 3 + 1.0) ** 3)

    m = len(x)
    within = sum(k(x[i], x[j]) + k(y[i], y[j]) for i in range(m) for j in range(m) if i != j)
    across = sum(k(x[i], y[j]) for i in range(m) for j in range(m))
    expected = within / (m * (m - 1)) - 2.0 * across / m**2
    value = polynomial_mmd(torch.from_numpy(x), torch.from_numpy(y))
    assert value == pytest.approx(expected)
    # a full-size "subset" is a permutation, which leaves the estimate unchanged
    full, spread = kid(x.astype(np.float32), y.astype(np.float32), subsets=3, subset_size=9, seed=0)
    assert full == pytest.approx(expected, rel=1e-5) and spread == pytest.approx(0.0, abs=1e-9)


def test_real_faces(real_root: Path) -> None:
    test = RealFaces(real_root, "test", 4)
    assert len(test) == 7 and test.lr_size == 8
    sample = test[0]
    assert sample["lr"].shape == (3, 8, 8) and sample["lr"].dtype == torch.float32
    assert len(RealFaces(real_root, "val", 4)) == 3
    assert RealFaces(real_root, "test", 8).lr_size == 4
    with pytest.raises(ValueError, match="no crops"):
        RealFaces(real_root, "val", 8)


def test_protocol_run(real_root: Path, celeba_root: Path, tmp_path: Path) -> None:
    config = protocol(real_root, celeba_root, tmp_path)
    extractor = ImageStats()
    out = evaluate_realworld(
        bicubic(4), config, "bicubic", {"name": "bicubic", "parameters": 0}, extractor=extractor
    )
    summary = json.loads((out / "metrics.json").read_text())
    assert summary["kind"] == "realworld" and summary["level"] == "real"
    assert summary["reference"]["samples"] == 4 and summary["reference"]["split"] == "val"
    subsets = summary["subsets"]
    assert {name: s["count"] for name, s in subsets.items()} == {
        "all": 7,
        "clear": 3,
        "normal_blur": 2,
        "heavy_blur": 2,
        "unoccluded": 4,
        "occluded": 3,
        "atypical_pose": 1,
        "unpadded": 6,
    }
    assert subsets["all"]["lr_psnr_y"]["mean"] > 25  # bicubic explains its input well
    assert math.isfinite(subsets["all"]["fid"]) and subsets["clear"]["fid"] is None
    assert subsets["all"]["kid"] is not None and subsets["atypical_pose"]["kid"] is None
    assert summary["inputs_sha256"] == realworld_digest(config)

    table = pq.read_table(out / "per_image.parquet")
    assert table.num_rows == 7
    assert {"lr_psnr_y", "attr_blur", "source_image", "path"} <= set(table.column_names)
    samples = sorted((out / "samples").glob("*.png"))
    assert len(samples) == 2 and Image.open(samples[0]).size == (HR, HR)
    assert len(list((out.parent / "reference" / "lr").glob("*.png"))) == 2

    # The reference features are cached and reused by the next model.
    assert len(list((tmp_path / CACHE).glob("inception_ref-*.npz"))) == 1
    calls = extractor.calls
    evaluate_realworld(
        bicubic(4), config, "again", {"name": "bicubic", "parameters": 0}, extractor=extractor
    )
    assert extractor.calls - calls == math.ceil(7 / 3)  # the SR batches only


def test_frozen_real_inputs(real_root: Path, celeba_root: Path, tmp_path: Path) -> None:
    config = protocol(real_root, celeba_root, tmp_path, inputs_sha256={"test": "0" * 64})
    with pytest.raises(InputsChangedError, match="changed"):
        evaluate_realworld(bicubic(4), config, "bicubic", {"name": "bicubic", "parameters": 0})


def test_invalid_settings(real_root: Path, celeba_root: Path, tmp_path: Path) -> None:
    good = protocol(real_root, celeba_root, tmp_path)
    for values, message in (
        ({"kind": "synthetic"}, "protocol kind"),
        ({"reference_split": "dev"}, "reference split"),
        ({"scale": 3}, "does not divide"),
        ({"reference": DataConfig(root=str(celeba_root), hr_size=64)}, "reference HR size"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_realworld(replace(good, **values))
    with pytest.raises(ValueError, match="expects"):
        evaluate_realworld(
            bicubic(4),
            replace(good, hr_size=64, reference=DataConfig(root=str(celeba_root), hr_size=64)),
            "bicubic",
            {"name": "bicubic", "parameters": 0},
        )


def test_cli_report_and_grid(
    real_root: Path,
    celeba_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
) -> None:
    path = tmp_path / "real_x4.yaml"
    path.write_text(
        "\n".join(
            [
                "kind: realworld",
                "name: tiny_real_x4",
                "scale: 4",
                f"root: {real_root}",
                f"hr_size: {HR}",
                "device: cpu",
                "workers: 0",
                "grid_samples: 2",
                f"output: {tmp_path / 'eval'}",
                "reference:",
                f"  root: {celeba_root}",
                f"  hr_size: {HR}",
            ]
        )
        + "\n"
    )
    assert main(["eval", "--config", str(path), "--model", "bicubic"]) == 0
    out = Path(capsys.readouterr().out.strip().splitlines()[-1])
    summary = json.loads((out / "metrics.json").read_text())
    assert summary["subsets"]["all"]["fid"] is None  # no Inception weights configured

    digest = load_script("protocol_digest")
    assert digest.main([str(path)]) == 0
    assert summary["inputs_sha256"] in capsys.readouterr().out

    summaries = collect(tmp_path / "eval")
    text = render(summaries)
    assert "tiny_real_x4 — real-world faces, no reference" in text
    assert "| blur 0 (clear) | 3 |" in text
    assert "Summary by degradation level" not in text
    grid = comparison_grid(summaries, "tiny_real_x4", "test")
    assert grid.size == (2 * HR, 2 * HR)  # LR and one model, no HR column
