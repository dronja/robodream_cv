"""Tests for facehall.eval: metrics on known pairs, protocol runs, report, weights download.

No network and no third-party weights: LPIPS is disabled and the recognition model is replaced
by a small deterministic embedder.
"""

import hashlib
import json
import math
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import pytest
import torch

from facehall.align import CELEBA_TEMPLATE
from facehall.cli import main
from facehall.data.config import DataConfig, DegradeConfig, NoiseConfig, StageConfig
from facehall.data.dataset import AlignedFaces
from facehall.eval.cli import load_model
from facehall.eval.config import EvalConfig, MetricsConfig, validate_eval
from facehall.eval.metrics import (
    ARCFACE_SIZE,
    ARCFACE_TEMPLATE_112,
    IdentitySimilarity,
    psnr_y,
    recognition_matrix,
    rgb_to_y,
    ssim_y,
)
from facehall.eval.report import collect, comparison_grid, render
from facehall.eval.runner import Metric, evaluate_model, identity_fingerprint
from facehall.models import ModelConfig, build_model
from facehall.train import ExperimentConfig, SmokeConfig, TrainConfig, train
from facehall.weights import Asset, download

ROWS = (
    [("train", False, (i % 3 - 1, i % 2)) for i in range(4)]
    + [("val", False, (i, -i)) for i in range(2)]
    + [("test", i == 1, (i - 2, 1 - i)) for i in range(5)]
)
TEST_SAMPLES = 5


class MeanColours:
    """A deterministic stand-in for a face-recognition model: 4x4 block means per channel."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, faces: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating]:
        self.shapes.append(faces.shape)
        blocks = faces.reshape(len(faces), 4, 28, 4, 28, 3).mean(axis=(2, 4))
        features: npt.NDArray[np.floating] = blocks.reshape(len(faces), -1) - 127.5
        return features


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory, make_dataset: Callable[..., Path]) -> Path:
    return make_dataset(tmp_path_factory.mktemp("faces"), ROWS)


def protocol(root: Path, output: Path, **values: Any) -> EvalConfig:
    config = EvalConfig(
        name="tiny_x4",
        scale=4,
        device="cpu",
        workers=0,
        batch_size=2,
        output=str(output),
        grid_samples=2,
        subgroups=["attr_male", "attr_young"],
        data=DataConfig(root=str(root), hr_size=32),
        metrics=MetricsConfig(lpips="", identity=""),
    )
    return replace(config, **values)


def identity_metric() -> IdentitySimilarity:
    return IdentitySimilarity(MeanColours(), CELEBA_TEMPLATE, 32)


def test_rgb_to_y_uses_bt601_studio_range() -> None:
    images = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1).expand(2, 3, 2, 2)
    torch.testing.assert_close(rgb_to_y(images)[:, 0, 0, 0], torch.tensor([16.0, 235.0]))
    red = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
    torch.testing.assert_close(rgb_to_y(red).flatten(), torch.tensor([16.0 + 65.481]))


def test_psnr_on_a_known_offset() -> None:
    hr = torch.full((2, 3, 8, 8), 0.5)
    offset = 10 / 255
    expected = 20 * math.log10(255 / (10 * 219 / 255))  # Y changes by 219/255 per RGB level
    torch.testing.assert_close(psnr_y(hr + offset, hr), torch.full((2,), expected))
    assert psnr_y(hr, hr).min() > 140  # identical images: capped, not inf


def test_ssim() -> None:
    generator = torch.Generator().manual_seed(0)
    hr = torch.rand(2, 3, 32, 32, generator=generator)
    torch.testing.assert_close(ssim_y(hr, hr), torch.ones(2))
    noisy = (hr + 0.1 * torch.randn(hr.shape, generator=generator)).clamp(0, 1)
    assert (ssim_y(noisy, hr) < 0.95).all()


def test_recognition_matrix_maps_the_template_to_arcface() -> None:
    for size in (32, 128):
        matrix = recognition_matrix(CELEBA_TEMPLATE, size)
        points = np.asarray(CELEBA_TEMPLATE) * size - 0.5
        mapped = points @ matrix[:, :2].T + matrix[:, 2]
        assert np.abs(mapped - ARCFACE_TEMPLATE_112).max() < 3.0  # least-squares fit of 5 points
        assert abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))) < 1.0


def test_identity_similarity() -> None:
    embedder = MeanColours()
    metric = IdentitySimilarity(embedder, CELEBA_TEMPLATE, 32)
    generator = torch.Generator().manual_seed(0)
    hr = torch.rand(3, 3, 32, 32, generator=generator)
    torch.testing.assert_close(metric(hr, hr), torch.ones(3, dtype=torch.float64))
    assert embedder.shapes[0] == (3, ARCFACE_SIZE, ARCFACE_SIZE, 3)
    assert (metric(1 - hr, hr) < 0.5).all()


def test_protocol_run(root: Path, tmp_path: Path) -> None:
    config = protocol(root, tmp_path)
    model = build_model(ModelConfig(name="bicubic"), 4, 8)
    out = evaluate_model(
        model,
        config,
        "bicubic",
        {"name": "bicubic", "parameters": 0},
        metrics=({"psnr_y": psnr_y, "ssim_y": ssim_y}, identity_metric()),
    )
    assert out == tmp_path / "tiny_x4" / "test" / "bicubic"
    summary = json.loads((out / "metrics.json").read_text())
    subsets = summary["subsets"]
    assert subsets["all"]["count"] == TEST_SAMPLES
    assert subsets["non_blurry"]["count"] == TEST_SAMPLES - 1
    assert subsets["male"]["count"] + subsets["not_male"]["count"] == TEST_SAMPLES
    assert set(subsets["all"]) == {"count", "psnr_y", "ssim_y", "identity"}
    table = pq.read_table(out / "per_image.parquet").to_pydict()
    assert table["index"] == list(range(TEST_SAMPLES))
    assert np.mean(table["psnr_y"]) == pytest.approx(subsets["all"]["psnr_y"]["mean"])
    assert all(0.0 < v <= 1.0 + 1e-9 for v in table["identity"])
    assert len(list((out / "samples").glob("*.png"))) == 2
    assert len(list((out.parent / "reference" / "hr").glob("*.png"))) == 2
    assert summary["model"]["name"] == "bicubic" and summary["metric_setup"]["identity"] is None


def test_identity_reference_cache(root: Path, tmp_path: Path) -> None:
    weights = tmp_path / "recognition.onnx"
    weights.write_bytes(b"fingerprinted only")
    config = protocol(root, tmp_path, metrics=MetricsConfig(lpips="", identity=str(weights)))
    model = build_model(ModelConfig(name="bicubic"), 4, 8)

    def run(config: EvalConfig, label: str) -> tuple[list[float], int]:
        embedder = MeanColours()
        metric = IdentitySimilarity(embedder, CELEBA_TEMPLATE, 32)
        out = evaluate_model(
            model, config, label, {"name": "bicubic", "parameters": 0}, metrics=({}, metric)
        )
        identity: list[float] = (
            pq.read_table(out / "per_image.parquet").column("identity").to_pylist()
        )
        return identity, sum(shape[0] for shape in embedder.shapes)

    first, embedded_first = run(config, "a")
    second, embedded_second = run(config, "b")
    assert (embedded_first, embedded_second) == (2 * TEST_SAMPLES, TEST_SAMPLES)  # HR cached
    assert second == pytest.approx(first)
    changed = replace(config, data=replace(config.data, interpolation="bilinear"))
    _, embedded_changed = run(changed, "c")
    assert embedded_changed == 2 * TEST_SAMPLES  # other HR crops: the cache is not used


def test_hr_fingerprint_follows_degradations_that_change_the_target(
    root: Path, tmp_path: Path
) -> None:
    base = protocol(root, tmp_path)

    def fingerprint(degrade: DegradeConfig) -> str:
        config = replace(base, degrade=degrade)
        dataset = AlignedFaces(
            config.data, config.degrade, config.split, config.scale, augment=False
        )
        return identity_fingerprint(dataset, config)

    bicubic = fingerprint(DegradeConfig())
    noise = StageConfig(noise=NoiseConfig(prob=1.0, gaussian=1.0, sigma=[5.0, 5.0]))
    assert fingerprint(DegradeConfig(kind="pipeline", first=noise)) == bicubic
    assert fingerprint(DegradeConfig(kind="pipeline", grayscale=0.5)) != bicubic


def test_checkpoint_model_report_and_grid(root: Path, tmp_path: Path) -> None:
    experiment = ExperimentConfig(
        name="tiny",
        scale=4,
        device="cpu",
        output=str(tmp_path / "runs"),
        data=DataConfig(root=str(root), hr_size=32),
        model=ModelConfig(name="fnet", channels=4, residual_blocks=1),
        train=TrainConfig(epochs=1, batch_size=2, precision="fp32", workers=0),
        smoke=SmokeConfig(),
    )
    checkpoint = train(experiment) / "best.safetensors"
    config = protocol(root, tmp_path / "eval")
    model, label, info = load_model(config, checkpoint=checkpoint)
    assert label == f"{checkpoint.parent.name}-best" and info["name"] == "fnet"
    assert info["epoch"] == 1 and info["parameters"] > 0
    with pytest.raises(ValueError, match="the protocol tiny_x4 is x8"):
        load_model(replace(config, scale=8), checkpoint=checkpoint)
    with pytest.raises(ValueError, match="has trainable parameters"):
        load_model(config, baseline="fnet")

    metrics: tuple[dict[str, Metric], IdentitySimilarity | None] = (
        {"psnr_y": psnr_y, "ssim_y": ssim_y},
        identity_metric(),
    )
    evaluate_model(model, config, label, info, metrics=metrics)
    bicubic, name, base = load_model(config, baseline="bicubic")
    evaluate_model(bicubic, config, name, base, metrics=metrics)
    summaries = collect(tmp_path / "eval")
    assert [s["label"] for s in summaries] == ["bicubic", label]
    text = render(summaries)
    assert "## tiny_x4 — test (5 images, ×4)" in text
    bicubic_row = next(line for line in text.splitlines() if line.startswith("| bicubic | 0 |"))
    psnr = summaries[0]["subsets"]["all"]["psnr_y"]["mean"]
    assert f"{psnr:.2f}" in bicubic_row
    assert "| `Male` = 1 |" in text
    grid = comparison_grid(summaries, "tiny_x4", "test")
    assert grid.size == (32 * 4, 32 * 2)  # LR, bicubic, fnet, HR; 2 samples


def test_eval_cli(root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "protocol.yaml"
    config.write_text(
        f"name: cli_x4\nscale: 4\ndevice: cpu\nworkers: 0\noutput: {tmp_path / 'eval'}\n"
        f"data:\n  root: {root}\n  hr_size: 32\nmetrics:\n  lpips: ''\n  identity: ''\n"
    )
    assert main(["eval", "--config", str(config), "--model", "bicubic", "--split", "test"]) == 0
    out = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert out == tmp_path / "eval" / "cli_x4" / "test" / "bicubic"
    assert json.loads((out / "metrics.json").read_text())["subsets"]["all"]["count"] == 5
    with pytest.raises(SystemExit):
        main(["eval", "--config", str(config)])  # needs --ckpt or --model


def test_invalid_protocol_settings(root: Path, tmp_path: Path) -> None:
    config = protocol(root, tmp_path)
    for bad, message in (
        (replace(config, split="holdout"), "unknown split"),
        (replace(config, metrics=MetricsConfig(lpips="resnet")), "unknown LPIPS network"),
        (replace(config, batch_size=0), "batch_size must be"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_eval(bad)


def test_download_checks_size_and_checksum(tmp_path: Path) -> None:
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pack/model.onnx", b"weights")
        zf.writestr("pack/other.onnx", b"unused")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    asset = Asset("pack", archive.as_uri(), archive.stat().st_size, digest, ("model.onnx",), "test")
    target = download(asset, tmp_path / "weights")
    assert sorted(p.name for p in target.iterdir()) == ["SOURCE.json", "model.onnx"]
    source = json.loads((target / "SOURCE.json").read_text())
    assert source["archive_sha256"] == digest
    assert source["files"]["model.onnx"] == hashlib.sha256(b"weights").hexdigest()
    with pytest.raises(ValueError, match="expected"):
        download(replace(asset, sha256="0" * 64), tmp_path / "other")
    with pytest.raises(ValueError, match="lacks"):
        download(replace(asset, members=("missing.onnx",)), tmp_path / "third")


def test_reproduce_script_parses() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_phase1.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
