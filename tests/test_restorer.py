"""Restoration backends (PR 5.2): PyTorch and ONNX Runtime, and the pipeline without PyTorch.

Tiny models only: no GPU, weights or dataset.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from facehall.align.template import CELEBA_TEMPLATE
from facehall.export import export_checkpoint, export_model
from facehall.marking import read_record
from facehall.models import build_model
from facehall.pipeline import Enhancer, ModelsConfig, PipelineConfig, bicubic_matrix
from facehall.restorer import (
    CPU,
    CUDA,
    OnnxRestorer,
    TorchRestorer,
    load_restorer,
    onnx_providers,
)
from facehall.train.checkpoint import save_checkpoint
from facehall.train.config import experiment_to_dict, load_experiment

# A face with 8 px between the eyes (the x4 input size), centred in a 64 x 64 photo.
FACE = CELEBA_TEMPLATE * 32 - 0.5 + 16.0


def tiny_checkpoint(folder: Path, scale: int = 4, run: str = "tiny-test") -> Path:
    config = load_experiment(
        f"configs/train/srresnet_x{scale}.yaml", ["model.residual_blocks=1", "model.channels=8"]
    )
    torch.manual_seed(1)
    model = build_model(config.model, config.scale, config.data.hr_size // config.scale)
    path = folder / run / "best.safetensors"
    path.parent.mkdir(parents=True)
    save_checkpoint(path, model, {"config": experiment_to_dict(config), "run": run})
    return path


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A tiny x4 checkpoint and its ONNX export."""
    folder = tmp_path_factory.mktemp("models")
    checkpoint = tiny_checkpoint(folder)
    return checkpoint, export_checkpoint(checkpoint, folder / "x4.onnx").path


def photo(size: int = 64, seed: int = 0) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 128 + 60 * np.sin(x / 5.0)[..., None] * np.cos(y / 7.0)[..., None]
    pixels: npt.NDArray[np.uint8] = np.clip(
        base + rng.normal(0, 5, (size, size, 3)), 0, 255
    ).astype(np.uint8)
    return pixels


@pytest.mark.parametrize(("size_in", "size_out"), [(32, 128), (16, 128), (8, 16), (12, 7)])
def test_bicubic_matrix_matches_pytorch(size_in: int, size_out: int) -> None:
    image = torch.rand(2, 3, size_in, size_in, generator=torch.Generator().manual_seed(0))
    expected = F.interpolate(image, size=(size_out, size_out), mode="bicubic", align_corners=False)
    m = bicubic_matrix(size_in, size_out)
    got = m @ image.numpy() @ m.T
    assert np.abs(got - expected.numpy()).max() < 1e-5


def test_onnx_restorer_matches_the_checkpoint(models: tuple[Path, Path]) -> None:
    checkpoint, onnx = models
    reference, exported = TorchRestorer.from_checkpoint(checkpoint), OnnxRestorer(onnx)
    for attribute in ("scale", "lr_size", "hr_size", "device"):
        assert getattr(exported, attribute) == getattr(reference, attribute)
    lr = np.random.default_rng(0).random((3, 3, 32, 32), dtype=np.float32)
    assert np.abs(exported(lr) - reference(lr)).max() < 1e-5
    info = exported.info
    assert {k: info[k] for k in reference.info} == reference.info  # the same source model
    assert info["format"] == "onnx" and len(info["onnx_sha256"]) == 64
    assert info["runtime"].startswith("onnxruntime ") and info["runtime"].endswith(CPU)


def test_load_restorer_picks_the_backend_by_suffix(models: tuple[Path, Path]) -> None:
    checkpoint, onnx = models
    assert isinstance(load_restorer(onnx, "cpu"), OnnxRestorer)
    assert isinstance(load_restorer(checkpoint, "cpu"), TorchRestorer)


def test_onnx_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: [CPU])
    assert onnx_providers("cpu") == onnx_providers("auto") == [CPU]
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        onnx_providers("cuda")
    with pytest.raises(ValueError, match="auto, cpu or cuda"):
        onnx_providers("cuda:1")
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA, CPU])
    cuda = [(CUDA, {"use_tf32": "0"}), CPU]  # TF32 off: full float32, like the export check
    assert onnx_providers("auto") == onnx_providers("cuda") == cuda
    assert onnx_providers("cpu") == [CPU]


def test_an_onnx_model_without_the_record_is_refused(tmp_path: Path) -> None:
    model = build_model(
        load_experiment("configs/train/srresnet_x4.yaml", ["model.residual_blocks=1"]).model, 4, 32
    )
    path = tmp_path / "foreign.onnx"
    export_model(model, 32).save(path)
    with pytest.raises(ValueError, match="not a facehall ONNX model"):
        OnnxRestorer(path)


@pytest.mark.parametrize("color_correction", [False, True])
def test_the_pipeline_gives_the_same_photo_on_both_backends(
    models: tuple[Path, Path], color_correction: bool
) -> None:
    checkpoint, onnx = models
    config = PipelineConfig(color_correction=color_correction)
    image = photo()
    outputs = [
        Enhancer.load(path, config, "cpu").enhance(image, landmarks=[FACE])
        for path in (checkpoint, onnx)
    ]
    assert all(r.faces[0].restored for r in outputs)
    difference = np.abs(outputs[0].image.astype(int) - outputs[1].image.astype(int))
    assert difference.max() <= 1 and (difference > 0).mean() < 1e-3
    assert set(outputs[1].seconds) == {"detect", "restore", "paste"}


def test_color_correction_matches_the_pytorch_version(models: tuple[Path, Path]) -> None:
    enhancer = Enhancer.load(models[0], PipelineConfig(), "cpu")
    rng = np.random.default_rng(3)
    lr = rng.random((2, 3, 32, 32), dtype=np.float32)
    sr = rng.random((2, 3, 128, 128), dtype=np.float32)
    # the PyTorch implementation of PR 4.2
    lr_t, sr_t = torch.from_numpy(lr), torch.from_numpy(sr)
    reference = F.interpolate(lr_t, size=(128, 128), mode="bicubic", align_corners=False)
    weights = torch.from_numpy(enhancer.mask)[None, None]
    shift = ((reference.clamp(0, 1) - sr_t) * weights).sum(dim=(2, 3), keepdim=True)
    expected = (sr_t + shift / weights.sum()).clamp(0, 1).numpy()
    assert np.abs(enhancer.correct_color(sr, lr) - expected).max() < 1e-5


def test_models_config_selects_the_backend() -> None:
    models = ModelsConfig(x4="a.safetensors", onnx_x4="a.onnx")
    assert models.path(4) == "a.safetensors" == models.checkpoint(4)
    onnx = ModelsConfig(backend="onnx", x4="a.safetensors", onnx_x4="a.onnx")
    assert onnx.path(4) == "a.onnx" and onnx.checkpoint(4) == "a.safetensors"
    with pytest.raises(ValueError, match="no restoration ONNX model configured for scale 8"):
        onnx.path(8)
    with pytest.raises(ValueError, match="models.backend"):
        ModelsConfig(backend="tensorrt")


BLOCK_TORCH = """
import importlib.abc, sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"{name} is blocked: this path must run without PyTorch")
        return None

sys.meta_path.insert(0, BlockTorch())
from facehall.cli import main

code = main(sys.argv[1:])
loaded = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
assert not loaded, loaded
sys.exit(code)
"""


def test_enhance_with_onnx_models_runs_without_pytorch(
    models: tuple[Path, Path], tmp_path: Path
) -> None:
    _, onnx = models
    source = tmp_path / "in.png"
    Image.fromarray(photo()).save(source)
    landmarks = ",".join(f"{v:.2f}" for v in FACE.ravel())
    folder = tmp_path / "exports"
    folder.mkdir()
    (folder / "x4.onnx").write_bytes(onnx.read_bytes())
    runs = [
        ["--ckpt", str(onnx), "-o", str(tmp_path / "a.png")],
        ["--scale", "4", "--set", "models.backend=onnx", "-o", str(tmp_path / "b.png")],
    ]
    for arguments in runs:
        result = subprocess.run(
            [sys.executable, "-c", BLOCK_TORCH, "enhance", str(source), *arguments]
            + ["--landmarks", landmarks, "--device", "cpu"],
            capture_output=True,
            text=True,
            env={**os.environ, "FACEHALL_MODELS": str(folder)},
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
    for name in ("a.png", "b.png"):
        rec = read_record(tmp_path / name)
        assert rec is not None and rec["model"]["format"] == "onnx"
        assert rec["model"]["name"] == "tiny-test" and rec["faces"][0]["restored"]


def test_compare_backends_script_summarizes_and_compares(load_script: Any, tmp_path: Path) -> None:
    script: ModuleType = load_script("compare_backends")
    per_scene = {
        "single_clean_0000.png": {
            "faces": 1,
            "restored": 1,
            "seconds": {"detect": 0.01, "restore": 0.002, "paste": 0.001, "total": 0.013},
        },
        "mosaic_00_clean.png": {
            "faces": 4,
            "restored": 3,
            "seconds": {"detect": 0.1, "restore": 0.03, "paste": 0.02, "total": 0.15},
        },
    }
    summary = script.summarize(per_scene)
    assert summary["single"]["median_total_ms"] == pytest.approx(13.0)
    assert summary["mosaic"]["restore_ms_per_face"] == pytest.approx(10.0)
    runs = {}
    for label, offset in (("a", 0), ("b", 2)):
        folder = tmp_path / label
        folder.mkdir()
        for name in per_scene:
            image = photo(seed=1).copy()
            image[0, 0, 0] = 100 + offset
            Image.fromarray(image).save(folder / name)
        (folder / "run.json").write_text(json.dumps({"scenes": per_scene}))
        runs[label] = folder
    result = script.compare(runs["b"], runs["a"], per_scene)
    assert result["max_level_diff"] == 2
    assert result["values_changed"] == pytest.approx(1 / (64 * 64 * 3))
    assert result["scenes_with_the_same_restored_faces"] == 1.0
    square = script.pad(photo(size=8), 12)
    assert square.shape == (12, 12, 3) and (square[2:10, 2:10] == photo(size=8)).all()
