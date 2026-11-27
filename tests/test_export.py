"""ONNX export and its parity check (PR 5.1), on tiny models: no GPU, weights or dataset."""

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from facehall import marking
from facehall.cli import build_parser, main
from facehall.export import (
    INPUT,
    MODEL_KEY,
    OPSET,
    OUTPUT,
    TOLERANCE,
    compare,
    export_checkpoint,
    export_model,
    parity_inputs,
    read_record,
    run,
    session,
)
from facehall.models import build_model
from facehall.train.checkpoint import load_checkpoint, read_metadata, save_checkpoint
from facehall.train.config import experiment_to_dict, load_experiment

REFERENCE_X4 = "srresnet_id_perc_robust_x4-20261126-121325"  # configs/pipeline/default.yaml


def tiny_checkpoint(folder: Path, scale: int, run: str = "tiny-test") -> Path:
    config = load_experiment(
        f"configs/train/srresnet_x{scale}.yaml", ["model.residual_blocks=1", "model.channels=8"]
    )
    torch.manual_seed(scale)
    model = build_model(config.model, config.scale, config.data.hr_size // config.scale)
    path = folder / run / "best.safetensors"
    path.parent.mkdir(parents=True)
    save_checkpoint(path, model, {"config": experiment_to_dict(config), "run": run})
    return path


@pytest.fixture(scope="module", params=[4, 8])
def exported(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
    folder = tmp_path_factory.mktemp(f"x{request.param}")
    checkpoint = tiny_checkpoint(folder, request.param)
    return checkpoint, export_checkpoint(checkpoint, folder / "model.onnx")


def test_export_matches_pytorch_for_any_batch(exported: Any) -> None:
    checkpoint, result = exported
    model, _ = load_checkpoint(checkpoint)
    onnx = session(result.path)
    lr_size = result.record["lr_size"]
    (given,), (returned,) = onnx.get_inputs(), onnx.get_outputs()
    assert (given.name, returned.name) == (INPUT, OUTPUT)
    assert isinstance(given.shape[0], str) and given.shape[1:] == [3, lr_size, lr_size]
    for count in (1, 5):
        lr = parity_inputs(lr_size, count=count, seed=7)
        sr = run(onnx, lr)
        hr = result.record["hr_size"]
        assert sr.shape == (count, 3, hr, hr)
        with torch.no_grad():
            expected = model(torch.from_numpy(lr)).numpy()
        assert np.abs(sr - expected).max() < 1e-5
        assert sr.min() >= 0.0 and sr.max() <= 1.0  # the eval-mode clamp is in the graph


def test_record_describes_the_source_checkpoint(exported: Any) -> None:
    checkpoint, result = exported
    record = read_record(result.path)
    assert record == result.record
    info = read_metadata(checkpoint)
    scale = info["config"]["scale"]
    assert record["name"] == "tiny-test" and record["checkpoint"] == "best.safetensors"
    assert record["sha256"] == info["sha256"] and record["architecture"] == "srresnet"
    assert (record["scale"], record["lr_size"], record["hr_size"]) == (scale, 128 // scale, 128)
    assert record["format"] == "onnx" and record["opset"] == OPSET
    assert record["notice"] == marking.NOTICE
    parity = record["parity"]
    assert parity["tolerance"] == TOLERANCE and parity["max_abs_diff"] == result.max_abs_diff
    assert result.max_abs_diff < 1e-5
    assert set(record["exported_with"]) == {"facehall", "torch", "onnxscript", "onnxruntime"}
    assert str(checkpoint.parent) not in json.dumps(record), "no local paths in the record"


def test_export_refuses_to_overwrite_and_needs_the_suffix(tmp_path: Path) -> None:
    checkpoint = tiny_checkpoint(tmp_path, 4)
    out = tmp_path / "x4.onnx"
    out.write_bytes(b"old")
    with pytest.raises(FileExistsError, match="--overwrite"):
        export_checkpoint(checkpoint, out)
    assert out.read_bytes() == b"old"
    with pytest.raises(ValueError, match=r"\.onnx"):
        export_checkpoint(checkpoint, tmp_path / "x4.bin")
    export_checkpoint(checkpoint, out, overwrite=True)
    assert read_record(out)["scale"] == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tiny-test", "x4.onnx"]


def test_failed_parity_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import facehall.export.onnx as onnx_export

    checkpoint = tiny_checkpoint(tmp_path, 8)
    monkeypatch.setattr(onnx_export, "max_abs_diff", lambda *args: 0.01)
    with pytest.raises(ValueError, match="differs from PyTorch by 1.00e-02"):
        export_checkpoint(checkpoint, tmp_path / "out" / "x8.onnx")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_a_model_without_the_record_is_refused(tmp_path: Path) -> None:
    model = build_model(
        load_experiment("configs/train/srresnet_x8.yaml", ["model.residual_blocks=1"]).model, 8, 16
    )
    path = tmp_path / "foreign.onnx"
    export_model(model, 16).save(path)
    with pytest.raises(ValueError, match=f"no '{MODEL_KEY}' metadata"):
        read_record(path)


class Shifted:
    """An ONNX Runtime stand-in that returns the model output with a few values changed."""

    def __init__(self, model: nn.Module, delta: float) -> None:
        self.model, self.delta = model, delta

    def run(self, names: list[str], feeds: dict[str, npt.NDArray[np.float32]]) -> list[Any]:
        with torch.no_grad():
            out = self.model(torch.from_numpy(feeds[INPUT])).numpy().copy()
        out[0, 0, 0, :4] = np.clip(out[0, 0, 0, :4] + self.delta, 0.0, 1.0)
        return [out]


def test_compare_counts_changed_levels() -> None:
    model = build_model(load_experiment("configs/train/fnet_x4.yaml").model, 4, 32).eval()
    lr = np.full((2, 3, 32, 32), 0.5, dtype=np.float32)
    same = compare(model, Shifted(model, 0.0), lr)
    assert same == {
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "max_level_diff": 0.0,
        "levels_changed": 0.0,
    }
    off = compare(model, Shifted(model, -3.0 / 255.0), lr)
    assert off["max_abs_diff"] == pytest.approx(3.0 / 255.0, abs=1e-6)
    assert off["max_level_diff"] == 3.0
    assert off["levels_changed"] == pytest.approx(4 / (2 * 3 * 128 * 128))


def test_cli_exports_the_reference_of_a_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tiny_checkpoint(tmp_path / "runs", 4, run=REFERENCE_X4)
    monkeypatch.setenv("FACEHALL_RUNS", str(tmp_path / "runs"))
    out = tmp_path / "models" / "x4.onnx"
    assert main(["export", "onnx", "--scale", "4", "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"exported {REFERENCE_X4} (x4, srresnet) to {out}" in printed
    assert "max abs diff from PyTorch" in printed and "tolerance 1e-03" in printed
    assert read_record(out)["name"] == REFERENCE_X4
    with pytest.raises(SystemExit) as stop:
        main(["export", "onnx", "--scale", "4", "-o", str(out)])
    assert stop.value.code == 2 and "--overwrite" in capsys.readouterr().err


def test_cli_defaults_match_the_module() -> None:
    args = build_parser().parse_args(["export", "onnx", "--ckpt", "x", "-o", "y.onnx"])
    assert (args.opset, args.tolerance) == (OPSET, TOLERANCE)


def test_measure_script_renders_its_table(load_script: Any) -> None:
    script: ModuleType = load_script("measure_onnx")
    row = {
        "max_abs_diff": 6.6e-7,
        "mean_abs_diff": 3.0e-8,
        "max_level_diff": 1.0,
        "levels_changed": 7e-6,
        "cuda_vs_cpu": None,
    }
    results = {
        "samples": 2,
        "split": None,
        "batch": 32,
        "repeats": 3,
        "protocols": {"celeba128_light_x4": row},
        "timing": {"torch_cpu_ms": 65.1, "onnxruntime_cpu_ms": 227.8},
    }
    text = script.render("label", results)
    assert "| celeba128_light_x4 | 6.6e-07 | 3.0e-08 | 1 | 0.0007% | — |" in text
    assert "torch_cpu 65.1 ms, onnxruntime_cpu 227.8 ms" in text
