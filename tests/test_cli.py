from collections.abc import Callable
from pathlib import Path

import pytest

import facehall
from facehall.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert facehall.__version__ in capsys.readouterr().out


def test_no_args_returns_zero() -> None:
    assert main([]) == 0


def test_train_smoke_run(
    tmp_path: Path, make_dataset: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [("train", False, (i, 0)) for i in range(4)] + [("val", False, (0, i)) for i in range(2)]
    root = make_dataset(tmp_path / "faces", rows)
    model = Path(__file__).resolve().parents[1] / "configs" / "model" / "fnet.yaml"
    config = tmp_path / "tiny.yaml"
    config.write_text(
        f"include:\n  model: {model}\n"
        f"name: tiny\nscale: 4\ndevice: cpu\noutput: {tmp_path / 'runs'}\n"
        f"data:\n  root: {root}\n  hr_size: 32\n"
        "train:\n  epochs: 5\n  batch_size: 2\n  precision: fp32\n  workers: 0\n"
        "smoke:\n  train_samples: 2\n  val_samples: 1\n  epochs: 1\n"
    )
    overrides = ["--set", "model.channels=4", "--set", "model.residual_blocks=1"]
    assert main(["train", "--config", str(config), "--smoke", *overrides]) == 0
    run = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert run.parent == tmp_path / "runs" and run.name.endswith("-smoke")
    assert (run / "best.safetensors").is_file()
    assert "channels: 4" in (run / "config.yaml").read_text()


def test_train_resume_rejects_other_options(tmp_path: Path) -> None:
    for extra in (["--smoke"], ["--set", "seed=1"]):
        with pytest.raises(SystemExit) as exc:
            main(["train", "--resume", str(tmp_path), *extra])
        assert exc.value.code == 2
    with pytest.raises(SystemExit):
        main(["train", "--resume", str(tmp_path), "--config", str(tmp_path / "c.yaml")])
    with pytest.raises(SystemExit):
        main(["train"])
