"""Tests for facehall.config and the configs in the repository."""

from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, ValidationError

from facehall.config import load_config
from facehall.data.config import DataConfig, DegradeConfig, JitterConfig
from facehall.train import ExperimentConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_repository_data_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACEHALL_DATA", raising=False)
    config = load_config(CONFIGS / "data" / "celeba_128.yaml", DataConfig)
    assert config.root == "/mnt/storage/AI/Face/celeba"
    assert config.hr_size == 128
    assert config.exclude_blurry == ["train"]
    assert config.jitter == JitterConfig(shift=0.03, rotation=5.0, scale=0.05)


def test_data_root_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACEHALL_DATA", "/data/faces")
    config = load_config(CONFIGS / "data" / "celeba_128.yaml", DataConfig)
    assert config.root == "/data/faces/celeba"


def test_repository_degrade_config_loads() -> None:
    config = load_config(CONFIGS / "degrade" / "bicubic.yaml", DegradeConfig)
    assert config == DegradeConfig(kind="bicubic", antialias=True, quantize=True)


def test_unknown_key_wrong_type_and_missing_value_fail(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("root: /x\nhr_sise: 64\n")
    with pytest.raises(ConfigKeyError):
        load_config(path, DataConfig)
    path.write_text("root: /x\nhr_size: large\n")
    with pytest.raises(ValidationError):
        load_config(path, DataConfig)
    path.write_text("hr_size: 64\n")
    with pytest.raises(MissingMandatoryValue):
        load_config(path, DataConfig)


def test_include_file_and_overrides(tmp_path: Path) -> None:
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "data.yaml").write_text("root: /x\nhflip: 0.5\nhr_size: 64\n")
    path = tmp_path / "train" / "experiment.yaml"
    path.parent.mkdir()
    path.write_text(
        "include:\n  data: ../parts/data.yaml\n"
        "name: e\nscale: 4\nmodel:\n  name: fnet\ntrain:\n  epochs: 1\ndata:\n  hflip: 0.0\n"
    )
    config = load_config(path, ExperimentConfig, ["data.hr_size=32"])
    assert (config.data.root, config.data.hflip, config.data.hr_size) == ("/x", 0.0, 32)
    path.write_text("include: [a.yaml]\nname: e\n")
    with pytest.raises(TypeError, match="must map section names"):
        load_config(path, ExperimentConfig)
    path.write_text("name: e\nscale: 4\nmodel:\n  name: fnet\ntrain:\n  epochs: 1\n")
    with pytest.raises(ConfigKeyError):
        load_config(path, ExperimentConfig, ["train.epoch=2"])
