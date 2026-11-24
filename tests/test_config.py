"""Tests for facehall.config and the configs in the repository."""

from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, ValidationError

from facehall.config import load_config
from facehall.data.config import DataConfig, DegradeConfig, JitterConfig

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
