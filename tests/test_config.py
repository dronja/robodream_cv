"""Tests for facehall.config and the configs in the repository."""

from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, ValidationError

from facehall.config import load_config
from facehall.data.config import DataConfig, DegradeConfig, JitterConfig
from facehall.data.degrade import build_degradation
from facehall.eval.cli import protocol_kind
from facehall.eval.config import EvalConfig, validate_eval
from facehall.eval.realworld import KIND, RealWorldConfig, validate_realworld
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


LEVELS = ("light", "medium", "heavy")


def test_test_levels_are_graded() -> None:
    """Every level's ranges lie above the previous level's (stronger), without overlap."""
    configs = [
        load_config(CONFIGS / "degrade" / "levels" / f"{n}.yaml", DegradeConfig) for n in LEVELS
    ]
    training = load_config(CONFIGS / "degrade" / "realistic.yaml", DegradeConfig)
    for weaker, stronger in zip(configs, configs[1:], strict=False):
        a, b = weaker.first, stronger.first
        assert a.blur.sigma[1] <= b.blur.sigma[0]
        assert a.blur.motion_length[1] <= b.blur.motion_length[0]
        assert a.blur.defocus_radius[1] <= b.blur.defocus_radius[0]
        assert a.noise.sigma[1] <= b.noise.sigma[0]
        assert a.noise.peak[0] >= b.noise.peak[1]  # fewer photons: more noise
        assert a.jpeg.quality[0] >= b.jpeg.quality[1]
    for config in configs:
        first = config.first
        assert (first.blur.prob, first.noise.prob, first.jpeg.prob) == (1.0, 1.0, 1.0)
        assert config.grayscale == 0.0
        # no stronger than the training mix
        assert first.blur.sigma[1] <= training.first.blur.sigma[1]
        assert first.noise.sigma[1] <= training.first.noise.sigma[1]
        assert first.noise.peak[0] >= training.first.noise.peak[0]
        assert first.jpeg.quality[0] >= training.first.jpeg.quality[0]
        build_degradation(config)


def test_repository_eval_protocols_load() -> None:
    names = sorted(p.name for p in (CONFIGS / "eval").glob("*.yaml"))
    assert len(names) == 10
    for path in sorted((CONFIGS / "eval").glob("*.yaml")):
        if protocol_kind(path) == KIND:
            real = load_config(path, RealWorldConfig)
            validate_realworld(real)
            assert real.name == f"wider_real_x{real.scale}" and real.split == "test"
            assert real.root.endswith("wider_face/realworld") and real.reference_split == "val"
            assert real.metrics.inception.endswith("weights-inception-2015-12-05-6726825d.pth")
            assert real.inputs_sha256.get("test"), f"{path.name}: inputs are not frozen"
            continue
        config = load_config(path, EvalConfig)
        validate_eval(config)
        assert config.split == "test" and config.name.endswith(f"_x{config.scale}")
        assert config.level in ("bicubic", *LEVELS)
        assert config.name == f"celeba128_{config.level}_x{config.scale}"
        assert config.inputs_sha256.get("test"), f"{path.name}: inputs are not frozen"
