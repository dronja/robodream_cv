"""``facehall eval``: load the protocol and the model, then run the evaluation.

A protocol with ``kind: realworld`` (no HR reference) runs ``evaluate_realworld``; every other
protocol runs ``evaluate_model``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import nn

from facehall.config import load_config
from facehall.eval.config import EvalConfig, validate_eval
from facehall.eval.realworld import KIND, RealWorldConfig, evaluate_realworld, validate_realworld
from facehall.eval.runner import evaluate_model
from facehall.models import ModelConfig, build_model, count_parameters
from facehall.train.checkpoint import load_checkpoint


def run_eval(
    config_path: Path,
    overrides: Sequence[str] = (),
    *,
    checkpoint: Path | None = None,
    baseline: str | None = None,
) -> Path:
    """Evaluate a checkpoint or a parameter-free baseline; returns the output directory."""
    if protocol_kind(config_path) == KIND:
        realworld = load_config(config_path, RealWorldConfig, overrides)
        validate_realworld(realworld)
        model, label, info = load_model(realworld, checkpoint=checkpoint, baseline=baseline)
        return evaluate_realworld(model, realworld, label, info)
    config = load_config(config_path, EvalConfig, overrides)
    validate_eval(config)
    model, label, info = load_model(config, checkpoint=checkpoint, baseline=baseline)
    return evaluate_model(model, config, label, info)


def protocol_kind(path: Path) -> str:
    """``kind`` of a protocol file; ``synthetic`` when it is not set."""
    raw = OmegaConf.load(path)
    return str(raw.get("kind", "synthetic")) if isinstance(raw, DictConfig) else "synthetic"


def load_model(
    config: EvalConfig | RealWorldConfig,
    *,
    checkpoint: Path | None = None,
    baseline: str | None = None,
) -> tuple[nn.Module, str, dict[str, Any]]:
    """The model, its label (output directory name) and the metadata recorded with results."""
    hr_size = config.hr_size if isinstance(config, RealWorldConfig) else config.data.hr_size
    lr_size = hr_size // config.scale
    if (checkpoint is None) == (baseline is None):
        raise ValueError("pass exactly one of a checkpoint or a baseline model name")
    if baseline is not None:
        model = build_model(ModelConfig(name=baseline), config.scale, lr_size)
        if count_parameters(model):
            raise ValueError(f"model {baseline!r} has trainable parameters: evaluate a checkpoint")
        return model, baseline, {"name": baseline, "parameters": 0, "checkpoint": None}
    assert checkpoint is not None
    model, meta = load_checkpoint(checkpoint)
    trained = meta["config"]
    if (trained["scale"], trained["data"]["hr_size"]) != (config.scale, hr_size):
        raise ValueError(
            f"{checkpoint} is x{trained['scale']} at {trained['data']['hr_size']} px; the protocol "
            f"{config.name} is x{config.scale} at {hr_size} px"
        )
    label = f"{meta['run']}-{checkpoint.stem}"
    info = {
        "name": trained["model"]["name"],
        "parameters": count_parameters(model),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": meta["sha256"],
        "run": meta["run"],
        "epoch": meta["epoch"],
        "weights": meta["weights"],
        "smoke": meta["smoke"],
        "val_metrics": meta["metrics"],
    }
    return model, label, info
