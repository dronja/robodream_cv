"""Tests for facehall.train: CPU runs of a tiny model on a synthetic prepared dataset."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from facehall.config import load_config
from facehall.data.config import DataConfig, DegradeConfig, JitterConfig
from facehall.data.dataset import AlignedFaces
from facehall.models import ModelConfig
from facehall.train import (
    ExperimentConfig,
    ModelEMA,
    SmokeConfig,
    TrainConfig,
    load_checkpoint,
    load_experiment,
    read_metadata,
    save_checkpoint,
    train,
)
from facehall.train.config import (
    SchedulerConfig,
    experiment_from_dict,
    experiment_to_dict,
    validate,
)
from facehall.train.trainer import (
    EarlyStopping,
    EpochShuffle,
    evaluate,
    learning_rate,
    make_loader,
    resume,
    steps_per_epoch,
    train_epoch,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"
TRAIN_SAMPLES, VAL_SAMPLES = 8, 3
ROWS = [("train", False, (i % 3 - 1, i % 2)) for i in range(TRAIN_SAMPLES)] + [
    ("val", False, (i, -i)) for i in range(VAL_SAMPLES)
]


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory, make_dataset: Callable[..., Path]) -> Path:
    return make_dataset(tmp_path_factory.mktemp("faces"), ROWS)


def experiment(root: Path, output: Path, **train: Any) -> ExperimentConfig:
    """A tiny FNet (8 channels, 1 block) on 32 px HR crops, x4, on CPU."""
    settings = {"epochs": 2, "batch_size": 2, "precision": "fp32", "workers": 0, "log_every": 1}
    return ExperimentConfig(
        name="tiny",
        scale=4,
        device="cpu",
        output=str(output),
        data=DataConfig(root=str(root), hr_size=32, hflip=0.5, jitter=JitterConfig(0.03, 5, 0.05)),
        degrade=DegradeConfig(),
        model=ModelConfig(name="fnet", channels=8, residual_blocks=1),
        train=TrainConfig(**{**settings, **train}),
        smoke=SmokeConfig(train_samples=4, val_samples=2, epochs=1),
    )


def records(run: Path, event: str) -> list[dict[str, Any]]:
    lines = (run / "metrics.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["event"] == event]


def test_run_directory(root: Path, tmp_path: Path) -> None:
    run = train(experiment(root, tmp_path))
    assert run.parent == tmp_path and run.name.startswith("tiny-")
    names = {p.name for p in run.iterdir()}
    assert names == {
        "config.yaml",
        "run.json",
        "metrics.jsonl",
        "train.log",
        "summary.json",
        "last_state.pt",
        *(f"{c}.{s}" for c in ("last", "best") for s in ("safetensors", "json")),
    }
    epochs = records(run, "epoch")
    assert [r["epoch"] for r in epochs] == [1, 2]
    assert [r["step"] for r in epochs] == [4, 8]  # 8 samples, batch 2
    assert [r["step"] for r in records(run, "step")] == list(range(1, 9))
    info = json.loads((run / "run.json").read_text())
    assert info["data"]["train"]["used"] == TRAIN_SAMPLES
    assert info["data"]["val"]["used"] == VAL_SAMPLES
    assert info["versions"]["torch"] is not None and info["device"]["type"] == "cpu"
    assert "git_commit" in info and info["manifest"] is None
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "finished" and summary["epochs"] == 2
    best = min(epochs, key=lambda r: r["val_loss"])
    assert summary["best"]["epoch"] == best["epoch"]
    assert load_experiment(run / "config.yaml") == experiment(root, tmp_path)


def test_best_checkpoint_reproduces_its_validation(root: Path, tmp_path: Path) -> None:
    config = experiment(root, tmp_path)
    run = train(config)
    model, info = load_checkpoint(run / "best.safetensors")
    assert info["weights"] == "model" and info["config"] == experiment_to_dict(config)
    val = AlignedFaces(config.data, config.degrade, "val", config.scale, seed=config.seed)
    device = torch.device("cpu")
    result = evaluate(model, make_loader(val, 2, 0, device), F.mse_loss, device=device)
    assert result["loss"] == pytest.approx(info["metrics"]["val_loss"], rel=1e-6)


def test_training_is_reproducible(root: Path, tmp_path: Path) -> None:
    def weights(seed: int, output: str) -> dict[str, torch.Tensor]:
        config = replace(experiment(root, tmp_path / output), seed=seed)
        return load_file(train(config) / "last.safetensors")

    first, again, other = weights(0, "a"), weights(0, "b"), weights(1, "c")
    assert all(torch.equal(first[k], again[k]) for k in first)
    assert not all(torch.equal(first[k], other[k]) for k in first)


def test_smoke_run_with_workers_and_ema(root: Path, tmp_path: Path) -> None:
    config = experiment(root, tmp_path, workers=2, ema_decay=0.9)
    run = train(config, smoke=True)
    assert run.name.endswith("-smoke")
    assert [r["epoch"] for r in records(run, "epoch")] == [1]
    info = json.loads((run / "run.json").read_text())
    assert (info["smoke"], info["data"]["train"]["used"], info["data"]["val"]["used"]) == (
        True,
        4,
        2,
    )
    assert read_metadata(run / "best.safetensors")["weights"] == "ema"
    assert read_metadata(run / "last_ema.safetensors")["weights"] == "ema"


def test_early_stopping() -> None:
    stopper = EarlyStopping(patience=2)
    assert [stopper.update(v) for v in (1.0, 0.9, 0.9, 0.95)] == [True, True, False, False]
    assert stopper.stop
    never = EarlyStopping(patience=0)
    for value in (1.0, 2.0, 3.0):
        never.update(value)
    assert not never.stop


def test_epoch_shuffle() -> None:
    sampler = EpochShuffle(10, seed=3)
    sampler.set_epoch(1)
    first = list(sampler)
    assert sorted(first) == list(range(10)) and len(sampler) == 10
    assert list(sampler) == first
    sampler.set_epoch(2)
    assert list(sampler) != first


def test_gradient_accumulation_matches_one_large_batch() -> None:
    def trained(batches: list[dict[str, torch.Tensor]], grad_accum: int) -> torch.Tensor:
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 3, 3, padding=1)
        model = nn.Sequential(conv, nn.Upsample(scale_factor=2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        result = train_epoch(
            model, batches, optimizer, F.mse_loss, device=torch.device("cpu"), grad_accum=grad_accum
        )
        assert result.steps == 1 and result.images == 4
        return conv.weight.detach()

    generator = torch.Generator().manual_seed(1)
    lr = torch.rand(4, 3, 4, 4, generator=generator)
    hr = torch.rand(4, 3, 8, 8, generator=generator)
    whole = trained([{"lr": lr, "hr": hr}], grad_accum=1)
    halves = [{"lr": lr[:2], "hr": hr[:2]}, {"lr": lr[2:], "hr": hr[2:]}]
    torch.testing.assert_close(trained(halves, grad_accum=2), whole)


def test_ema_update() -> None:
    model = nn.BatchNorm2d(2)
    ema = ModelEMA(model, decay=0.75)
    average = ema.module
    assert isinstance(average, nn.BatchNorm2d)
    assert model.running_mean is not None and model.num_batches_tracked is not None
    with torch.no_grad():
        model.weight.fill_(5.0)
        model.running_mean.fill_(4.0)
        model.num_batches_tracked.fill_(7)
    ema.update(model)
    assert average.running_mean is not None and average.num_batches_tracked is not None
    torch.testing.assert_close(average.weight, torch.full((2,), 0.75 * 1.0 + 0.25 * 5.0))
    torch.testing.assert_close(average.running_mean, torch.full((2,), 0.25 * 4.0))
    assert int(average.num_batches_tracked) == 7
    assert not average.training and not any(p.requires_grad for p in average.parameters())


def test_checkpoint_round_trip_and_checksum(root: Path, tmp_path: Path) -> None:
    config = experiment(root, tmp_path)
    model, _ = load_checkpoint(train(config) / "last.safetensors")
    path = tmp_path / "copy.safetensors"
    save_checkpoint(path, model, {"config": experiment_to_dict(config), "note": "test"})
    loaded, info = load_checkpoint(path)
    assert info["note"] == "test" and info["file"] == "copy.safetensors"
    lr = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(loaded(lr), model(lr))
    path.write_bytes(path.read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError, match="does not match the sha256"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match=r"must end in \.safetensors"):
        save_checkpoint(tmp_path / "x.pt", model, {})


def test_config_round_trip(root: Path, tmp_path: Path) -> None:
    config = experiment(root, tmp_path)
    assert experiment_from_dict(json.loads(json.dumps(experiment_to_dict(config)))) == config


# train config -> its degradation config (configs/degrade/)
TRAIN_CONFIGS = {
    "fnet_x4": "bicubic",
    "fnet_x8": "bicubic",
    "cnet_x4": "bicubic",
    "cnet_x8": "bicubic",
    "fnet_x4_realistic": "realistic",
    "fnet_x8_realistic": "realistic",
    "fnet_x8_tuned": "realistic_x8",
    "srresnet_x4": "realistic",
    "srresnet_x8": "realistic_x8",  # Q7
}


@pytest.mark.parametrize(("name", "degrade"), TRAIN_CONFIGS.items())
def test_repository_train_configs(name: str, degrade: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACEHALL_DATA", raising=False)
    config = load_experiment(CONFIGS / f"{name}.yaml")
    model, rest = name.split("_x")
    scale = rest.partition("_")[0]
    assert (config.name, config.model.name, config.scale) == (name, model, int(scale))
    expected = load_config(CONFIGS.parent / "degrade" / f"{degrade}.yaml", DegradeConfig)
    assert config.degrade == expected
    assert config.data.root == "/mnt/storage/AI/Face/celeba" and config.data.hr_size == 128
    assert config.train.batch_size == 16
    recipe = (config.train.loss, config.train.optimizer.lr, config.train.scheduler.name)
    if model == "srresnet":
        assert recipe == ("l1", 2e-4, "cosine")
        assert (config.train.epochs, config.train.early_stopping) == (30, 0)
    else:  # the legacy recipe
        assert recipe == ("mse", 1e-3, "constant") and config.train.optimizer.eps == 1e-7
        assert (config.train.epochs, config.train.early_stopping) == (
            (10, 3) if model == "fnet" else (50, 0)
        )
    overridden = load_experiment(CONFIGS / f"{name}.yaml", ["train.batch_size=64", "seed=7"])
    assert (overridden.train.batch_size, overridden.seed) == (64, 7)


def test_invalid_settings_are_rejected(root: Path, tmp_path: Path) -> None:
    config = experiment(root, tmp_path)
    bad = [
        (replace(config, train=replace(config.train, loss="huber")), "unknown loss"),
        (replace(config, train=replace(config.train, precision="fp16")), "unknown precision"),
        (replace(config, train=replace(config.train, grad_accum=0)), "grad_accum must be"),
        (replace(config, train=replace(config.train, ema_decay=1.0)), "ema_decay must be"),
        (replace(config, model=ModelConfig(name="srgan")), "unknown model"),
        (
            replace(config, train=replace(config.train, scheduler=SchedulerConfig(name="step"))),
            "unknown scheduler",
        ),
        (
            replace(
                config,
                train=replace(config.train, scheduler=SchedulerConfig("cosine", min_lr=1.0)),
            ),
            "min_lr must be",
        ),
    ]
    for value, message in bad:
        with pytest.raises(ValueError, match=message):
            validate(value)
    with pytest.raises(ValueError, match="no trainable parameters"):
        train(replace(config, model=ModelConfig(name="bicubic")))
    with pytest.raises(ValueError, match="smaller than one batch"):
        train(replace(config, train=replace(config.train, batch_size=TRAIN_SAMPLES + 1)))


def test_cosine_schedule() -> None:
    settings = TrainConfig(epochs=2, batch_size=3, grad_accum=2)
    settings.optimizer.lr = 1e-3
    assert steps_per_epoch(14, settings) == 2  # 4 full batches of 3, 2 steps of 2 batches
    assert learning_rate(settings, 10, 7) == 1e-3  # constant
    settings.scheduler = SchedulerConfig(name="cosine", min_lr=1e-5)
    rates = [learning_rate(settings, 10, i) for i in range(11)]
    assert rates[0] == pytest.approx(1e-3) and rates[10] == pytest.approx(1e-5)
    assert rates[5] == pytest.approx((1e-3 + 1e-5) / 2)
    assert all(a > b for a, b in zip(rates, rates[1:], strict=False))


def interrupt_after(monkeypatch: pytest.MonkeyPatch, calls: int) -> None:
    """Make the ``calls``-th validation raise KeyboardInterrupt (an interrupted run)."""
    original = evaluate
    count = 0

    def evaluate_or_stop(*args: Any, **kwargs: Any) -> dict[str, float]:
        nonlocal count
        count += 1
        if count == calls:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr("facehall.train.trainer.evaluate", evaluate_or_stop)


def test_resume_continues_as_if_uninterrupted(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def config(output: str) -> ExperimentConfig:
        base = experiment(root, tmp_path / output, epochs=3, ema_decay=0.5)
        base.train.scheduler = SchedulerConfig(name="cosine", min_lr=1e-5)
        return base

    reference = train(config("reference"))
    with monkeypatch.context() as patch:
        interrupt_after(patch, 3)
        with pytest.raises(KeyboardInterrupt):
            train(config("interrupted"))
    run = next((tmp_path / "interrupted").iterdir())
    assert json.loads((run / "summary.json").read_text())["status"] == "interrupted"
    assert resume(run) == run

    for name in ("last", "last_ema", "best"):
        expected = load_file(reference / f"{name}.safetensors")
        resumed = load_file(run / f"{name}.safetensors")
        assert all(torch.equal(expected[k], resumed[k]) for k in expected), name
    summary = json.loads((run / "summary.json").read_text())
    assert (summary["status"], summary["epochs"], summary["steps"]) == ("finished", 3, 12)
    assert [r["epoch"] for r in records(run, "epoch")] == [1, 2, 3]
    assert [r["lr"] for r in records(run, "epoch")] == [
        r["lr"] for r in records(reference, "epoch")
    ]
    assert [(r["epoch"], r["step"]) for r in records(run, "resume")] == [(2, 8)]
    info = json.loads((run / "run.json").read_text())
    assert [r["after_epoch"] for r in info["resumed"]] == [2]
    with pytest.raises(ValueError, match="already complete"):
        resume(run)


def test_resume_needs_a_completed_epoch(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        train(experiment(root, tmp_path))
    run = next(tmp_path.iterdir())
    with pytest.raises(FileNotFoundError, match="no completed epoch"):
        resume(run)
