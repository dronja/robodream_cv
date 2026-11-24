"""Supervised training of the SR models (docs/PROJECT.md §9, PR 1.6).

``train(config)`` runs one experiment and writes one run directory,
``<output>/<name>-<YYYYmmdd-HHMMSS>[-smoke]/``:

- ``config.yaml``: the resolved experiment config;
- ``run.json``: provenance: command, git commit, library versions, device, dataset settings and
  counts, and the manifest checksum from ``prepare_info.json``;
- ``metrics.jsonl``: one JSON record per logged optimizer step (``"event": "step"``) and per
  epoch (``"event": "epoch"``);
- ``train.log``: the text log;
- checkpoints (``facehall.train.checkpoint``): ``last`` after every epoch and ``best`` at the
  lowest validation loss. With EMA, validation and ``best`` use the EMA weights, and
  ``last_ema`` is written too;
- ``summary.json``: the status and the best epoch, written when the run ends.

Reproducibility: the seed fixes the model initialisation, the order of the training samples
(``EpochShuffle``) and the augmentation (``AlignedFaces``). On CPU two runs with one seed give
identical weights; on GPU, cuDNN autotuning and non-deterministic kernels allow small
differences.
"""

import json
import logging
import math
import platform
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from facehall.data.dataset import AlignedFaces, Sample
from facehall.models import build_model, count_parameters
from facehall.provenance import git_commit, package_versions
from facehall.train.checkpoint import save_checkpoint
from facehall.train.config import ExperimentConfig, experiment_to_dict, validate
from facehall.train.ema import ModelEMA

log = logging.getLogger(__name__)

LossFunction = Callable[..., torch.Tensor]  # (prediction, target, reduction=...) -> loss
LOSS_FUNCTIONS: dict[str, LossFunction] = {"mse": F.mse_loss, "l1": F.l1_loss}
VERSIONED = (
    "facehall",
    "torch",
    "torchvision",
    "numpy",
    "omegaconf",
    "opencv-python-headless",
    "pillow",
    "pyarrow",
    "safetensors",
)


class EpochShuffle(Sampler[int]):
    """A random order of ``range(size)`` that depends only on ``(seed, epoch)``.

    The generator is ``SeedSequence(seed, spawn_key=(epoch,))``, a stream independent of the
    per-sample generators of ``AlignedFaces``.
    """

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(self.epoch,)))
        return iter(rng.permutation(self.size).tolist())

    def __len__(self) -> int:
        return self.size


@dataclass
class EarlyStopping:
    """Stop after ``patience`` epochs without a lower value (Keras ``EarlyStopping`` with
    ``min_delta=0``). ``patience=0`` never stops."""

    patience: int
    best: float = math.inf
    stale: int = 0

    def update(self, value: float) -> bool:
        """Record one epoch; True if ``value`` is a new best."""
        if value < self.best:
            self.best, self.stale = value, 0
            return True
        self.stale += 1
        return False

    @property
    def stop(self) -> bool:
        return self.patience > 0 and self.stale >= self.patience


@dataclass
class EpochResult:
    loss: float  # mean training loss over the micro-batches
    steps: int  # optimizer steps
    images: int
    seconds: float


def autocast(device: torch.device, precision: str) -> torch.autocast:
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16")


def train_epoch(
    model: nn.Module,
    batches: Iterable[Sample],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFunction,
    *,
    device: torch.device,
    precision: str = "fp32",
    grad_accum: int = 1,
    ema: ModelEMA | None = None,
    step: int = 0,
    log_every: int = 0,
    on_log: Callable[..., None] | None = None,
) -> EpochResult:
    """One pass over ``batches``: an optimizer step every ``grad_accum`` micro-batches.

    Micro-batches left over at the end (fewer than ``grad_accum``) are discarded. ``on_log`` is
    called as ``on_log(step=…, loss=…)`` every ``log_every`` optimizer steps, counted from
    ``step``, with the mean loss since the previous call.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    window = torch.zeros((), device=device)
    micro = images = steps = window_size = 0
    start = time.perf_counter()
    for batch in batches:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        with autocast(device, precision):
            sr = model(lr)
        loss = loss_fn(sr.float(), hr)
        torch.autograd.backward(loss / grad_accum)
        total += loss.detach()
        window += loss.detach()
        micro, window_size, images = micro + 1, window_size + 1, images + lr.shape[0]
        if micro % grad_accum:
            continue
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        if ema is not None:
            ema.update(model)
        if on_log is not None and log_every and (step + steps) % log_every == 0:
            on_log(step=step + steps, loss=_finite(window.item() / window_size, step + steps))
            window.zero_()
            window_size = 0
    optimizer.zero_grad(set_to_none=True)
    if micro == 0:
        raise ValueError("no training batches: the train set is smaller than one batch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mean = _finite(total.item() / micro, step + steps)
    return EpochResult(mean, steps, images, time.perf_counter() - start)


@torch.no_grad()
def evaluate(
    model: nn.Module, batches: Iterable[Sample], loss_fn: LossFunction, *, device: torch.device
) -> dict[str, float]:
    """Mean per-image loss and PSNR (RGB, [0, 1] range) in fp32.

    The PSNR is a cheap monitoring value; the evaluation protocol (Y-channel PSNR, SSIM, LPIPS,
    ID similarity) is ``facehall eval``.
    """
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    psnr_sum = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    for batch in batches:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        sr = model(lr).float()
        loss_sum += loss_fn(sr, hr, reduction="none").flatten(1).mean(dim=1).sum()
        mse = (sr.clamp(0.0, 1.0) - hr).square().flatten(1).mean(dim=1)
        psnr_sum += (-10.0 * torch.log10(mse.clamp_min(1e-10))).sum()
        count += lr.shape[0]
    if count == 0:
        raise ValueError("no validation samples")
    return {"loss": loss_sum.item() / count, "psnr_rgb": psnr_sum.item() / count, "samples": count}


def train(config: ExperimentConfig, *, smoke: bool = False) -> Path:
    """Train the model of ``config``; returns the run directory."""
    validate(config)
    if smoke:
        config = replace(config, train=replace(config.train, epochs=config.smoke.epochs))
    settings = config.train
    device = resolve_device(config.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    if settings.workers > 0:
        # Before the workers fork: calling it inside a forked worker can deadlock.
        cv2.setNumThreads(1)

    train_set = AlignedFaces(config.data, config.degrade, "train", config.scale, seed=config.seed)
    val_set = AlignedFaces(config.data, config.degrade, "val", config.scale, seed=config.seed)
    train_data: Dataset[Sample] = train_set
    val_data: Dataset[Sample] = val_set
    if smoke:
        train_data = random_subset(train_set, config.smoke.train_samples, config.seed)
        val_data = random_subset(val_set, config.smoke.val_samples, config.seed)
    train_size, val_size = len(train_data), len(val_data)  # type: ignore[arg-type]

    torch.manual_seed(config.seed)
    model = build_model(config.model, config.scale, config.data.hr_size // config.scale)
    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError(f"model {config.model.name!r} has no trainable parameters")
    opt = settings.optimizer
    optimizer = torch.optim.Adam(
        parameters,
        lr=opt.lr,
        betas=(opt.beta1, opt.beta2),
        eps=opt.eps,
        weight_decay=opt.weight_decay,
    )
    ema = ModelEMA(model, settings.ema_decay) if settings.ema_decay > 0 else None
    eval_model = ema.module if ema is not None else model
    loss_fn = LOSS_FUNCTIONS[settings.loss]

    run_dir = create_run_dir(Path(config.output), config.name, smoke)
    commit = git_commit()
    device_info = describe_device(device)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(experiment_to_dict(config)))
    run_info = {
        "name": config.name,
        "run": run_dir.name,
        "created_at": _now(),
        "command": sys.argv,
        "smoke": smoke,
        "seed": config.seed,
        "git_commit": commit,
        "python": platform.python_version(),
        "versions": package_versions(VERSIONED),
        "device": device_info,
        "model": {"name": config.model.name, "parameters": count_parameters(model)},
        "data": {
            "train": {**train_set.describe(), "used": train_size},
            "val": {**val_set.describe(), "used": val_size},
        },
        "manifest": manifest_info(Path(config.data.root)),
    }
    write_json(run_dir / "run.json", run_info)

    package_log = logging.getLogger("facehall")
    level = package_log.level
    package_log.setLevel(logging.INFO)
    handler = logging.FileHandler(run_dir / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    package_log.addHandler(handler)
    metrics = (run_dir / "metrics.jsonl").open("a")

    sampler = EpochShuffle(train_size, config.seed)
    val_loader = make_loader(val_data, settings.val_batch_size, settings.workers, device)
    stopper = EarlyStopping(settings.early_stopping)
    summary: dict[str, Any] = {"status": "running"}
    best: dict[str, Any] | None = None
    epoch = step = 0
    start = time.perf_counter()
    try:
        log.info(
            "run %s: %s x%d, %s parameters, %s train / %s val samples, %s, %s",
            run_dir,
            config.model.name,
            config.scale,
            f"{count_parameters(model):,}",
            f"{train_size:,}",
            f"{val_size:,}",
            device_info["name"],
            settings.precision,
        )
        status = "finished"
        for epoch in range(1, settings.epochs + 1):
            train_set.set_epoch(epoch)
            sampler.set_epoch(epoch)
            loader = make_loader(
                train_data, settings.batch_size, settings.workers, device, sampler, drop_last=True
            )
            result = train_epoch(
                model,
                loader,
                optimizer,
                loss_fn,
                device=device,
                precision=settings.precision,
                grad_accum=settings.grad_accum,
                ema=ema,
                step=step,
                log_every=settings.log_every,
                on_log=partial(write_record, metrics, "step", epoch=epoch),
            )
            step += result.steps
            val_start = time.perf_counter()
            val = evaluate(eval_model, val_loader, loss_fn, device=device)
            val_seconds = time.perf_counter() - val_start
            if not math.isfinite(val["loss"]):
                raise FloatingPointError(f"validation loss is {val['loss']} after epoch {epoch}")
            improved = stopper.update(val["loss"])
            record = {
                "epoch": epoch,
                "step": step,
                "train_loss": result.loss,
                "val_loss": val["loss"],
                "val_psnr_rgb": val["psnr_rgb"],
                "best": improved,
                "train_seconds": result.seconds,
                "train_images_per_s": result.images / result.seconds,
                "val_seconds": val_seconds,
            }
            write_record(metrics, "epoch", **record)
            meta = {
                "name": config.name,
                "run": run_dir.name,
                "smoke": smoke,
                "epoch": epoch,
                "step": step,
                "metrics": {"val_loss": val["loss"], "val_psnr_rgb": val["psnr_rgb"]},
                "git_commit": commit,
                "created_at": _now(),
                "config": experiment_to_dict(config),
            }
            save_checkpoint(run_dir / "last.safetensors", model, {**meta, "weights": "model"})
            if ema is not None:
                save_checkpoint(
                    run_dir / "last_ema.safetensors", ema.module, {**meta, "weights": "ema"}
                )
            if improved:
                weights = "ema" if ema is not None else "model"
                save_checkpoint(
                    run_dir / "best.safetensors", eval_model, {**meta, "weights": weights}
                )
                best = record
            log.info(
                "epoch %d/%d: train loss %.6f, val loss %.6f, val PSNR (RGB) %.2f dB%s; "
                "%.0f s, %s img/s",
                epoch,
                settings.epochs,
                result.loss,
                val["loss"],
                val["psnr_rgb"],
                " (best)" if improved else "",
                result.seconds,
                f"{result.images / result.seconds:,.0f}",
            )
            if stopper.stop:
                status = "early_stopped"
                log.info("early stopping: no lower val loss for %d epochs", stopper.stale)
                break
        summary["status"] = status
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = repr(error)
        raise
    finally:
        summary.update(epochs=epoch, steps=step, seconds=time.perf_counter() - start, best=best)
        write_json(run_dir / "summary.json", summary)
        metrics.close()
        package_log.removeHandler(handler)
        handler.close()
        package_log.setLevel(level)
    log.info("run %s %s in %.0f s", run_dir, summary["status"], summary["seconds"])
    return run_dir


def make_loader(
    data: Dataset[Sample],
    batch_size: int,
    workers: int,
    device: torch.device,
    sampler: Sampler[int] | None = None,
    *,
    drop_last: bool = False,
) -> DataLoader[Sample]:
    """Workers are not persistent, so every epoch's workers see the dataset's current epoch."""
    return DataLoader(
        data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )


def random_subset(dataset: AlignedFaces, size: int, seed: int) -> Subset[Sample]:
    """``size`` samples drawn without replacement (all of them if the dataset is smaller)."""
    count = min(size, len(dataset))
    indices = np.sort(np.random.default_rng(seed).choice(len(dataset), count, replace=False))
    return Subset(dataset, indices.tolist())


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {name!r} requested, but CUDA is not available")
    return device


def describe_device(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return {
            "type": "cuda",
            "name": torch.cuda.get_device_name(index),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        }
    return {
        "type": device.type,
        "name": platform.processor() or platform.machine(),
        "threads": torch.get_num_threads(),
    }


def manifest_info(root: Path) -> dict[str, Any] | None:
    """The ``manifest`` section of ``prepare_info.json`` (row count, checksums), if present."""
    path = root / "prepare_info.json"
    if not path.is_file():
        return None
    manifest: dict[str, Any] | None = json.loads(path.read_text()).get("manifest")
    return manifest


def create_run_dir(output: Path, name: str, smoke: bool) -> Path:
    base = f"{name}-{datetime.now():%Y%m%d-%H%M%S}" + ("-smoke" if smoke else "")
    for attempt in range(100):
        candidate = output / (base if attempt == 0 else f"{base}-{attempt}")
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not create a run directory for {base} in {output}")


def write_record(file: TextIO, event: str, **values: Any) -> None:
    file.write(json.dumps({"event": event, "time": _now(), **values}) + "\n")
    file.flush()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _finite(value: float, step: int) -> float:
    if not math.isfinite(value):
        raise FloatingPointError(f"training loss is {value} at step {step}")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
