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
  lowest validation loss (or after every epoch with ``train.best_metric: none``). With EMA,
  validation and ``best`` use the EMA weights, and ``last_ema`` is written too;
- ``last_state.pt``: the training state after the last completed epoch, for ``resume``;
- ``summary.json``: the status and the best epoch, written when the run ends.

The learning rate follows ``train.scheduler`` per optimizer step. ``resume(run_dir)`` continues
an interrupted run (``facehall train --resume``).

The generator loss is ``facehall.train.objective.Objective``: the pixel loss ``train.loss``,
optionally plus a perceptual (VGG19) and an adversarial term with a discriminator, which is
updated after every generator step (the ESRGAN recipe, PR 3.2), and an identity term on facenet
embeddings (PR 3.3). ``train.init`` starts the model from the weights of another run's
checkpoint, e.g. the L1-trained generator. Validation always reports the pixel loss; with an
identity term it also reports the mean facenet cosine similarity of SR and HR.

Reproducibility: the seed fixes the model initialisation, the order of the training samples
(``EpochShuffle``) and the augmentation (``AlignedFaces``). On CPU two runs with one seed give
identical weights; on GPU, cuDNN autotuning and non-deterministic kernels allow small
differences.
"""

import json
import logging
import math
import os
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
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from facehall.data.dataset import AlignedFaces, Sample
from facehall.losses.facenet import load_facenet
from facehall.losses.identity import IdentityLoss
from facehall.losses.perceptual import PerceptualLoss, vgg19_features
from facehall.models import build_model, count_parameters
from facehall.models.discriminators import VGGDiscriminator
from facehall.provenance import file_sha256, git_commit, package_versions
from facehall.train.checkpoint import read_metadata, save_checkpoint
from facehall.train.config import (
    ExperimentConfig,
    OptimizerConfig,
    TrainConfig,
    experiment_from_dict,
    experiment_to_dict,
    load_experiment,
    validate,
)
from facehall.train.ema import ModelEMA
from facehall.train.objective import Objective

log = logging.getLogger(__name__)

LossFunction = Callable[..., torch.Tensor]  # (prediction, target, reduction=...) -> loss
LOSS_FUNCTIONS: dict[str, LossFunction] = {"mse": F.mse_loss, "l1": F.l1_loss}
COMPLETE = ("finished", "early_stopped")  # summary statuses that cannot be resumed
STATE_FILE = "last_state.pt"
STATE_FORMAT = "facehall-train-state"
STATE_VERSION = 1
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
    schedule: Callable[[int], float] | None = None,
    after_step: Callable[[float], None] | None = None,
    log_every: int = 0,
    on_log: Callable[..., None] | None = None,
) -> EpochResult:
    """One pass over ``batches``: an optimizer step every ``grad_accum`` micro-batches.

    ``schedule(i)`` sets the learning rate of optimizer step ``i`` (0-based, counted over the
    run from ``step``). ``after_step(lr)`` is called after every optimizer step with its
    learning rate (the discriminator update of an adversarial ``loss_fn``).

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
        if schedule is not None:
            rate = schedule(step + steps)
            for group in optimizer.param_groups:
                group["lr"] = rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        if after_step is not None:
            after_step(optimizer.param_groups[0]["lr"])
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
    model: nn.Module,
    batches: Iterable[Sample],
    loss_fn: LossFunction,
    *,
    device: torch.device,
    identity: IdentityLoss | None = None,
) -> dict[str, float]:
    """Mean per-image loss and PSNR (RGB, [0, 1] range) in fp32; with ``identity``, also the
    mean cosine similarity of its embeddings of SR and HR.

    These are cheap monitoring values; the evaluation protocol (Y-channel PSNR, SSIM, LPIPS,
    ID similarity with another recognition model) is ``facehall eval``.
    """
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    psnr_sum = torch.zeros((), dtype=torch.float64, device=device)
    identity_sum = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    for batch in batches:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        sr = model(lr).float()
        loss_sum += loss_fn(sr, hr, reduction="none").flatten(1).mean(dim=1).sum()
        mse = (sr.clamp(0.0, 1.0) - hr).square().flatten(1).mean(dim=1)
        psnr_sum += (-10.0 * torch.log10(mse.clamp_min(1e-10))).sum()
        if identity is not None:
            identity_sum += identity.similarity(sr.clamp(0.0, 1.0), hr).sum()
        count += lr.shape[0]
    if count == 0:
        raise ValueError("no validation samples")
    result = {"loss": loss_sum.item() / count, "psnr_rgb": psnr_sum.item() / count}
    if identity is not None:
        result["identity"] = identity_sum.item() / count
    return {**result, "samples": count}


def train(config: ExperimentConfig, *, smoke: bool = False) -> Path:
    """Train the model of ``config``; returns the run directory."""
    validate(config)
    if smoke:
        config = replace(config, train=replace(config.train, epochs=config.smoke.epochs))
    run_dir = create_run_dir(Path(config.output), config.name, smoke)
    return _run(config, run_dir, smoke=smoke, state=None)


def resume(run_dir: Path | str) -> Path:
    """Continue an interrupted run in ``run_dir`` after its last completed epoch.

    The config, the smoke flag and the training state (``last_state.pt``: weights, EMA weights,
    optimizer state, epoch, step, early-stopping state) come from the run directory, so the run
    continues as if it had not stopped: the sample order and augmentation depend only on the
    seed and the epoch, and the learning rate only on the step. ``run.json`` records the resume,
    and ``metrics.jsonl`` gets a ``resume`` record; step records of the interrupted epoch that
    precede it are superseded.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        status = json.loads(summary_path.read_text()).get("status")
        if status in COMPLETE:
            raise ValueError(f"{run_dir} is already complete ({status})")
    info = json.loads((run_dir / "run.json").read_text())
    config = load_experiment(run_dir / "config.yaml")
    return _run(config, run_dir, smoke=bool(info["smoke"]), state=load_state(run_dir))


def _run(
    config: ExperimentConfig, run_dir: Path, *, smoke: bool, state: dict[str, Any] | None
) -> Path:
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
    init = None
    if settings.init:
        init = init_info(Path(settings.init), config)
        if state is None:
            model.load_state_dict(load_file(settings.init))
    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError(f"model {config.model.name!r} has no trainable parameters")
    optimizer = make_optimizer(parameters, settings.optimizer)
    ema = ModelEMA(model, settings.ema_decay) if settings.ema_decay > 0 else None
    eval_model = ema.module if ema is not None else model
    loss_fn = LOSS_FUNCTIONS[settings.loss]
    objective, extras = make_objective(config, device)
    total_steps = settings.epochs * steps_per_epoch(train_size, settings)
    schedule = partial(learning_rate, settings, total_steps)
    stopper = EarlyStopping(settings.early_stopping)
    best: dict[str, Any] | None = None
    epoch = step = 0
    elapsed = 0.0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if ema is not None:
            ema.module.load_state_dict(state["ema"])
        epoch, step, elapsed, best = state["epoch"], state["step"], state["seconds"], state["best"]
        stopper.best, stopper.stale = state["stopper"]["best"], state["stopper"]["stale"]
        if objective.discriminator is not None and objective.d_optimizer is not None:
            objective.discriminator.load_state_dict(state["discriminator"])
            objective.d_optimizer.load_state_dict(state["d_optimizer"])
        if objective.identity is not None and state.get("identity_rng") is not None:
            objective.identity.generator.set_state(state["identity_rng"])

    commit = git_commit()
    device_info = describe_device(device)
    if state is None:
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
            "init": init,
            **extras,
            "data": {
                "train": {**train_set.describe(), "used": train_size},
                "val": {**val_set.describe(), "used": val_size},
            },
            "manifest": manifest_info(Path(config.data.root)),
        }
    else:
        run_info = json.loads((run_dir / "run.json").read_text())
        run_info.setdefault("resumed", []).append(
            {
                "created_at": _now(),
                "command": sys.argv,
                "after_epoch": epoch,
                "git_commit": commit,
                "versions": package_versions(VERSIONED),
                "device": device_info,
            }
        )
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
    summary: dict[str, Any] = {"status": "running"}
    start = time.perf_counter()
    try:
        if state is not None:
            write_record(metrics, "resume", epoch=epoch, step=step, git_commit=commit)
            if commit != run_info.get("git_commit"):
                log.warning(
                    "resuming at commit %s; the run started at %s",
                    commit,
                    run_info.get("git_commit"),
                )
        log.info(
            "run %s: %s x%d, %s parameters, %s train / %s val samples, %s, %s%s",
            run_dir,
            config.model.name,
            config.scale,
            f"{count_parameters(model):,}",
            f"{train_size:,}",
            f"{val_size:,}",
            device_info["name"],
            settings.precision,
            f", resumed after epoch {epoch}" if state is not None else "",
        )
        status = "finished"
        first_epoch = epoch + 1
        for epoch in range(first_epoch, settings.epochs + 1):
            train_set.set_epoch(epoch)
            sampler.set_epoch(epoch)
            loader = make_loader(
                train_data, settings.batch_size, settings.workers, device, sampler, drop_last=True
            )
            result = train_epoch(
                model,
                loader,
                optimizer,
                objective,
                device=device,
                precision=settings.precision,
                grad_accum=settings.grad_accum,
                ema=ema,
                step=step,
                schedule=schedule,
                after_step=objective.after_step,
                log_every=settings.log_every,
                on_log=partial(log_step, metrics, objective, epoch=epoch),
            )
            terms = objective.take_epoch()
            step += result.steps
            val_start = time.perf_counter()
            val = evaluate(
                eval_model, val_loader, loss_fn, device=device, identity=objective.identity
            )
            val_metrics = {f"val_{name}": value for name, value in val.items() if name != "samples"}
            val_seconds = time.perf_counter() - val_start
            if not math.isfinite(val["loss"]):
                raise FloatingPointError(f"validation loss is {val['loss']} after epoch {epoch}")
            improved = stopper.update(val["loss"]) or settings.best_metric == "none"
            record = {
                "epoch": epoch,
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": result.loss,
                "train_terms": terms,
                **val_metrics,
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
                "metrics": val_metrics,
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
            save_state(
                run_dir,
                epoch=epoch,
                step=step,
                seconds=elapsed + time.perf_counter() - start,
                model=model,
                ema=ema,
                optimizer=optimizer,
                stopper=stopper,
                best=best,
                objective=objective,
            )
            log.info(
                "epoch %d/%d: train loss %.6f (%s), val loss %.6f, val PSNR (RGB) %.2f dB%s%s; "
                "%.0f s, %s img/s",
                epoch,
                settings.epochs,
                result.loss,
                ", ".join(f"{name} {value:.4g}" for name, value in terms.items()),
                val["loss"],
                val["psnr_rgb"],
                f", val identity {val['identity']:.4f}" if "identity" in val else "",
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
        summary.update(
            epochs=epoch, steps=step, seconds=elapsed + time.perf_counter() - start, best=best
        )
        write_json(run_dir / "summary.json", summary)
        metrics.close()
        package_log.removeHandler(handler)
        handler.close()
        package_log.setLevel(level)
    log.info("run %s %s in %.0f s", run_dir, summary["status"], summary["seconds"])
    return run_dir


def make_optimizer(parameters: Iterable[nn.Parameter], opt: OptimizerConfig) -> torch.optim.Adam:
    return torch.optim.Adam(
        parameters,
        lr=opt.lr,
        betas=(opt.beta1, opt.beta2),
        eps=opt.eps,
        weight_decay=opt.weight_decay,
    )


def make_objective(
    config: ExperimentConfig, device: torch.device
) -> tuple[Objective, dict[str, Any]]:
    """The generator objective of ``config.train``, and what ``run.json`` records about it."""
    settings = config.train
    weights = settings.losses
    extras: dict[str, Any] = {"losses": {"loss": settings.loss, **vars(weights)}}
    perceptual = None
    if weights.perceptual > 0:
        path = Path(settings.perceptual.weights)
        features = vgg19_features(path, settings.perceptual.layer)
        perceptual = PerceptualLoss(features).to(device)
        extras["perceptual"] = {
            "layer": settings.perceptual.layer,
            "weights": str(path),
            "sha256": file_sha256(path),
        }
    identity = None
    if weights.identity > 0:
        path = Path(settings.identity.weights)
        jitter = settings.identity.jitter
        identity = IdentityLoss(
            load_facenet(path),
            config.data.hr_size,
            side=settings.identity.crop_side,
            center_y=settings.identity.crop_center_y,
            shift=jitter.shift,
            rotation=jitter.rotation,
            scale=jitter.scale,
            blur=settings.identity.blur,
            seed=config.seed,
        ).to(device)
        top, left, side = identity.box
        extras["identity"] = {
            "network": "facenet InceptionResnetV1",
            "weights": str(path),
            "sha256": file_sha256(path),
            "crop": {"top": top, "left": left, "side": side},
            "jitter": vars(jitter),
            "blur": settings.identity.blur,
        }
    discriminator = d_optimizer = None
    if weights.adversarial > 0:
        d_config = settings.discriminator
        discriminator = VGGDiscriminator(config.data.hr_size, d_config.channels).to(device)
        d_optimizer = make_optimizer(discriminator.parameters(), d_config.optimizer)
        extras["discriminator"] = {
            "name": d_config.name,
            "parameters": count_parameters(discriminator),
        }
    objective = Objective(
        LOSS_FUNCTIONS[settings.loss],
        weights,
        perceptual=perceptual,
        identity=identity,
        discriminator=discriminator,
        d_optimizer=d_optimizer,
        g_lr=settings.optimizer.lr,
        device=device,
        precision=settings.precision,
    )
    return objective, extras


def init_info(path: Path, config: ExperimentConfig) -> dict[str, Any]:
    """Check that checkpoint ``path`` holds the model of ``config``; returns its provenance."""
    info = read_metadata(path)
    source = experiment_from_dict(info["config"])
    if (source.model, source.scale, source.data.hr_size) != (
        config.model,
        config.scale,
        config.data.hr_size,
    ):
        raise ValueError(f"{path} holds another model, scale or HR size than the config")
    return {
        "path": str(path),
        "sha256": info["sha256"],
        "run": info.get("run"),
        "epoch": info.get("epoch"),
        "weights": info.get("weights"),
        "git_commit": info.get("git_commit"),
    }


def log_step(file: TextIO, objective: Objective, *, epoch: int, step: int, loss: float) -> None:
    write_record(file, "step", epoch=epoch, step=step, loss=loss, terms=objective.take_window())


def steps_per_epoch(train_size: int, settings: TrainConfig) -> int:
    """Optimizer steps per epoch: full batches only, leftover micro-batches discarded."""
    return (train_size // settings.batch_size) // settings.grad_accum


def learning_rate(settings: TrainConfig, total_steps: int, step: int) -> float:
    """Learning rate of optimizer step ``step`` (0-based) of ``total_steps``."""
    base = settings.optimizer.lr
    if settings.scheduler.name == "constant":
        return base
    low = settings.scheduler.min_lr
    progress = min(step / max(total_steps, 1), 1.0)
    return low + 0.5 * (base - low) * (1.0 + math.cos(math.pi * progress))


def save_state(
    run_dir: Path,
    *,
    epoch: int,
    step: int,
    seconds: float,
    model: nn.Module,
    ema: ModelEMA | None,
    optimizer: torch.optim.Optimizer,
    stopper: EarlyStopping,
    best: dict[str, Any] | None,
    objective: Objective | None = None,
) -> None:
    """Write ``last_state.pt``, everything ``resume`` needs, under a temporary name first."""
    discriminator = objective.discriminator if objective is not None else None
    d_optimizer = objective.d_optimizer if objective is not None else None
    identity = objective.identity if objective is not None else None
    state = {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "epoch": epoch,
        "step": step,
        "seconds": seconds,
        "model": model.state_dict(),
        "ema": ema.module.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "stopper": {"best": stopper.best, "stale": stopper.stale},
        "best": best,
        "discriminator": discriminator.state_dict() if discriminator is not None else None,
        "d_optimizer": d_optimizer.state_dict() if d_optimizer is not None else None,
        "identity_rng": identity.generator.get_state() if identity is not None else None,
    }
    path = run_dir / STATE_FILE
    partial_path = path.with_name(path.name + ".partial")
    torch.save(state, partial_path)
    os.replace(partial_path, path)


def load_state(run_dir: Path) -> dict[str, Any]:
    """Read ``last_state.pt`` (tensors and plain values only: ``weights_only=True``)."""
    path = run_dir / STATE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the run has no completed epoch to resume")
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != STATE_FORMAT or state.get("version") != STATE_VERSION:
        raise ValueError(f"{path} is not a {STATE_FORMAT} version {STATE_VERSION}")
    return state


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
