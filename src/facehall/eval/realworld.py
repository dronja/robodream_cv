"""Score one model on a real-world protocol without HR reference (``kind: realworld``).

Output directory ``<output>/<protocol>/<split>/<model label>/``, as for the synthetic protocols:

- ``metrics.json``: per subset the count, LR-PSNR-Y (mean, std) and KID (mean, std over random
  subsets); FID for all faces; the metric setup and provenance;
- ``per_image.parquet``: LR-PSNR-Y per face with the manifest attributes, so other breakdowns
  need no re-run;
- ``samples/``: SR crops of the grid samples; their LR inputs go to ``../reference/lr/``. These
  are faces of real people under CC BY-NC-ND and stay local (D20).

FID and KID compare the Inception features of the SR outputs with those of real HR faces: the
HR crops of a CelebA split (``reference``, validation by default). These do not depend on the
model, so they are cached once per fingerprint in ``<output>/_cache/inception_ref-<fp>.npz``.

Frozen inputs: the sha256 of every LR crop in dataset order (``inputs_sha256``).
"""

import hashlib
import json
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import MISSING
from torch import nn

from facehall.config import require_choice
from facehall.data.config import DataConfig, DegradeConfig
from facehall.data.dataset import AlignedFaces
from facehall.data.manifest import SPLITS
from facehall.data.realworld import RealFaces
from facehall.eval.noref import FeatureExtractor, InceptionFeatures, fid, kid, lr_psnr_y, to_uint8
from facehall.eval.runner import (
    CACHE,
    CPU,
    InputsChangedError,
    grid_indices,
    quantize,
    to_bytes,
    to_image,
)
from facehall.provenance import file_sha256, git_commit, package_versions
from facehall.train.trainer import (
    VERSIONED,
    describe_device,
    make_loader,
    manifest_info,
    resolve_device,
    write_json,
)

log = logging.getLogger(__name__)

KIND = "realworld"

#: Subsets of the WIDER FACE manifest (docs/DATA.md §7): name -> (column, accepted values).
SUBSETS: dict[str, tuple[str, tuple[int, ...]]] = {
    "clear": ("attr_blur", (0,)),
    "normal_blur": ("attr_blur", (1,)),
    "heavy_blur": ("attr_blur", (2,)),
    "unoccluded": ("attr_occlusion", (0,)),
    "occluded": ("attr_occlusion", (1, 2)),
    "atypical_pose": ("attr_pose", (1,)),
}
ATTRIBUTE_COLUMNS = (
    "attr_blur",
    "attr_expression",
    "attr_illumination",
    "attr_occlusion",
    "attr_pose",
    "padded",
    "align_residual",
)


@dataclass
class NoRefMetricsConfig:
    # InceptionV3 weights (scripts/download_weights.py inception); "" disables FID and KID.
    inception: str = ""
    kid_subsets: int = 100
    kid_subset_size: int = 1000
    kid_min_faces: int = 200  # smaller subsets get no KID


@dataclass
class RealWorldConfig:
    """A frozen real-world protocol: LR crops without HR, scored without reference."""

    kind: str = KIND
    name: str = MISSING
    level: str = "real"  # report column
    scale: int = MISSING
    split: str = "test"
    root: str = MISSING  # prepared real-world set (scripts/prepare_wider.py)
    hr_size: int = 128
    seed: int = 0  # KID subsets and the grid samples
    device: str = "auto"
    batch_size: int = 256
    workers: int = 4
    output: str = "outputs/eval"
    grid_samples: int = 16  # local only (D20); 0 disables
    inputs_sha256: dict[str, str] = field(default_factory=dict)  # split -> sha256 of the LR crops
    # Real HR faces for FID and KID: the HR crops of this CelebA split.
    reference: DataConfig = field(default_factory=DataConfig)
    reference_split: str = "val"
    metrics: NoRefMetricsConfig = field(default_factory=NoRefMetricsConfig)


def validate_realworld(config: RealWorldConfig) -> None:
    require_choice(config.kind, (KIND,), "protocol kind")
    require_choice(config.split, SPLITS, "split")
    require_choice(config.reference_split, SPLITS, "reference split")
    for split, digest in config.inputs_sha256.items():
        require_choice(split, SPLITS, "split in inputs_sha256")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"inputs_sha256.{split} must be a lowercase sha256, got {digest!r}")
    if config.scale < 1 or config.hr_size % config.scale:
        raise ValueError(f"scale {config.scale} does not divide hr_size {config.hr_size}")
    if config.reference.hr_size != config.hr_size:
        raise ValueError(
            f"reference HR size {config.reference.hr_size} differs from hr_size {config.hr_size}"
        )
    for name, value, lowest in (
        ("batch_size", config.batch_size, 1),
        ("workers", config.workers, 0),
        ("grid_samples", config.grid_samples, 0),
        ("metrics.kid_subsets", config.metrics.kid_subsets, 1),
        ("metrics.kid_subset_size", config.metrics.kid_subset_size, 2),
        ("metrics.kid_min_faces", config.metrics.kid_min_faces, 2),
    ):
        if value < lowest:
            raise ValueError(f"{name} must be at least {lowest}, got {value}")


def protocol_dataset(config: RealWorldConfig) -> RealFaces:
    dataset = RealFaces(config.root, config.split, config.scale)
    if dataset.lr_size != config.hr_size // config.scale:
        raise ValueError(
            f"{config.root}: x{config.scale} crops are {dataset.lr_size} px, the protocol "
            f"expects {config.hr_size // config.scale} px"
        )
    return dataset


def realworld_digest(config: RealWorldConfig) -> str:
    """sha256 of every LR crop of the protocol split, in dataset order, at 8-bit levels."""
    validate_realworld(config)
    digest = hashlib.sha256()
    for batch in make_loader(protocol_dataset(config), config.batch_size, config.workers, CPU):
        digest.update(to_bytes(batch["lr"]))
    return digest.hexdigest()


def check_inputs(config: RealWorldConfig, digest: str) -> None:
    expected = config.inputs_sha256.get(config.split)
    if expected and expected != digest:
        raise InputsChangedError(
            f"the {config.split} inputs of protocol {config.name} changed: sha256 {digest}, "
            f"frozen {expected}. Re-run scripts/prepare_wider.py with the pinned archives, or "
            "refreeze on purpose and re-evaluate every model"
        )


def build_extractor(config: RealWorldConfig, device: torch.device) -> FeatureExtractor | None:
    if not config.metrics.inception:
        return None
    return InceptionFeatures(config.metrics.inception, device)


def evaluate_realworld(
    model: nn.Module,
    config: RealWorldConfig,
    label: str,
    model_info: dict[str, Any],
    *,
    extractor: FeatureExtractor | None = None,
) -> Path:
    """Score ``model`` on the real-world protocol ``config``; returns the output directory.

    ``extractor`` defaults to the InceptionV3 features when ``metrics.inception`` is set.
    """
    validate_realworld(config)
    device = resolve_device(config.device)
    if config.workers > 0:
        cv2.setNumThreads(1)  # before the DataLoader workers fork (docs/PROJECT.md §9)
    dataset = protocol_dataset(config)
    if extractor is None:
        extractor = build_extractor(config, device)
    reference = reference_features(config, extractor, device) if extractor is not None else None
    model = model.to(device).eval()
    out = Path(config.output) / config.name / config.split / label
    (out / "samples").mkdir(parents=True, exist_ok=True)
    grid = set(grid_indices(len(dataset), config.grid_samples, config.seed))

    scores: list[npt.NDArray[np.float64]] = []
    features: list[npt.NDArray[np.float32]] = []
    indices: list[npt.NDArray[np.int64]] = []
    inputs = hashlib.sha256()
    start = time.perf_counter()
    log.info("evaluating %s on %s/%s: %s faces", label, config.name, config.split, len(dataset))
    with torch.no_grad():
        for batch in make_loader(dataset, config.batch_size, config.workers, device):
            inputs.update(to_bytes(batch["lr"]))
            lr = batch["lr"].to(device, non_blocking=True)
            sr = quantize(model(lr).float())
            if sr.shape[-1] != config.hr_size:
                raise ValueError(f"{label} outputs {sr.shape[-1]} px, expected {config.hr_size}")
            scores.append(lr_psnr_y(sr, lr).double().cpu().numpy())
            if extractor is not None:
                features.append(extractor(to_uint8(sr)).cpu().numpy().astype(np.float32))
            batch_indices = batch["index"].numpy()
            indices.append(batch_indices)
            for row, index in enumerate(batch_indices.tolist()):
                if index in grid:
                    save_lr_sample(index, lr[row], sr[row], out / "samples", out.parent)
    digest = inputs.hexdigest()
    check_inputs(config, digest)

    order = np.concatenate(indices)
    columns: dict[str, npt.NDArray[Any]] = {"index": order, "lr_psnr_y": np.concatenate(scores)}
    for name in ATTRIBUTE_COLUMNS:
        if dataset.has_column(name):
            columns[name] = dataset.column(name)[order]
    sr_features = np.concatenate(features)[np.argsort(order)] if features else None
    masks = subset_masks(dataset)
    subsets = summarize(columns, masks, order, sr_features, reference, config, device)
    seconds = time.perf_counter() - start

    files = [dataset.path(i).name for i in order.tolist()]
    extra = {
        name: dataset.column(name)[order]
        for name in ("source_image", "face_index")
        if dataset.has_column(name)
    }
    pq.write_table(pa.table({**columns, **extra, "path": files}), out / "per_image.parquet")
    summary = {
        "kind": KIND,
        "protocol": config.name,
        "level": config.level,
        "split": config.split,
        "scale": config.scale,
        "label": label,
        "model": model_info,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": seconds,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "versions": package_versions((*VERSIONED, "torchmetrics", "torch-fidelity")),
        "device": describe_device(device),
        "dataset": dataset.describe(),
        "inputs_sha256": digest,
        "manifest": manifest_info(Path(config.root)),
        "reference": reference_info(config, len(reference)) if reference is not None else None,
        "metric_setup": metric_setup(config, extractor),
        "subsets": subsets,
    }
    write_json(out / "metrics.json", summary)
    log.info("%s: %s in %.0f s", label, format_all(subsets["all"]), seconds)
    return out


def save_lr_sample(
    index: int, lr: torch.Tensor, sr: torch.Tensor, samples: Path, protocol_dir: Path
) -> None:
    """Like ``facehall.eval.runner.save_samples``, without an HR image."""
    samples.mkdir(parents=True, exist_ok=True)
    (protocol_dir / "reference" / "lr").mkdir(parents=True, exist_ok=True)
    to_image(sr).save(samples / f"{index:06d}.png")
    to_image(lr).save(protocol_dir / "reference" / "lr" / f"{index:06d}.png")


def subset_masks(dataset: RealFaces) -> dict[str, npt.NDArray[np.bool_]]:
    """Boolean masks in dataset order: ``all``, ``unpadded`` and the known ``SUBSETS``."""
    size = len(dataset)
    masks: dict[str, npt.NDArray[np.bool_]] = {"all": np.ones(size, dtype=bool)}
    for name, (column, values) in SUBSETS.items():
        if dataset.has_column(column):
            masks[name] = np.isin(dataset.column(column), values)
    if dataset.has_column("padded"):
        masks["unpadded"] = dataset.column("padded") == 0.0
    return masks


def summarize(
    columns: dict[str, npt.NDArray[Any]],
    masks: dict[str, npt.NDArray[np.bool_]],
    order: npt.NDArray[np.int64],
    sr_features: npt.NDArray[np.float32] | None,
    reference: npt.NDArray[np.float32] | None,
    config: RealWorldConfig,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Per subset: count, LR-PSNR-Y mean and std, KID (if large enough); FID for ``all``.

    ``columns`` are in loader order (``order``); masks and features in dataset order.
    """
    scores = np.empty(len(order))
    scores[order] = columns["lr_psnr_y"]
    result: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        stats: dict[str, Any] = {
            "count": count,
            "lr_psnr_y": (
                {"mean": float(scores[mask].mean()), "std": float(scores[mask].std())}
                if count
                else None
            ),
            "kid": None,
            "fid": None,
        }
        if sr_features is not None and reference is not None:
            if count >= config.metrics.kid_min_faces:
                mean, std = kid(
                    reference,
                    sr_features[mask],
                    subsets=config.metrics.kid_subsets,
                    subset_size=config.metrics.kid_subset_size,
                    seed=config.seed,
                    device=device,
                )
                stats["kid"] = {"mean": mean, "std": std}
            if name == "all":
                stats["fid"] = fid(reference, sr_features)
        result[name] = stats
    return result


def reference_dataset(config: RealWorldConfig) -> AlignedFaces:
    """CelebA HR crops; the bicubic degradation only fills the unused LR."""
    return AlignedFaces(
        config.reference,
        DegradeConfig(),
        config.reference_split,
        config.scale,
        seed=0,
        augment=False,
    )


def reference_fingerprint(config: RealWorldConfig, extractor: FeatureExtractor) -> str:
    """sha256 of what the reference features depend on: data, split, manifest, extractor."""
    dataset = reference_dataset(config)
    described = dataset.describe()
    weights = (
        file_sha256(extractor.weights)
        if isinstance(extractor, InceptionFeatures)
        else type(extractor).__name__
    )
    key = {
        "version": 1,
        "data": described["data"],
        "split": described["split"],
        "samples": described["samples"],
        "dropped": described["dropped"],
        "manifest": manifest_info(Path(config.reference.root)),
        "extractor": weights,
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def reference_features(
    config: RealWorldConfig, extractor: FeatureExtractor, device: torch.device
) -> npt.NDArray[np.float32]:
    """Features of the reference HR crops, from the cache when it matches."""
    fingerprint = reference_fingerprint(config, extractor)
    path = Path(config.output) / CACHE / f"inception_ref-{fingerprint[:16]}.npz"
    if path.is_file():
        with np.load(path) as stored:
            if str(stored["fingerprint"]) == fingerprint:
                log.info("using cached reference features from %s", path)
                cached: npt.NDArray[np.float32] = stored["features"]
                return cached
    dataset = reference_dataset(config)
    log.info("computing reference features: %s HR crops", len(dataset))
    chunks = []
    for batch in make_loader(dataset, config.batch_size, config.workers, device):
        chunks.append(extractor(to_uint8(batch["hr"])).cpu().numpy().astype(np.float32))
    features = np.concatenate(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial.npz")
    np.savez(partial, fingerprint=np.array(fingerprint), features=features)
    partial.replace(path)
    return features


def reference_info(config: RealWorldConfig, count: int) -> dict[str, Any]:
    return {
        "root": config.reference.root,
        "split": config.reference_split,
        "samples": count,
        "hr_size": config.reference.hr_size,
        "manifest": manifest_info(Path(config.reference.root)),
    }


def metric_setup(config: RealWorldConfig, extractor: FeatureExtractor | None) -> dict[str, Any]:
    setup: dict[str, Any] = {
        "sr_quantized_to_8_bit": True,
        "lr_psnr_y": "SR downscaled with torch bicubic + antialias, rounded to 8 bit; PSNR-Y vs LR",
        "features": None,
    }
    if extractor is not None:
        weights = getattr(extractor, "weights", None)
        setup["features"] = {
            "extractor": type(extractor).__name__,
            "weights": str(weights) if weights else None,
            "sha256": file_sha256(weights) if weights else None,
            "resize": "bilinear to 299 px, TensorFlow 1.x compatible (antialias=False)",
            "fid": "all faces vs all reference faces; float64; eigenvalue square root",
            "kid": {
                "kernel": "(x . y / d + 1) ** 3",
                "subsets": config.metrics.kid_subsets,
                "subset_size": config.metrics.kid_subset_size,
                "min_faces": config.metrics.kid_min_faces,
                "seed": config.seed,
            },
        }
    return setup


def format_all(stats: dict[str, Any]) -> str:
    parts = [f"LR-PSNR-Y {stats['lr_psnr_y']['mean']:.2f}"]
    if stats.get("fid") is not None:
        parts.append(f"FID {stats['fid']:.2f}")
    if stats.get("kid"):
        parts.append(f"KID {stats['kid']['mean']:.5f}")
    return ", ".join(parts)
