"""Score one model on one evaluation protocol (``facehall eval``).

Output directory ``<output>/<protocol>/<split>/<model label>/``:

- ``metrics.json``: means and standard deviations per subset, the metric setup and provenance
  (git commit, versions, device, checkpoint and recognition-model checksums, dataset counts),
  and ``spectrum``: the SR / HR power-spectrum ratios over all images, a periodic-pattern check
  (``facehall.eval.spectrum``, D28);
- ``per_image.parquet``: every metric per image, with the ``padded``, ``blurry`` and subgroup
  columns and the drawn degradation parameters (``degradation``, JSON), so other breakdowns need
  no re-run;
- ``samples/``: SR crops of the grid samples. LR and HR of the same samples go to
  ``../reference/``. Both contain CelebA faces and stay local (D19).

The HR embeddings of the identity metric do not depend on the model, and usually not on the
protocol's scale or degradation, so they are cached once per fingerprint in
``<output>/_cache/identity_hr-<fingerprint>.npz``. The fingerprint covers everything they depend
on (dataset settings and counts, manifest checksum, recognition model checksum, and the
degradation if it changes the HR target).

Frozen inputs: every run records the sha256 of all its LR and HR crops (``InputDigest``). When the
protocol lists a digest for the split (``inputs_sha256``), a different digest is an error: the
inputs changed (e.g. another OpenCV JPEG encoder), so results would not be comparable.

SR outputs are rounded to 8-bit levels before scoring, as they would be when saved as images.
Subsets: ``all``; ``non_blurry`` (D15); ``unpadded`` (D16); and ``<attr>`` / ``not_<attr>`` for
every boolean manifest column in ``subgroups`` (e.g. CelebA ``Male``, ``Young``).
"""

import hashlib
import json
import logging
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import nn

from facehall.data.dataset import TEMPLATES, AlignedFaces
from facehall.eval.config import EvalConfig, validate_eval
from facehall.eval.metrics import (
    ArcFaceOnnx,
    IdentitySimilarity,
    Lpips,
    cosine,
    psnr_y,
    ssim_y,
)
from facehall.eval.spectrum import SpectrumRatio
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

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
METRIC_NAMES = ("psnr_y", "ssim_y", "lpips", "identity")
CACHE = "_cache"  # under the output root; not a protocol directory
CPU = torch.device("cpu")


class InputsChangedError(RuntimeError):
    """The inputs of a frozen protocol differ from the recorded digest."""


class InputDigest:
    """sha256 of every LR and HR crop, in dataset order, at 8-bit levels.

    LR and HR are hashed as two streams, so the digest does not depend on the batch size.
    """

    def __init__(self) -> None:
        self.lr = hashlib.sha256()
        self.hr = hashlib.sha256()

    def update(self, lr: torch.Tensor, hr: torch.Tensor) -> None:
        self.lr.update(to_bytes(lr))
        self.hr.update(to_bytes(hr))

    def hexdigest(self) -> str:
        return hashlib.sha256(self.lr.digest() + self.hr.digest()).hexdigest()


def to_bytes(images: torch.Tensor) -> bytes:
    pixels = (images.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return bytes(pixels.contiguous().numpy().tobytes())


def protocol_dataset(config: EvalConfig) -> AlignedFaces:
    return AlignedFaces(
        config.data, config.degrade, config.split, config.scale, seed=config.seed, augment=False
    )


def input_digest(config: EvalConfig) -> str:
    """The inputs digest of a protocol, without a model (``scripts/protocol_digest.py``)."""
    validate_eval(config)
    if config.workers > 0:
        cv2.setNumThreads(1)  # before the DataLoader workers fork (docs/PROJECT.md §9)
    digest = InputDigest()
    for batch in make_loader(protocol_dataset(config), config.batch_size, config.workers, CPU):
        digest.update(batch["lr"], batch["hr"])
    return digest.hexdigest()


def check_inputs(config: EvalConfig, digest: str) -> None:
    expected = config.inputs_sha256.get(config.split)
    if expected and expected != digest:
        raise InputsChangedError(
            f"the {config.split} inputs of protocol {config.name} changed: sha256 {digest}, "
            f"frozen {expected}. Results would not be comparable with earlier ones. Check the "
            "library versions (uv.lock) and the data; refreeze only on purpose, and then "
            "re-evaluate every model"
        )


def quantize(images: torch.Tensor) -> torch.Tensor:
    """Clamp to [0, 1] and round to 8-bit levels."""
    return torch.round(images.clamp(0.0, 1.0) * 255.0) / 255.0


def grid_indices(size: int, count: int, seed: int) -> list[int]:
    count = min(count, size)
    return sorted(np.random.default_rng(seed).choice(size, count, replace=False).tolist())


def build_metrics(
    config: EvalConfig, device: torch.device
) -> tuple[dict[str, Metric], IdentitySimilarity | None]:
    """The pixel and perceptual metrics enabled in ``config``, and the identity metric."""
    metrics: dict[str, Metric] = {"psnr_y": psnr_y, "ssim_y": ssim_y}
    if config.metrics.lpips:
        metrics["lpips"] = Lpips(config.metrics.lpips, device)
    identity = None
    if config.metrics.identity:
        embedder = ArcFaceOnnx(config.metrics.identity, threads=config.metrics.identity_threads)
        template = TEMPLATES[config.data.template]
        identity = IdentitySimilarity(embedder, template, config.data.hr_size)
    return metrics, identity


def evaluate_model(
    model: nn.Module,
    config: EvalConfig,
    label: str,
    model_info: dict[str, Any],
    *,
    metrics: tuple[dict[str, Metric], IdentitySimilarity | None] | None = None,
) -> Path:
    """Score ``model`` on the protocol ``config``; returns the output directory.

    ``metrics`` defaults to the ones enabled in ``config`` (``build_metrics``).
    """
    validate_eval(config)
    device = resolve_device(config.device)
    if config.workers > 0:
        cv2.setNumThreads(1)  # before the DataLoader workers fork (docs/PROJECT.md §9)
    dataset = protocol_dataset(config)
    subgroups = {name: dataset.column(name).astype(bool) for name in config.subgroups}
    scores, identity = metrics if metrics is not None else build_metrics(config, device)
    model = model.to(device).eval()
    out = Path(config.output) / config.name / config.split / label
    (out / "samples").mkdir(parents=True, exist_ok=True)
    reference = out.parent / "reference"
    grid = set(grid_indices(len(dataset), config.grid_samples, config.seed))
    fingerprint = identity_fingerprint(dataset, config) if identity is not None else ""
    cache_path = Path(config.output) / CACHE / f"identity_hr-{fingerprint[:16]}.npz"
    cached = load_embeddings(cache_path, fingerprint, len(dataset)) if identity else None
    hr_embeddings: list[npt.NDArray[np.float64]] = []
    spectrum = SpectrumRatio(config.data.hr_size, config.data.hr_size, device)

    names = [*scores, *(["identity"] if identity is not None else [])]
    values: dict[str, list[npt.NDArray[np.float64]]] = {name: [] for name in names}
    indices: list[npt.NDArray[np.int64]] = []
    padded: list[npt.NDArray[np.float64]] = []
    blurry: list[npt.NDArray[np.bool_]] = []
    degradations: list[str] = []
    inputs = InputDigest()
    start = time.perf_counter()
    log.info("evaluating %s on %s/%s: %s samples", label, config.name, config.split, len(dataset))
    with torch.no_grad():
        for batch in make_loader(dataset, config.batch_size, config.workers, device):
            inputs.update(batch["lr"], batch["hr"])
            degradations.extend(batch["degradation"])
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            sr = quantize(model(lr).float())
            spectrum.update(sr, hr)
            batch_indices = batch["index"].numpy()
            for name, metric in scores.items():
                values[name].append(metric(sr, hr).double().cpu().numpy())
            if identity is not None:
                hr_embedding = cached[batch_indices] if cached is not None else identity.embed(hr)
                hr_embeddings.append(hr_embedding)
                values["identity"].append(cosine(identity.embed(sr), hr_embedding))
            indices.append(batch_indices)
            padded.append(batch["padded"].double().numpy())
            blurry.append(batch["blurry"].numpy())
            for row, index in enumerate(batch_indices.tolist()):
                if index in grid:
                    save_samples(index, lr[row], sr[row], hr[row], out / "samples", reference)
    seconds = time.perf_counter() - start
    digest = inputs.hexdigest()
    check_inputs(config, digest)
    if identity is not None and cached is None:
        order = np.argsort(np.concatenate(indices))
        save_embeddings(cache_path, fingerprint, np.concatenate(hr_embeddings)[order])

    columns: dict[str, npt.NDArray[Any]] = {
        "index": np.concatenate(indices),
        "padded": np.concatenate(padded),
        "blurry": np.concatenate(blurry),
    }
    order = columns["index"]
    columns |= {name: column[order] for name, column in subgroups.items()}
    columns |= {name: np.concatenate(chunks) for name, chunks in values.items()}
    files = [dataset.path(i).name for i in order.tolist()]
    table = pa.table({**columns, "path": files, "degradation": degradations})
    pq.write_table(table, out / "per_image.parquet")

    subsets = summarize(columns, names, list(subgroups))
    spectral = spectrum.summary()
    summary = {
        "protocol": config.name,
        "level": config.level or config.name,
        "split": config.split,
        "scale": config.scale,
        "label": label,
        "model": model_info,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": seconds,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "versions": package_versions((*VERSIONED, "torchmetrics", "onnxruntime")),
        "device": describe_device(device),
        "dataset": dataset.describe(),
        "inputs_sha256": digest,
        "manifest": manifest_info(Path(config.data.root)),
        "metric_setup": metric_setup(config),
        "subsets": subsets,
        "spectrum": spectral,
    }
    write_json(out / "metrics.json", summary)
    log.info(
        "%s: %s, spectral peak %.2f in %.0f s",
        label,
        format_means(subsets["all"]),
        spectral["peak_ratio"],
        seconds,
    )
    return out


def identity_fingerprint(dataset: AlignedFaces, config: EvalConfig) -> str:
    """sha256 of everything the HR embeddings depend on.

    The scale and the degradation do not change the HR target, except a degradation with
    grayscale (``changes_target``); only then is the degradation part of the key. Neither does
    ``data.scene``, which only changes how the LR input is made, so it is left out.
    """
    described = dataset.describe()
    data = dict(described["data"])  # type: ignore[call-overload]
    data.pop("scene", None)
    key = {
        "version": 1,
        "data": data,
        "split": described["split"],
        "samples": described["samples"],
        "dropped": described["dropped"],
        "manifest": manifest_info(Path(config.data.root)),
        "identity_model": (
            file_sha256(Path(config.metrics.identity)) if config.metrics.identity else "injected"
        ),
    }
    degradation = described["degradation"]
    if isinstance(degradation, dict) and degradation.get("changes_target"):
        key["degradation"] = degradation
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def load_embeddings(path: Path, fingerprint: str, size: int) -> npt.NDArray[np.float64] | None:
    if not path.is_file():
        return None
    with np.load(path) as stored:
        if str(stored["fingerprint"]) != fingerprint or len(stored["embeddings"]) != size:
            log.info("ignoring %s: it was computed for other settings", path)
            return None
        embeddings: npt.NDArray[np.float64] = stored["embeddings"].astype(np.float64)
    log.info("using cached HR embeddings from %s", path)
    return embeddings


def save_embeddings(path: Path, fingerprint: str, embeddings: npt.NDArray[np.float64]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial.npz")
    np.savez(partial, fingerprint=np.array(fingerprint), embeddings=embeddings)
    partial.replace(path)


def summarize(
    columns: dict[str, npt.NDArray[Any]], metric_names: list[str], subgroups: list[str]
) -> dict[str, dict[str, Any]]:
    """Count, mean and standard deviation of every metric for every subset."""
    size = len(columns["index"])
    masks: dict[str, npt.NDArray[np.bool_]] = {
        "all": np.ones(size, dtype=bool),
        "non_blurry": ~columns["blurry"],
        "unpadded": columns["padded"] == 0.0,
    }
    for name in subgroups:
        short = name.removeprefix("attr_")
        masks[short] = columns[name]
        masks[f"not_{short}"] = ~columns[name]
    result: dict[str, dict[str, Any]] = {}
    for subset, mask in masks.items():
        stats: dict[str, Any] = {"count": int(mask.sum())}
        for metric in metric_names:
            selected = columns[metric][mask]
            stats[metric] = (
                {"mean": float(selected.mean()), "std": float(selected.std())}
                if len(selected)
                else None
            )
        result[subset] = stats
    return result


def metric_setup(config: EvalConfig) -> dict[str, Any]:
    setup: dict[str, Any] = {
        "sr_quantized_to_8_bit": True,
        "psnr_y": "luma Y of BT.601 in [16, 235], peak 255, no border crop",
        "ssim_y": "torchmetrics SSIM on Y: Gaussian window 11, sigma 1.5, data range 255",
        "lpips": {"net": config.metrics.lpips} if config.metrics.lpips else None,
        "identity": None,
        "spectrum": "luma (BT.601 weights), mean removed, 2-D Hann window; power summed over all "
        "images; SR / HR per frequency bin, peak over radius >= 0.25 cycles/px",
    }
    if config.metrics.identity:
        path = Path(config.metrics.identity)
        setup["identity"] = {
            "model": str(path),
            "sha256": file_sha256(path),
            "alignment": "fixed similarity from the crop template to InsightFace arcface_dst",
        }
    return setup


def save_samples(
    index: int, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, samples: Path, reference: Path
) -> None:
    for folder, image in ((samples, sr), (reference / "lr", lr), (reference / "hr", hr)):
        folder.mkdir(parents=True, exist_ok=True)
        to_image(image).save(folder / f"{index:06d}.png")


def to_image(image: torch.Tensor) -> Image.Image:
    pixels = (image.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels)


def format_means(stats: dict[str, Any]) -> str:
    parts = [f"{name} {stats[name]['mean']:.4f}" for name in METRIC_NAMES if stats.get(name)]
    return ", ".join(parts)
