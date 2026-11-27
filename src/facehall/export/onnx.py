"""ONNX export of restoration models with a parity check (Phase 5, PR 5.1).

    facehall export onnx --ckpt runs/<id>/best.safetensors --out models/x4.onnx

- **Graph.** The model in eval mode, so its output is clamped to [0, 1]. Input ``lr``:
  (N, 3, h, w) float32 RGB in [0, 1]; output ``sr``: (N, 3, H, W), the same. The batch axis is
  dynamic, so one session restores any number of faces.
- **Exporter.** ``torch.onnx.export`` with the dynamo exporter (it needs ``onnxscript``), at a
  pinned opset (``OPSET``), so a new PyTorch does not change the files silently.
- **Metadata.** The model record (``MODEL_KEY``, JSON) is stored in the ONNX metadata: the source
  checkpoint (run name, file, sha256), architecture, scale, crop sizes, tensor layout, the
  exporting versions, the measured parity and the notice of ``facehall.marking``. ONNX Runtime
  reads it (``read_record``), so inference needs neither PyTorch nor the ``onnx`` package.
- **Parity.** Every export runs ONNX Runtime (CPU) against PyTorch (CPU, float32) on a seeded
  batch of noise. Above the tolerance (``TOLERANCE``, ``CLAUDE.md`` §10) nothing is written.

The weights stay unpublished (D7, D13): an exported model is a local file like a checkpoint, and
``*.onnx`` is in ``.gitignore``.
"""

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from facehall import __version__

if TYPE_CHECKING:
    from torch import nn

MODEL_KEY = "facehall"  # metadata key of the model record
INPUT, OUTPUT = "lr", "sr"
OPSET = 20
TOLERANCE = 1e-3  # max abs diff from PyTorch, in [0, 1] units (CLAUDE.md §10)
PARITY_CROPS = 16
PARITY_SEED = 0
MAX_BATCH = 4096  # upper bound of the dynamic batch axis
QUIET_LOGGERS = ("onnxscript", "onnx_ir")


@dataclass(frozen=True)
class Exported:
    """An exported model: its file, its record and the parity measured on export."""

    path: Path
    record: dict[str, Any]
    max_abs_diff: float


def parity_inputs(
    lr_size: int, count: int = PARITY_CROPS, seed: int = PARITY_SEED
) -> npt.NDArray[np.float32]:
    """A seeded batch of uniform noise in [0, 1]: (count, 3, lr_size, lr_size)."""
    rng = np.random.default_rng(seed)
    return rng.random((count, 3, lr_size, lr_size), dtype=np.float32)


def session(
    model: Path | bytes,
    providers: Sequence[str | tuple[str, dict[str, str]]] = ("CPUExecutionProvider",),
) -> Any:
    """An ONNX Runtime session of a model file or of serialized model bytes; ``providers`` as
    ONNX Runtime takes them (names, or (name, options) pairs)."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    source = str(model) if isinstance(model, Path) else model
    return ort.InferenceSession(source, sess_options=options, providers=list(providers))


def run(onnx_session: Any, lr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Restore a batch of LR crops with an ONNX Runtime session."""
    (sr,) = onnx_session.run([OUTPUT], {INPUT: np.ascontiguousarray(lr, dtype=np.float32)})
    return np.asarray(sr, dtype=np.float32)


def compare(model: "nn.Module", onnx_session: Any, lr: npt.NDArray[np.float32]) -> dict[str, float]:
    """Differences between the PyTorch model (CPU, float32) and the ONNX session on ``lr``.

    ``max_abs_diff`` and ``mean_abs_diff`` are in [0, 1] units; after rounding both outputs to
    8-bit levels, ``max_level_diff`` is the largest difference in levels and ``levels_changed``
    the fraction of values that differ.
    """
    import torch

    with torch.no_grad():
        expected = model.cpu().eval()(torch.from_numpy(lr)).float().numpy()
    got = run(onnx_session, lr)
    if got.shape != expected.shape:
        raise ValueError(f"ONNX output {got.shape} differs from PyTorch's {expected.shape}")
    diff = np.abs(got.astype(np.float64) - expected)
    levels = np.abs(np.round(got * 255.0) - np.round(expected * 255.0))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_level_diff": float(levels.max()),
        "levels_changed": float((levels > 0).mean()),
    }


def max_abs_diff(model: "nn.Module", onnx_session: Any, lr: npt.NDArray[np.float32]) -> float:
    """Largest absolute difference between the PyTorch model and the ONNX session on ``lr``."""
    return compare(model, onnx_session, lr)["max_abs_diff"]


def export_model(model: "nn.Module", lr_size: int, *, opset: int = OPSET) -> Any:
    """Export ``model`` (eval mode, on CPU) with a dynamic batch axis; returns the ONNX program."""
    import torch

    model = model.cpu().eval()
    example = torch.from_numpy(parity_inputs(lr_size, count=2))
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    quiet = [logging.getLogger(name) for name in QUIET_LOGGERS]
    levels = [logger.level for logger in quiet]
    for logger in quiet:  # the graph passes log every step at INFO
        logger.setLevel(logging.WARNING)
    try:
        program = torch.onnx.export(
            model,
            (example,),
            dynamo=True,
            verbose=False,
            input_names=[INPUT],
            output_names=[OUTPUT],
            dynamic_shapes=({0: batch},),
            opset_version=opset,
        )
    finally:
        for logger, level in zip(quiet, levels, strict=True):
            logger.setLevel(level)
    if program is None:  # only the TorchScript exporter returns None
        raise RuntimeError("torch.onnx.export returned no program")
    return program


def read_record(model: Path | Any) -> dict[str, Any]:
    """The model record of an exported model (a file or an ONNX Runtime session)."""
    onnx_session = session(model) if isinstance(model, Path) else model
    text = onnx_session.get_modelmeta().custom_metadata_map.get(MODEL_KEY)
    if text is None:
        raise ValueError(f"{model}: not a facehall ONNX model (no '{MODEL_KEY}' metadata)")
    record: dict[str, Any] = json.loads(text)
    return record


def export_checkpoint(
    checkpoint: Path,
    out: Path,
    *,
    opset: int = OPSET,
    tolerance: float = TOLERANCE,
    overwrite: bool = False,
) -> Exported:
    """Export a checkpoint to ``out`` after checking parity; nothing is written if it fails."""
    import torch

    from facehall.marking import NOTICE
    from facehall.provenance import package_versions
    from facehall.train.checkpoint import load_checkpoint, model_record
    from facehall.train.config import experiment_from_dict

    if out.suffix != ".onnx":
        raise ValueError(f"{out}: the output must end in .onnx")
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists (pass --overwrite to replace it)")
    model, info = load_checkpoint(checkpoint, "cpu")
    experiment = experiment_from_dict(info["config"])
    hr_size = experiment.data.hr_size
    lr_size = hr_size // experiment.scale
    program = export_model(model, lr_size, opset=opset)
    inputs = parity_inputs(lr_size)
    diff = max_abs_diff(model, session(program.model_proto.SerializeToString()), inputs)
    if not diff <= tolerance:
        raise ValueError(
            f"{checkpoint}: ONNX output differs from PyTorch by {diff:.2e}, "
            f"more than the tolerance {tolerance:.0e}; nothing written"
        )
    versions = package_versions(("torch", "onnxscript", "onnxruntime"))
    record = {
        **model_record(checkpoint, info),
        "format": "onnx",
        "lr_size": lr_size,
        "hr_size": hr_size,
        "tensors": {
            INPUT: f"float32 RGB in [0, 1], (N, 3, {lr_size}, {lr_size})",
            OUTPUT: f"float32 RGB in [0, 1], (N, 3, {hr_size}, {hr_size})",
        },
        "opset": opset,
        "exported_with": {"facehall": __version__, **versions, "torch": torch.__version__},
        "parity": {
            "max_abs_diff": diff,
            "tolerance": tolerance,
            "inputs": f"{len(inputs)} crops of uniform noise, seed {PARITY_SEED}",
            "runtime": f"onnxruntime {versions['onnxruntime']}, CPUExecutionProvider",
        },
        "notice": NOTICE,
    }
    program.model.metadata_props[MODEL_KEY] = json.dumps(record, sort_keys=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.name}.partial")
    try:
        program.save(partial, external_data=False)
        os.replace(partial, out)
    finally:
        partial.unlink(missing_ok=True)
    return Exported(out, record, diff)


def describe(exported: Exported) -> str:
    """One line about an export, for the command line."""
    r: Mapping[str, Any] = exported.record
    size = exported.path.stat().st_size / 1e6
    return (
        f"exported {r['name']} (x{r['scale']}, {r['architecture']}) to {exported.path}: "
        f"{size:.1f} MB, opset {r['opset']}, max abs diff from PyTorch "
        f"{exported.max_abs_diff:.1e} (tolerance {r['parity']['tolerance']:.0e})"
    )
