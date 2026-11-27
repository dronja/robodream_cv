"""Restoration backends of the pipeline: PyTorch checkpoints and ONNX exports (PR 5.2, D34).

A restorer maps a batch of LR crops to HR crops, both float32 RGB in [0, 1], (N, 3, h, w) →
(N, 3, H, W), and carries the model record that output marks cite (``info``).

- ``TorchRestorer``: a PyTorch model, from a checkpoint (``.safetensors``).
- ``OnnxRestorer``: an ONNX export (``facehall export onnx``) through ONNX Runtime, on the CPU
  or the CUDA execution provider (D35). TF32 is off on CUDA: with it, outputs moved by up to
  5e-4 from PyTorch on CPU on real crops, without it 1.3e-6, for about 1 ms more per 32 faces
  (PR 5.2).
- ``load_restorer`` picks the backend by the file suffix.

Nothing here imports PyTorch at module level: with an ONNX model, the pipeline runs where
PyTorch is not installed (the service images, D34).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from facehall.provenance import file_sha256

if TYPE_CHECKING:
    from torch import nn

Batch = npt.NDArray[np.float32]
CPU, CUDA = "CPUExecutionProvider", "CUDAExecutionProvider"
CUDA_OPTIONS = {"use_tf32": "0"}  # full float32, like the export check
Provider = str | tuple[str, dict[str, str]]
RECORD_KEYS = ("name", "checkpoint", "sha256", "architecture", "scale")


class Restorer(Protocol):
    scale: int
    lr_size: int
    hr_size: int
    device: str  # "cpu" or "cuda": where the model runs
    info: dict[str, Any]  # model record for output marks

    def __call__(self, lr: Batch) -> Batch: ...


class TorchRestorer:
    """A PyTorch restoration model in eval mode."""

    def __init__(
        self,
        model: "nn.Module",
        scale: int,
        *,
        hr_size: int = 128,
        device: str = "cpu",
        info: dict[str, Any] | None = None,
    ) -> None:
        import torch

        if hr_size % scale:
            raise ValueError(f"HR size {hr_size} is not a multiple of scale {scale}")
        self.torch_device = torch.device(device)
        self.model = model.to(self.torch_device).eval()
        self.scale, self.hr_size, self.lr_size = scale, hr_size, hr_size // scale
        self.device = self.torch_device.type
        self.info: dict[str, Any] = info if info is not None else {"scale": scale}

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu") -> "TorchRestorer":
        from facehall.train.checkpoint import load_checkpoint, model_record
        from facehall.train.config import experiment_from_dict

        model, info = load_checkpoint(path, device)
        experiment = experiment_from_dict(info["config"])
        return cls(
            model,
            experiment.scale,
            hr_size=experiment.data.hr_size,
            device=device,
            info=model_record(path, info),
        )

    def __call__(self, lr: Batch) -> Batch:
        import torch

        with torch.no_grad():
            sr = self.model(torch.from_numpy(lr).to(self.torch_device))
        return np.asarray(sr.float().cpu().numpy(), dtype=np.float32)


def onnx_providers(device: str) -> list[Provider]:
    """ONNX Runtime execution providers for ``device`` ("auto", "cpu" or "cuda")."""
    import onnxruntime as ort

    if device == "cpu":
        return [CPU]
    if device not in ("auto", "cuda"):
        raise ValueError(f"an ONNX model runs on auto, cpu or cuda, not {device!r}")
    if CUDA in ort.get_available_providers():
        return [(CUDA, dict(CUDA_OPTIONS)), CPU]
    if device == "cuda":
        raise RuntimeError(
            "this onnxruntime has no CUDA provider; install onnxruntime-gpu for --device cuda"
        )
    return [CPU]


class OnnxRestorer:
    """An ONNX export of a restoration model (``facehall.export``) in ONNX Runtime."""

    def __init__(self, path: Path, *, device: str = "auto") -> None:
        import onnxruntime as ort

        from facehall.export.onnx import read_record, session

        providers = onnx_providers(device)
        cuda = any(isinstance(p, tuple) and p[0] == CUDA for p in providers)
        if cuda and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # the CUDA and cuDNN libraries of the nvidia-* wheels
        self.session = session(path, providers)
        active = self.session.get_providers()
        if cuda and CUDA not in active and device == "cuda":
            raise RuntimeError(f"ONNX Runtime could not start the CUDA provider for {path}")
        record = read_record(self.session)
        if record.get("format") != "onnx":
            raise ValueError(f"{path}: unknown model record format {record.get('format')!r}")
        self.path = path
        self.scale, self.lr_size, self.hr_size = (
            int(record["scale"]),
            int(record["lr_size"]),
            int(record["hr_size"]),
        )
        self.provider = active[0]
        self.device = "cuda" if self.provider == CUDA else "cpu"
        self.info: dict[str, Any] = {
            **{key: record[key] for key in RECORD_KEYS},
            "format": "onnx",
            "onnx_sha256": file_sha256(path),
            "runtime": f"onnxruntime {ort.__version__}, {self.provider}",
        }

    def __call__(self, lr: Batch) -> Batch:
        from facehall.export.onnx import run

        return run(self.session, lr)


def load_restorer(path: Path, device: str = "auto") -> Restorer:
    """The restorer of a model file: an ONNX export (``.onnx``) or a checkpoint."""
    if path.suffix == ".onnx":
        return OnnxRestorer(path, device=device)
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TorchRestorer.from_checkpoint(path, device)
