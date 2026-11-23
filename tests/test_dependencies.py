"""Smoke tests for the locked runtime dependencies (CPU only)."""

import importlib

import pytest
import torch
import torchvision

RUNTIME_MODULES = [
    "cv2",
    "numpy",
    "omegaconf",
    "PIL",
    "pyarrow",
    "safetensors.torch",
    "torch",
    "torchmetrics",
    "torchvision",
]


@pytest.mark.parametrize("name", RUNTIME_MODULES)
def test_runtime_dependency_imports(name: str) -> None:
    importlib.import_module(name)


def test_torchvision_native_ops_match_torch() -> None:
    # torchvision's compiled ops fail to load when its build does not match torch
    # (for example a CPU torchvision next to a CUDA torch).
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]
