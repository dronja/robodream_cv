"""Shared fixtures."""

import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from facehall.align.template import CELEBA_TRAIN_MEAN_LANDMARKS
from facehall.data.manifest import REQUIRED_FIELDS, UNKNOWN_IDENTITY, write_manifest
from facehall.data.wider import LICENSE, SOURCE, manifest_schema

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="session")
def load_script() -> Callable[[str], ModuleType]:
    """Import ``scripts/<name>.py`` as a module (scripts are not part of the package)."""

    def load(name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    return load


# A synthetic prepared dataset (docs/DATA.md layout): one row is (split, blurry, offset), where
# offset shifts the CelebA mean train landmarks by (dx, dy) px, or is None for BROKEN_LANDMARKS.
Row = tuple[str, bool, tuple[int, int] | None]
BROKEN_LANDMARKS = [84, 105, 94, 104, 91, 109, 88, 116, 95, 114]  # all points between the eyes


def synthetic_image(seed: int) -> Image.Image:
    """A smooth 178x218 RGB pattern that differs per seed."""
    y, x = np.mgrid[0:218, 0:178]
    channels = [np.sin(x / (5.0 + seed) + y / 11.0), np.cos(y / 7.0 - seed), np.sin((x - y) / 13.0)]
    pixels = (127.5 + 120 * np.stack(channels, axis=-1)).astype(np.uint8)
    return Image.fromarray(pixels)


def write_synthetic_dataset(root: Path, rows: Sequence[Row]) -> Path:
    """Images (train PNG, val/test JPEG) and a manifest with landmarks and three attributes:
    ``attr_blurry`` from the row, ``attr_male`` = even row, ``attr_young`` = row not divisible
    by 3."""
    columns: dict[str, list[object]] = {f.name: [] for f in REQUIRED_FIELDS}
    for name in ("landmarks", "attr_blurry", "attr_male", "attr_young"):
        columns[name] = []
    for i, (split, blurry, offset) in enumerate(rows):
        suffix = ".png" if split == "train" else ".jpg"
        path = f"img_align/{split}/{i:06d}{suffix}"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        synthetic_image(i).save(root / path)
        if offset is None:
            landmarks = BROKEN_LANDMARKS
        else:
            points = np.round(CELEBA_TRAIN_MEAN_LANDMARKS + offset).astype(int)
            landmarks = points.reshape(-1).tolist()
        values = {
            "path": path,
            "sha256": f"{i:064x}",
            "split": split,
            "identity_id": i,
            "width": 178,
            "height": 218,
            "source": "synthetic",
            "license": "test",
            "landmarks": landmarks,
            "attr_blurry": blurry,
            "attr_male": i % 2 == 0,
            "attr_young": i % 3 != 0,
        }
        for name, value in values.items():
            columns[name].append(value)
    schema = pa.schema(
        [
            *REQUIRED_FIELDS,
            pa.field("landmarks", pa.list_(pa.int32(), 10)),
            pa.field("attr_blurry", pa.bool_()),
            pa.field("attr_male", pa.bool_()),
            pa.field("attr_young", pa.bool_()),
        ]
    )
    write_manifest(pa.Table.from_pydict(columns, schema=schema), root)
    return root


@pytest.fixture(scope="session")
def make_dataset() -> Callable[[Path, Sequence[Row]], Path]:
    """``make_dataset(root, rows)`` writes a synthetic prepared dataset to ``root``."""
    return write_synthetic_dataset


class ImageStats:
    """A stand-in for InceptionV3: per-channel means and standard deviations, 2x2 block means."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert images.dtype == torch.uint8
        self.calls += 1
        x = images.float() / 255.0
        blocks = F.adaptive_avg_pool2d(x, 2).flatten(1)
        return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3)), blocks], dim=1)


@pytest.fixture
def image_stats() -> Callable[[], Any]:
    """``image_stats()`` returns a new ``ImageStats`` feature extractor."""
    return ImageStats


def write_real_crops(root: Path, crops: Sequence[dict[str, Any]]) -> Path:
    """A prepared real-world set (``scripts/prepare_wider.py`` layout) from LR crops.

    Every crop has ``scale``, ``split`` and ``pixels`` (h, w, 3 uint8); ``attr_*`` and
    ``padded`` default to 0.
    """
    rows = []
    counters: dict[tuple[int, str], int] = {}
    for i, crop in enumerate(crops):
        scale, split, pixels = crop["scale"], crop["split"], crop["pixels"]
        index = counters.get((scale, split), 0)
        counters[(scale, split)] = index + 1
        path = f"x{scale}/{split}/{index:06d}.png"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(root / path)
        size = pixels.shape[0]
        rows.append(
            {
                "path": path,
                "sha256": f"{i:064x}",
                "split": split,
                "identity_id": UNKNOWN_IDENTITY,
                "width": size,
                "height": size,
                "source": SOURCE,
                "license": LICENSE,
                "scale": scale,
                "source_image": f"0--Event/{i // 2}.jpg",
                "event": "0--Event",
                "face_index": i % 2,
                "bbox": [0, 0, size, size],
                "landmarks": [0.0] * 10,
                "align_scale": 1.0,
                "align_angle": 0.0,
                "align_residual": 0.01,
                "padded": crop.get("padded", 0.0),
                **{
                    f"attr_{name}": crop.get(f"attr_{name}", 0)
                    for name in ("blur", "expression", "illumination", "occlusion", "pose")
                },
            }
        )
    write_manifest(pa.Table.from_pylist(rows, schema=manifest_schema()), root)
    return root


@pytest.fixture(scope="session")
def make_realworld() -> Callable[[Path, Sequence[dict[str, Any]]], Path]:
    """``make_realworld(root, crops)`` writes a prepared real-world set to ``root``."""
    return write_real_crops
