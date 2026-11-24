"""Shared fixtures."""

import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from facehall.align.template import CELEBA_TRAIN_MEAN_LANDMARKS
from facehall.data.manifest import REQUIRED_FIELDS, write_manifest

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
    """Images (train PNG, val/test JPEG) and a manifest with landmarks and attr_blurry."""
    columns: dict[str, list[object]] = {f.name: [] for f in REQUIRED_FIELDS}
    columns["landmarks"], columns["attr_blurry"] = [], []
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
        }
        for name, value in values.items():
            columns[name].append(value)
    schema = pa.schema(
        [
            *REQUIRED_FIELDS,
            pa.field("landmarks", pa.list_(pa.int32(), 10)),
            pa.field("attr_blurry", pa.bool_()),
        ]
    )
    write_manifest(pa.Table.from_pydict(columns, schema=schema), root)
    return root


@pytest.fixture(scope="session")
def make_dataset() -> Callable[[Path, Sequence[Row]], Path]:
    """``make_dataset(root, rows)`` writes a synthetic prepared dataset to ``root``."""
    return write_synthetic_dataset
