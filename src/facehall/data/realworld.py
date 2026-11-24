"""Prepared real-world LR crops without HR reference (``scripts/prepare_wider.py``).

The dataset reads the PNG crops of one scale and split from the manifest (docs/DATA.md §7.4),
in manifest order. Samples are deterministic: there is no degradation and no augmentation.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import torch
from PIL import Image
from torch.utils.data import Dataset

from facehall.data.manifest import SPLITS, read_manifest

Sample = dict[str, Any]


class RealFaces(Dataset[Sample]):
    """LR crops of one ``scale`` and ``split``. Each sample: ``lr`` (3, h, w) float32 in [0, 1]
    on 8-bit levels, and ``index``."""

    def __init__(self, root: Path | str, split: str, scale: int) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        self.root = Path(root)
        self.split = split
        self.scale = scale
        table = read_manifest(self.root)
        if "scale" not in table.schema.names:
            raise ValueError(f"{self.root}: the manifest has no 'scale' column")
        mask = pc.and_(pc.equal(table["split"], split), pc.equal(table["scale"], scale))
        self._table = table.filter(mask)
        if self._table.num_rows == 0:
            raise ValueError(f"{self.root}: no crops for scale {scale} in split {split!r}")
        self._paths: list[str] = self._table.column("path").to_pylist()
        widths, heights = self._table["width"].to_pylist(), self._table["height"].to_pylist()
        sizes = set(zip(widths, heights, strict=True))
        if len(sizes) != 1:
            raise ValueError(f"{self.root}: crops of scale {scale} have several sizes {sizes}")
        self.lr_size = sizes.pop()[0]

    def __len__(self) -> int:
        return len(self._paths)

    def path(self, index: int) -> Path:
        return self.root / self._paths[index]

    def column(self, name: str) -> npt.NDArray[Any]:
        """A manifest column in dataset order (e.g. ``attr_blur``)."""
        return np.asarray(self._table.column(name).to_numpy(zero_copy_only=False))

    def has_column(self, name: str) -> bool:
        return name in self._table.schema.names

    def __getitem__(self, index: int) -> Sample:
        with Image.open(self.path(index)) as file:
            image = np.asarray(file.convert("RGB"))
        if image.shape[:2] != (self.lr_size, self.lr_size):
            raise ValueError(f"{self.path(index)}: expected {self.lr_size} px, got {image.shape}")
        lr = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return {"lr": lr, "index": index}

    def describe(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "split": self.split,
            "scale": self.scale,
            "lr_size": self.lr_size,
            "samples": len(self),
        }
