"""Aligned HR/LR face pairs from a prepared manifest (docs/PROJECT.md §9, PR 1.4).

Every sample is cut from the source image on the fly: the alignment matrix (from the manifest
landmarks) is optionally perturbed by jitter and a horizontal flip, the image is warped once into
the HR crop, and the degradation produces the LR input.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import torch
from PIL import Image
from torch.utils.data import Dataset

from facehall.align import (
    BORDERS,
    CELEBA_TEMPLATE,
    INTERPOLATIONS,
    Jitter,
    alignment_matrices,
    compose,
    hflip_matrix,
    valid_mask,
    warp,
)
from facehall.align.similarity import Points
from facehall.config import require_choice
from facehall.data.config import DataConfig, DegradeConfig
from facehall.data.degrade import build_degradation
from facehall.data.manifest import SPLITS, read_manifest

TEMPLATES: dict[str, Points] = {"celeba": CELEBA_TEMPLATE}

Sample = dict[str, Any]


class AlignedFaces(Dataset[Sample]):
    """HR/LR pairs for one split of a prepared dataset.

    A sample is a dict:
    - ``hr``: (3, S, S) float32 in [0, 1], S = ``data.hr_size``;
    - ``lr``: (3, S / scale, S / scale) float32 in [0, 1];
    - ``index``: position in this dataset (``path(index)`` gives the image file);
    - ``padded``: fraction of HR pixels that lie outside the source image (D16);
    - ``blurry``: the CelebA ``Blurry`` attribute (D15).

    Rows are dropped when the HR crop would be an upscale of the source (``CLAUDE.md`` §7), and
    ``Blurry = 1`` rows are dropped in the splits listed in ``data.exclude_blurry``.

    Randomness comes from a generator seeded with ``(seed, epoch, index)`` when augmenting and
    ``(seed, index)`` otherwise, so samples do not depend on DataLoader workers or order. Call
    ``set_epoch`` before each training epoch.
    """

    def __init__(
        self,
        data: DataConfig,
        degrade: DegradeConfig,
        split: str,
        scale: int,
        *,
        seed: int = 0,
        augment: bool | None = None,
    ) -> None:
        _check_settings(data, split, scale)
        self.config = data
        self.split = split
        self.scale = scale
        self.seed = seed
        self.epoch = 0
        self.augment = split == "train" if augment is None else augment
        self.root = Path(data.root)
        self._jitter = Jitter(**asdict(data.jitter))
        self._degradation = build_degradation(degrade)

        table = read_manifest(self.root)
        table = table.filter(pc.equal(table.column("split"), split))
        landmarks = table.column("landmarks").combine_chunks().flatten().to_numpy()
        matrices = alignment_matrices(
            np.asarray(landmarks, dtype=np.float64).reshape(-1, 5, 2),
            TEMPLATES[data.template],
            data.hr_size,
        )
        # Output pixels per source pixel above 1 means the HR crop would be an upscale.
        upscale = np.sqrt(np.abs(np.linalg.det(matrices[:, :, :2]))) > 1.0
        blurry = np.asarray(table.column("attr_blurry").to_numpy(zero_copy_only=False), dtype=bool)
        drop_blurry = blurry & (split in data.exclude_blurry) & ~upscale
        keep = ~(upscale | drop_blurry)

        self.dropped = {"upscale": int(upscale.sum()), "blurry": int(drop_blurry.sum())}
        paths: list[str] = table.column("path").to_pylist()
        self._paths = [p for p, k in zip(paths, keep, strict=True) if k]
        self._table = table.filter(pa.array(keep))
        self._matrices = matrices[keep]
        self._blurry = blurry[keep]

    def __len__(self) -> int:
        return len(self._paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def path(self, index: int) -> Path:
        return self.root / self._paths[index]

    def column(self, name: str) -> npt.NDArray[Any]:
        """A manifest column for the kept rows, in dataset order (e.g. ``attr_male``)."""
        return np.asarray(self._table.column(name).to_numpy(zero_copy_only=False))

    def __getitem__(self, index: int) -> Sample:
        size = self.config.hr_size
        rng = np.random.default_rng(
            [self.seed, self.epoch, index] if self.augment else [self.seed, index]
        )
        with Image.open(self.path(index)) as file:
            image = np.asarray(file.convert("RGB"))

        matrix = self._matrices[index]
        if self.augment:
            matrix = compose(self._jitter.sample(rng, size), matrix)
            if rng.random() < self.config.hflip:
                matrix = compose(hflip_matrix(size), matrix)

        hr = warp(
            image,
            matrix,
            size,
            interpolation=self.config.interpolation,
            border=self.config.border,
        )
        height, width = image.shape[:2]
        padded = 1.0 - float(valid_mask(matrix, (width, height), size).mean())
        hr_tensor = torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1).float() / 255.0
        return {
            "hr": hr_tensor,
            "lr": self._degradation(hr_tensor, self.scale, rng),
            "index": index,
            "padded": padded,
            "blurry": bool(self._blurry[index]),
        }

    def describe(self) -> dict[str, object]:
        """Settings and counts, recorded with every run."""
        return {
            "root": str(self.root),
            "split": self.split,
            "scale": self.scale,
            "samples": len(self),
            "dropped": self.dropped,
            "augment": self.augment,
            "seed": self.seed,
            "data": asdict(self.config),
            "degradation": self._degradation.describe(),
        }


def _check_settings(data: DataConfig, split: str, scale: int) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if scale < 1 or data.hr_size % scale:
        raise ValueError(f"scale {scale} does not divide hr_size {data.hr_size}")
    unknown = set(data.exclude_blurry) - set(SPLITS)
    if unknown:
        raise ValueError(f"exclude_blurry has unknown splits: {sorted(unknown)}")
    require_choice(data.template, TEMPLATES, "template")
    require_choice(data.interpolation, INTERPOLATIONS, "interpolation")
    require_choice(data.border, BORDERS, "border")
