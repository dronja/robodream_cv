"""Aligned HR/LR face pairs from a prepared manifest (docs/PROJECT.md §9, PR 1.4).

Every sample is cut from the source image on the fly: the alignment matrix (from the manifest
landmarks) is optionally perturbed by jitter and a horizontal flip, the image is warped once into
the HR crop, and the degradation produces the LR input. With ``data.scene`` the LR input is made
like the inference pipeline's instead: degraded as part of a larger scene, then cut from it by a
warp at LR resolution (``facehall.data.config.SceneConfig``).
"""

import json
import math
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
    invert,
    resample_crop,
    rescale_matrix,
    valid_mask,
    warp,
)
from facehall.align.similarity import Matrix, Points
from facehall.config import require_choice
from facehall.data.config import DataConfig, DegradeConfig, SceneConfig
from facehall.data.degrade import build_degradation, quantized, to_gray, to_tensor
from facehall.data.manifest import SPLITS, read_manifest

TEMPLATES: dict[str, Points] = {"celeba": CELEBA_TEMPLATE}

Sample = dict[str, Any]


class AlignedFaces(Dataset[Sample]):
    """HR/LR pairs for one split of a prepared dataset.

    A sample is a dict:
    - ``hr``: (3, S, S) float32 in [0, 1], S = ``data.hr_size``; the training target, turned
      grey when the degradation draws grayscale;
    - ``lr``: (3, S / scale, S / scale) float32 in [0, 1];
    - ``degradation``: the drawn degradation parameters as a JSON string (``Degraded.params``);
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
        self._scene_side = scene_side(data.scene, data.hr_size, scale)

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
        scene = self.config.scene
        if scene.prob > 0.0 and rng.random() < scene.prob:
            lr, target, params = self._scene_sample(image, hr, matrix, rng)
        else:
            degraded = self._degradation(to_float(hr), self.scale, rng)
            lr, target, params = degraded.lr, degraded.hr, degraded.params
        return {
            "hr": target,
            "lr": lr,
            "degradation": json.dumps(params, sort_keys=True),
            "index": index,
            "padded": padded,
            "blurry": bool(self._blurry[index]),
        }

    def _scene_sample(
        self,
        image: npt.NDArray[np.uint8],
        hr: npt.NDArray[Any],
        matrix: Matrix,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """LR crop cut from a degraded scene, the HR target and the drawn parameters."""
        size, scale, side = self.config.hr_size, self.scale, self._scene_side
        to_crop, drawn = scene_transform(self.config.scene, size, side, scale, rng)
        scene_hr = warp(
            image,
            compose(invert(to_crop), matrix),
            side,
            interpolation=self.config.interpolation,
            border=self.config.border,
        )
        degraded = self._degradation(to_float(scene_hr), scale, rng)
        scene_lr = np.rint(degraded.lr.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        to_lr = rescale_matrix(1.0 / scale)  # HR pixel-centre coordinates to LR ones
        lr_matrix = compose(to_lr, compose(to_crop, invert(to_lr)))
        lr = resample_crop(scene_lr, lr_matrix, size // scale, border="reflect")
        target = to_float(hr)
        if degraded.params.get("gray"):
            target = to_tensor(quantized(to_gray(hr.astype(np.float32) / 255.0)))
        return to_float(lr), target, {"scene": drawn, **degraded.params}

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
            "scene_side": self._scene_side,
        }


def to_float(image: npt.NDArray[Any]) -> torch.Tensor:
    """(H, W, 3) uint8 to (3, H, W) float32 in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0


def scene_side(config: SceneConfig, size: int, scale: int) -> int:
    """Side of the HR scene: it holds the crop at any drawn rotation, size and shift, with
    two LR pixels to spare for the bicubic warp, and its margin is a whole number of LR pixels
    on each side, so an untransformed crop sits on the scene's LR grid."""
    angle = math.radians(config.rotation)
    half = size / 2 * (1.0 + config.scale) * (math.cos(angle) + math.sin(angle))
    margin = half - size / 2 + (config.shift + 2.0) * scale
    return size + 2 * scale * max(1, math.ceil(margin / scale))


def scene_transform(
    config: SceneConfig, size: int, side: int, scale: int, rng: np.random.Generator
) -> tuple[Matrix, dict[str, Any]]:
    """A random similarity from a ``side`` scene to the ``size`` crop at its centre (HR pixel
    coordinates). Draws four numbers, whatever the settings."""
    dx, dy, angle_unit, scale_unit = rng.uniform(-1.0, 1.0, size=4)
    angle = math.radians(config.rotation * angle_unit)
    factor = (1.0 + config.scale) ** scale_unit
    cos, sin = math.cos(angle), math.sin(angle)
    linear = factor * np.array([[cos, -sin], [sin, cos]])
    shift = np.array([dx, dy]) * config.shift * scale
    offset = np.full(2, (size - 1) / 2) - linear @ np.full(2, (side - 1) / 2) + shift
    drawn = {
        "rotation": round(float(np.degrees(angle)), 4),
        "scale": round(float(factor), 5),
        "shift": [round(float(v), 4) for v in shift / scale],
    }
    transform: Matrix = np.hstack([linear, offset[:, None]])
    return transform, drawn


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
    scene = data.scene
    if not 0.0 <= scene.prob <= 1.0 or min(scene.shift, scene.rotation, scene.scale) < 0:
        raise ValueError(f"data.scene: prob must be in [0, 1] and limits non-negative: {scene}")
