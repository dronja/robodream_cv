"""Preview a degradation config on real crops (``scripts/preview_degradations.py``).

The grid shows dataset faces, so it is written only to local, gitignored paths (D19).
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from facehall.data.config import DataConfig, DegradeConfig
from facehall.data.dataset import AlignedFaces
from facehall.data.degrade import build_degradation


@dataclass
class Preview:
    grid: Image.Image  # rows: samples; columns: HR, then one LR per variant (nearest upscale)
    records: list[dict[str, Any]]  # drawn parameters of every LR
    seconds_per_call: float  # mean time of one degradation call


def preview(
    data: DataConfig,
    degrade: DegradeConfig,
    split: str,
    scale: int,
    *,
    count: int = 8,
    variants: int = 6,
    seed: int = 0,
) -> Preview:
    """Degrade ``count`` random HR crops of ``split`` ``variants`` times each.

    Variant ``v`` of sample ``i`` uses the generator seeded with ``(seed, i, v)``.
    """
    crops = AlignedFaces(data, DegradeConfig(), split, scale, seed=seed, augment=False)
    degradation = build_degradation(degrade)
    chosen = np.random.default_rng(seed).choice(len(crops), min(count, len(crops)), replace=False)
    indices = sorted(int(i) for i in chosen)
    size = data.hr_size
    grid = Image.new("RGB", (size * (variants + 1), size * len(indices)), "white")
    records: list[dict[str, Any]] = []
    elapsed = 0.0
    for row, index in enumerate(indices):
        hr = crops[index]["hr"]
        grid.paste(to_image(hr), (0, row * size))
        for variant in range(variants):
            start = time.perf_counter()
            degraded = degradation(hr, scale, np.random.default_rng([seed, index, variant]))
            elapsed += time.perf_counter() - start
            lr = to_image(degraded.lr).resize((size, size), Image.Resampling.NEAREST)
            grid.paste(lr, ((variant + 1) * size, row * size))
            records.append(
                {
                    "index": index,
                    "file": crops.path(index).name,
                    "variant": variant,
                    "params": degraded.params,
                }
            )
    return Preview(grid, records, elapsed / max(1, len(records)))


def to_image(image: torch.Tensor) -> Image.Image:
    pixels = (image.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(pixels)
