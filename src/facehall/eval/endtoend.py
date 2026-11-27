"""End-to-end evaluation of the inference pipeline on synthetic CelebA scenes (PR 4.2).

A scene is an aligned CelebA image downscaled so that the face has the model's input size
(``facehall.eval.detection.make_scene``; 8 px between the eyes at ×4, 4 px at ×8), optionally
degraded. The variants, each a step closer to the product:

- ``bicubic``: the bicubic ×scale upscale of the scene (the pipeline's background);
- ``aligned``: the model on an LR crop made from the HR crop by area downscaling, then degraded
  like the scene. The face sits on the template's pixel grid, as in training, so no resampling
  happens at LR resolution: the upper bound for the pipeline;
- ``direct``: the model on the LR crop that the annotated landmarks cut from the scene. On a
  real photo the face is not on the template's grid, so this crop is resampled at LR
  resolution; against ``aligned`` it shows the cost of that;
- ``annotated``: the pipeline (paste-back included) with the annotated landmarks, aligned to
  ``CELEBA_TEMPLATE``; against ``direct`` it shows the cost of the paste-back;
- ``detected``: the full pipeline, with YuNet (D31); a missed face stays bicubic. Against
  ``annotated`` it shows the cost of the detection.

``aligned`` and ``direct`` are scored on the model's own HR crop. The frames (``bicubic``,
``annotated``, ``detected``) are scored on the 128 px crop that the **annotated** landmarks align
to ``CELEBA_TEMPLATE`` in them: the face where it really is, whatever the pipeline aligned to.
That crop resamples the frame once more (bicubic, about 1:1). Every variant is compared with the
dataset's HR crop of the same image.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from facehall.align.template import CELEBA_TEMPLATE
from facehall.align.warp import alignment_matrix, warp
from facehall.eval.detection import (
    Row,
    SceneDegradation,
    degrade,
    landmark_error,
    load_rgb,
    make_scene,
)
from facehall.eval.metrics import psnr_y, ssim_y
from facehall.pipeline import Enhancer, lr_crop, to_frame, upscale

VARIANTS = ("bicubic", "aligned", "direct", "annotated", "detected")
Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Sample:
    scene: npt.NDArray[np.uint8]  # LR scene, RGB
    truth: npt.NDArray[np.float64]  # (5, 2) annotated landmarks in the scene
    hr: npt.NDArray[np.uint8]  # the dataset's HR crop (hr_size, RGB)
    lr: npt.NDArray[np.uint8]  # LR crop on the template's grid (area downscale of hr, degraded)


def samples(
    rows: Sequence[Row],
    scale: int,
    hr_size: int,
    degradation: SceneDegradation | None,
    seed: int,
) -> Iterator[Sample]:
    eyes = float(CELEBA_TEMPLATE[1, 0] - CELEBA_TEMPLATE[0, 0]) * hr_size / scale
    lr_size = hr_size // scale
    rng = np.random.default_rng(seed)
    crop_rng = np.random.default_rng([seed, 1])  # keeps the scenes independent of the crops
    for path, points in rows:
        image = load_rgb(path)
        scene, truth = make_scene(image, points, eyes, degradation=degradation, rng=rng)
        matrix = alignment_matrix(points, CELEBA_TEMPLATE, hr_size)
        hr = np.asarray(warp(image, matrix, hr_size, border="reflect"), dtype=np.uint8)
        lr = np.asarray(cv2.resize(hr, (lr_size, lr_size), interpolation=cv2.INTER_AREA), np.uint8)
        if degradation is not None:
            lr = degrade(lr, degradation, crop_rng)
        yield Sample(scene, truth, hr, lr)


def face_crop(
    frame: npt.NDArray[np.uint8], truth: npt.ArrayLike, scale: int, size: int
) -> npt.NDArray[np.uint8]:
    """The crop of ``frame`` (×scale) that the annotated landmarks align to the CelebA template."""
    matrix = alignment_matrix(to_frame(truth, scale), CELEBA_TEMPLATE, size)
    return np.asarray(warp(frame, matrix, size, border="reflect"), dtype=np.uint8)


def to_tensor(images: Sequence[npt.NDArray[np.uint8]], device: torch.device | str) -> torch.Tensor:
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255.0).to(device)


def evaluate(
    enhancer: Enhancer,
    data: Iterator[Sample],
    metrics: dict[str, Metric] | None = None,
    *,
    batch: int = 64,
) -> dict[str, Any]:
    """Mean metrics per variant; how often the detector found the face (landmark NME below
    0.5, as in D31) and how often that face was restored."""
    scorers: dict[str, Metric] = {"psnr_y": psnr_y, "ssim_y": ssim_y, **(metrics or {})}
    values: dict[str, dict[str, list[float]]] = {v: {k: [] for k in scorers} for v in VARIANTS}
    pending: dict[str, list[npt.NDArray[np.uint8]]] = {v: [] for v in VARIANTS}
    references: list[npt.NDArray[np.uint8]] = []
    found = restored = scenes = 0
    scale, size = enhancer.scale, enhancer.hr_size

    def flush() -> None:
        if not references:
            return
        hr = to_tensor(references, enhancer.device)
        for variant, crops in pending.items():
            sr = to_tensor(crops, enhancer.device)
            for name, scorer in scorers.items():
                values[variant][name] += scorer(sr, hr).double().cpu().tolist()
            crops.clear()
        references.clear()

    for sample in data:
        scenes += 1
        references.append(sample.hr)
        pending["bicubic"].append(
            face_crop(upscale(sample.scene, scale), sample.truth, scale, size)
        )
        annotated = enhancer.enhance(sample.scene, [sample.truth]).image
        pending["annotated"].append(face_crop(annotated, sample.truth, scale, size))
        result = enhancer.enhance(sample.scene)
        # the D31 rule: a detection is the face when its landmark NME is below 0.5
        hit = [f for f in result.faces if landmark_error(f.landmarks, sample.truth) < 0.5]
        found += bool(hit)
        restored += any(f.restored for f in hit)
        pending["detected"].append(face_crop(result.image, sample.truth, scale, size))
        crop, _ = lr_crop(sample.scene, sample.truth, CELEBA_TEMPLATE, enhancer.lr_size)
        for variant, restored_crop in zip(
            ("aligned", "direct"), enhancer.restore([sample.lr, crop]), strict=True
        ):
            pending[variant].append(np.clip(np.rint(restored_crop), 0, 255).astype(np.uint8))
        if len(references) >= batch:
            flush()
    flush()
    if not scenes:
        raise ValueError("no scenes to evaluate")
    return {
        "scenes": scenes,
        "detected": found / scenes,
        "restored": restored / scenes,
        "variants": {
            variant: {name: float(np.mean(v)) for name, v in table.items()}
            for variant, table in values.items()
        },
    }
