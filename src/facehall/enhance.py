"""``facehall enhance``: restore the faces of photos and folders of photos (PR 4.4).

Each input image goes through ``facehall.pipeline.Enhancer`` and is written with its marks
(``facehall.marking``).

- **Inputs:** image files or folders (searched recursively).
- **Outputs:** one output file for one input file, or an output folder that mirrors the inputs.
  Outputs are PNG by default; JPEG and WebP are lossy.
- **Overwriting:** existing outputs are not overwritten unless asked, and an output never
  replaces its input.
- **Image orientation:** the EXIF orientation is applied, and alpha is dropped.
- **Messages:** only file names and face counts are printed, never image content.
"""

import hashlib
import io
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

from facehall import marking
from facehall.pipeline import Box, Enhancer, PipelineConfig

INPUT_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Job:
    source: Path
    target: Path


def find_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES
    )


def plan(
    inputs: Sequence[Path], output: Path, *, suffix: str = ".png", overwrite: bool = False
) -> list[Job]:
    """Input and output path pairs.

    One input file and an output with an image suffix give one output file. Otherwise
    ``output`` is a folder: an input file ``a/b.jpg`` goes to ``output/b<suffix>``, and the
    images of an input folder go to the same relative paths under ``output``.
    """
    marking.output_format(Path("x" + suffix))
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"{path}: no such file or folder")
    if len(inputs) == 1 and inputs[0].is_file() and output.suffix.lower() in marking.FORMATS:
        jobs = [Job(inputs[0], output)]
    else:
        if output.is_file():
            raise ValueError(f"{output}: several inputs or a folder need an output folder")
        jobs = []
        for path in inputs:
            if path.is_dir():
                jobs += [
                    Job(image, (output / image.relative_to(path)).with_suffix(suffix))
                    for image in find_images(path)
                ]
            else:
                jobs.append(Job(path, output / (path.stem + suffix)))
    if not jobs:
        raise ValueError("no input images found")
    targets = [j.target.resolve() for j in jobs]
    if len(set(targets)) < len(targets):
        raise ValueError("two inputs would be written to the same output file; rename one")
    sources = {j.source.resolve() for j in jobs}
    for job, target in zip(jobs, targets, strict=True):
        if target in sources:
            raise ValueError(f"{job.target}: an output would replace an input")
        if target.exists() and not overwrite:
            raise FileExistsError(f"{job.target} exists; pass --overwrite to replace it")
    return jobs


def load_image(path: Path) -> tuple[npt.NDArray[np.uint8], str]:
    """RGB uint8 pixels (EXIF orientation applied) and the sha256 of the file."""
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as picture:
        rgb = ImageOps.exif_transpose(picture).convert("RGB")
    return np.asarray(rgb, dtype=np.uint8), hashlib.sha256(data).hexdigest()


def parameters(config: PipelineConfig) -> dict[str, Any]:
    """The pipeline settings for the record, without local paths."""
    values = asdict(config)
    values.pop("models")
    weights = values["detector"].pop("weights")
    values["detector"]["weights_file"] = Path(weights).name if weights else None
    return values


def summary(faces: Sequence[Any]) -> str:
    """Face counts, with the faces left alone counted by kind ("too large for the model", …)."""
    restored = sum(f.restored for f in faces)
    text = f"{len(faces)} face{'s' if len(faces) != 1 else ''}, {restored} restored"
    kinds = Counter(str(f.reason).split(":")[0] for f in faces if not f.restored)
    return text + "".join(f", {n} {kind}" for kind, n in sorted(kinds.items()))


def enhance_files(
    enhancer: Enhancer,
    jobs: Sequence[Job],
    *,
    landmarks: Sequence[Sequence[float]] | None = None,
    boxes: Sequence[Box] | None = None,
    visible_mark: bool = False,
    quality: int = 95,
    report: Callable[[str], None] = print,
) -> list[tuple[Path, str]]:
    """Enhance and write every job; returns the failures (input, message)."""
    if (landmarks is not None or boxes is not None) and len(jobs) != 1:
        raise ValueError("landmarks and face boxes are for one input image only")
    faces_from = "detector"
    if landmarks is not None:
        faces_from = "landmarks"
    elif boxes is not None:
        faces_from = "boxes"
    settings = parameters(enhancer.config)
    failures: list[tuple[Path, str]] = []
    for job in jobs:
        try:
            image, digest = load_image(job.source)
            result = enhancer.enhance(image, landmarks=landmarks, boxes=boxes)
            out = marking.watermark(result.image) if visible_mark else result.image
            rec = marking.record(
                [f.to_dict() for f in result.faces],
                model=enhancer.info,
                parameters=settings,
                input_sha256=digest,
                input_size=(image.shape[1], image.shape[0]),
                output_size=(out.shape[1], out.shape[0]),
                faces_from=faces_from,
                watermark=visible_mark,
            )
            job.target.parent.mkdir(parents=True, exist_ok=True)
            partial = job.target.with_name(job.target.name + ".partial" + job.target.suffix)
            marking.save(partial, out, rec, quality=quality)
            partial.replace(job.target)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            failures.append((job.source, str(error)))
            report(f"{job.source}: failed: {error}")
            continue
        report(f"{job.source} -> {job.target}: {summary(result.faces)}")
    return failures


def numbers(text: str, count: int, what: str) -> list[float]:
    """``count`` comma-separated numbers (a command-line value)."""
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != count:
        raise ValueError(f"{what} needs {count} comma-separated numbers, got {text!r}")
    return values


def to_box(values: Sequence[float]) -> Box:
    left, top, width, height = values
    return left, top, width, height


def select_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_enhance(
    inputs: Sequence[Path],
    output: Path,
    *,
    pipeline: Path,
    overrides: Sequence[str] = (),
    checkpoint: Path | None = None,
    scale: int | None = None,
    landmarks: Sequence[str] = (),
    boxes: Sequence[str] = (),
    visible_mark: bool = False,
    suffix: str = ".png",
    quality: int = 95,
    overwrite: bool = False,
    device: str = "auto",
    report: Callable[[str], None] = print,
) -> int:
    """The ``facehall enhance`` command; returns the exit code (1 if any image failed)."""
    from facehall.config import load_config

    config = load_config(pipeline, PipelineConfig, overrides)
    if landmarks and boxes:
        raise ValueError("give --landmarks or --box, not both")
    points = [numbers(t, 10, "--landmarks") for t in landmarks] or None
    face_boxes = [to_box(numbers(t, 4, "--box")) for t in boxes] or None
    jobs = plan(inputs, output, suffix=suffix, overwrite=overwrite)
    path = checkpoint if checkpoint is not None else Path(config.models.checkpoint(scale or 4))
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no checkpoint; pass --ckpt, or set the models in {pipeline} "
            "(or FACEHALL_RUNS, the folder of the reference runs)"
        )
    enhancer = Enhancer.from_checkpoint(path, config, select_device(device))
    if scale is not None and enhancer.scale != scale:
        raise ValueError(f"{path} is a x{enhancer.scale} model, not x{scale}")
    report(f"model {enhancer.info['name']} (x{enhancer.scale}), {len(jobs)} image(s)")
    failures = enhance_files(
        enhancer,
        jobs,
        landmarks=points,
        boxes=face_boxes,
        visible_mark=visible_mark,
        quality=quality,
        report=report,
    )
    report(marking.NOTICE)
    if failures:
        report(f"{len(failures)} of {len(jobs)} image(s) failed")
    return 1 if failures else 0
