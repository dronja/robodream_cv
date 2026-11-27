"""Enhancements started from the web interface: a queue of batches, one worker thread.

A batch is one press of "Enhance": its inputs (files and folders), one scale, one output
folder and the output options. Batches run one at a time in submission order, so a single
model is busy at a time. The models are loaded when first needed and then kept.

State lives in memory only: the list of batches is gone when the interface stops, while the
written outputs stay where they are.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from facehall import marking, video
from facehall.enhance import Job, enhance_one, parameters, plan
from facehall.pipeline import Enhancer

logger = logging.getLogger("facehall.ui")

Status = Literal["queued", "loading", "running", "done", "failed", "cancelled"]
ImageFormat = Literal["png", "jpg", "webp"]
VideoFormat = Literal["mp4", "mkv", "mov"]


class Cancelled(Exception):
    """The user cancelled the batch; raised inside a video to stop it between frames."""


@dataclass(frozen=True)
class Options:
    """What the user chose for a batch."""

    inputs: tuple[Path, ...]
    output: Path  # the output folder
    scale: int = 4
    image_format: ImageFormat = "png"
    video_format: VideoFormat = "mp4"
    quality: int = 95
    watermark: bool = False
    overwrite: bool = False
    subfolder: bool = True  # write into a new folder <date>_<time>_x<scale> under ``output``


@dataclass
class Item:
    """One input file and its output."""

    source: Path
    target: Path
    kind: str  # "image" or "video"
    status: Status = "queued"
    message: str | None = None  # why it failed
    progress: float | None = None  # a video's fraction of frames done
    seconds: float | None = None
    record: dict[str, Any] | None = None  # the marks of the written output

    def to_dict(self) -> dict[str, Any]:
        rec = self.record or {}
        return {
            "source": str(self.source),
            "target": str(self.target),
            "name": self.source.name,
            "kind": self.kind,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "seconds": self.seconds,
            "input": rec.get("input"),
            "output": rec.get("output"),
            "faces": rec.get("faces"),
            "video": rec.get("video"),
        }


@dataclass
class Batch:
    id: int
    options: Options
    folder: Path  # where the outputs go
    items: list[Item]
    created: float = field(default_factory=time.time)
    status: Status = "queued"
    message: str | None = None  # why the whole batch failed (for example, no model)
    model: dict[str, Any] | None = None  # name, scale and sha256 of the model used
    cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self, *, items: bool = True) -> dict[str, Any]:
        counts = {s: 0 for s in ("queued", "running", "done", "failed", "cancelled")}
        progress = 0.0
        for item in self.items:
            counts[item.status if item.status in counts else "running"] += 1
            if item.status in ("done", "failed", "cancelled"):
                progress += 1.0
            elif item.status == "running":
                progress += item.progress or 0.0
        data: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "created": self.created,
            "scale": self.options.scale,
            "folder": str(self.folder),
            "model": self.model,
            "counts": counts,
            "total": len(self.items),
            "progress": progress / len(self.items) if self.items else 1.0,
            "options": {
                "image_format": self.options.image_format,
                "video_format": self.options.video_format,
                "quality": self.options.quality,
                "watermark": self.options.watermark,
            },
        }
        if items:
            data["items"] = [item.to_dict() for item in self.items]
        return data


def run_folder(options: Options, now: datetime | None = None) -> Path:
    """The output folder of a batch: ``output`` itself, or a new subfolder of it."""
    if options.output.suffix.lower() in (*marking.FORMATS, *video.OUTPUT_SUFFIXES):
        raise ValueError(f"{options.output}: choose a folder for the outputs, not a file name")
    if not options.subfolder:
        return options.output
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    folder = options.output / f"{stamp}_x{options.scale}"
    n = 2
    while folder.exists():
        folder = options.output / f"{stamp}_x{options.scale}-{n}"
        n += 1
    return folder


class Batches:
    """The queue of batches and the worker thread that runs them.

    ``load(scale)`` returns the enhancer of a scale; it is called in the worker thread, once
    per scale. ``available(scale)`` says whether a model of that scale is configured.
    """

    def __init__(
        self, load: Callable[[int], Enhancer], available: Callable[[int], bool] | None = None
    ) -> None:
        self.load = load
        self.available = available or (lambda _: True)
        self.enhancers: dict[int, Enhancer] = {}
        self.batches: dict[int, Batch] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue[Batch | None] = queue.Queue()
        self.worker = threading.Thread(target=self.work, name="facehall-ui", daemon=True)
        self.worker.start()

    # --- the API side ----------------------------------------------------------------------

    def submit(self, options: Options) -> Batch:
        """Plan a batch and queue it; an error says what is wrong with the options."""
        if options.scale not in (4, 8):
            raise ValueError(f"the scale must be 4 or 8, not {options.scale}")
        if not self.available(options.scale):
            raise ValueError(f"no x{options.scale} model is configured (see docs/USAGE.md §3)")
        if not 1 <= options.quality <= 100:
            raise ValueError(f"the quality must be in [1, 100], not {options.quality}")
        if not options.inputs:
            raise ValueError("choose photos, videos or a folder first")
        for path in (*options.inputs, options.output):
            if not path.is_absolute():
                raise ValueError(f"{path}: give an absolute path")
        if options.output.is_file():
            raise ValueError(f"{options.output}: choose a folder for the outputs, not a file")
        folder = run_folder(options)
        jobs = plan(
            options.inputs,
            folder,
            suffix="." + options.image_format,
            video_suffix="." + options.video_format,
            overwrite=options.overwrite,
            exclude=options.output,
        )
        if any(video.is_video(j.source) for j in jobs):
            video.ffmpeg()  # an error now rather than for every video
        items = [
            Item(j.source, j.target, "video" if video.is_video(j.source) else "image") for j in jobs
        ]
        with self.lock:
            batch = Batch(len(self.batches) + 1, options, folder, items)
            self.batches[batch.id] = batch
        self.queue.put(batch)
        logger.info("batch %d: %d input(s) at x%d", batch.id, len(items), options.scale)
        return batch

    def get(self, batch_id: int) -> dict[str, Any] | None:
        with self.lock:
            batch = self.batches.get(batch_id)
            return None if batch is None else batch.to_dict()

    def summaries(self) -> list[dict[str, Any]]:
        """Every batch, newest first, without its items."""
        with self.lock:
            return [b.to_dict(items=False) for b in reversed(self.batches.values())]

    def cancel(self, batch_id: int) -> bool:
        with self.lock:
            batch = self.batches.get(batch_id)
            if batch is None:
                return False
            batch.cancel.set()
            return True

    def folders(self) -> set[Path]:
        """The output folders of the batches: the only folders the interface may open."""
        with self.lock:
            return {b.folder.resolve() for b in self.batches.values()}

    def outputs(self) -> set[Path]:
        with self.lock:
            return {
                i.target.resolve()
                for b in self.batches.values()
                for i in b.items
                if i.status == "done"
            }

    def busy(self) -> bool:
        with self.lock:
            return any(b.status in ("queued", "loading", "running") for b in self.batches.values())

    def close(self, timeout: float = 10.0) -> None:
        """Cancel everything and stop the worker (a running image still finishes)."""
        with self.lock:
            for batch in self.batches.values():
                batch.cancel.set()
        self.queue.put(None)
        self.worker.join(timeout)

    # --- the worker side -------------------------------------------------------------------

    def set(self, target: Batch | Item, **values: Any) -> None:
        with self.lock:
            for name, value in values.items():
                setattr(target, name, value)

    def enhancer(self, scale: int) -> Enhancer:
        if scale not in self.enhancers:
            enhancer = self.load(scale)
            if enhancer.scale != scale:
                raise ValueError(f"the configured x{scale} model is a x{enhancer.scale} model")
            self.enhancers[scale] = enhancer
        return self.enhancers[scale]

    def work(self) -> None:
        while (batch := self.queue.get()) is not None:
            try:
                self.run(batch)
            except Exception as error:  # keep the worker alive for the next batch
                logger.error("batch %d failed: %s", batch.id, type(error).__name__)
                self.set(batch, status="failed", message=f"unexpected error: {error}")

    def run(self, batch: Batch) -> None:
        if batch.cancel.is_set():
            for item in batch.items:
                self.set(item, status="cancelled")
            self.set(batch, status="cancelled")
            return
        self.set(batch, status="loading")
        try:
            enhancer = self.enhancer(batch.options.scale)
        except (OSError, ValueError, RuntimeError) as error:
            self.set(batch, status="failed", message=str(error))
            for item in batch.items:
                self.set(item, status="cancelled")
            return
        model = {k: enhancer.info.get(k) for k in ("name", "scale", "sha256", "onnx_sha256")}
        self.set(batch, status="running", model=model)
        settings = parameters(enhancer.config)
        for item in batch.items:
            if batch.cancel.is_set():
                self.set(item, status="cancelled")
                continue
            self.process(batch, item, enhancer, settings)
        done = sum(i.status == "done" for i in batch.items)
        self.set(batch, status="cancelled" if batch.cancel.is_set() else "done")
        logger.info("batch %d: %d of %d written", batch.id, done, len(batch.items))

    def process(
        self, batch: Batch, item: Item, enhancer: Enhancer, settings: dict[str, Any]
    ) -> None:
        self.set(item, status="running", progress=0.0 if item.kind == "video" else None)
        started = time.perf_counter()

        def progress(frames: int, total: int) -> None:
            if batch.cancel.is_set():
                raise Cancelled
            if total > 0:
                self.set(item, progress=min(frames / total, 0.99))

        try:
            written = enhance_one(
                enhancer,
                Job(item.source, item.target),
                settings=settings,
                visible_mark=batch.options.watermark,
                quality=batch.options.quality,
                report=lambda _: None,
                progress=progress,
            )
        except Cancelled:
            self.set(item, status="cancelled", progress=None)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            self.set(item, status="failed", message=str(error), progress=None)
        except Exception as error:  # a bug must not end the batch
            logger.error("batch %d: an input failed: %s", batch.id, type(error).__name__)
            self.set(item, status="failed", message=f"unexpected error: {error}", progress=None)
        else:
            self.set(
                item,
                status="done",
                progress=None,
                record=written.record,
                seconds=round(time.perf_counter() - started, 2),
            )
