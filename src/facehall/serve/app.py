"""The REST API of facehall (``facehall serve``, Phase 5, PR 5.3; D36).

    facehall serve                       # http://127.0.0.1:8000, the x4 and x8 ONNX models
    curl --data-binary @photo.jpg "http://127.0.0.1:8000/enhance?scale=4" -o enhanced.png

**Endpoints**
- ``POST /enhance``: the request body is one photo, the file's bytes themselves (PNG, JPEG,
  WebP, BMP or TIFF; not a form). Query parameters: ``scale`` (4 or 8), ``format`` (png, jpeg,
  webp), ``quality``, ``watermark`` and ``box`` (a face the detector misses, repeatable). The
  answer is the enhanced photo, marked like those of ``facehall enhance`` (``facehall.marking``),
  with the notice and the face counts in its headers.
- ``GET /health`` and ``GET /version`` (the models and the limits).
- ``GET /docs``: FastAPI's page for trying the API by hand (D37).

**Privacy** (``CLAUDE.md`` §11)
- The upload is read into memory, never written to disk, and dropped with the answer.
- The log holds the method, path, status and time of each request: no pixels, file names,
  query parameters (face boxes) or face positions. uvicorn's access log is off.
- FastAPI's own telemetry (OpenTelemetry spans, metrics and logs) is off.

**Limits** (``serve`` in the pipeline config, ``facehall.pipeline.ServeConfig``): the upload
size, the output pixels (checked from the image header, before decoding), one enhancement at a
time with a bounded queue, and a time per request (waiting included).

There is no authentication: the service binds to 127.0.0.1 by default (D13: local use only).
"""

import asyncio
import io
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from importlib import metadata
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.telemetry import TelemetryConfig
from PIL import Image, UnidentifiedImageError

from facehall import __version__, marking
from facehall.enhance import decode_image, marked, numbers, parameters, summary, to_box
from facehall.pipeline import Enhancer, ServeConfig

logger = logging.getLogger("facehall.serve")

ACCEPTED = ("PNG", "JPEG", "MPO", "WEBP", "BMP", "TIFF")  # PIL formats; MPO: camera JPEGs
MEDIA_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
TELEMETRY_OFF: TelemetryConfig = {
    "tracing": False,
    "metrics": False,
    "logs": False,
    "operation_spans": False,
    "auto_configure": False,
}
BODY = {
    "requestBody": {
        "required": True,
        "description": "The photo: the file's bytes themselves, not a form "
        "(curl --data-binary @photo.jpg).",
        "content": {
            media: {"schema": {"type": "string", "format": "binary"}}
            for media in ("image/jpeg", "image/png", "image/webp", "application/octet-stream")
        },
    }
}


class Busy(Exception):
    """The service cannot take or finish the request in time (HTTP 503)."""


class Expired(Exception):
    """Work whose request has already been answered as timed out."""


class Worker:
    """Runs one enhancement at a time in its own thread, with a bounded queue.

    ``pending`` counts the running and the waiting requests. A request beyond ``1 + queue`` is
    refused at once, and one not answered within ``timeout`` seconds gets a 503. Its work, once
    started, still runs to the end (a thread cannot be stopped), so the next one waits for it;
    work that had not started by then is skipped.
    """

    def __init__(self, queue: int, timeout: float) -> None:
        self.queue, self.timeout = queue, timeout
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="facehall-enhance")
        self.pending = 0  # changed on the event loop's thread only

    async def run[T](self, work: Callable[[], T]) -> T:
        if self.pending >= 1 + self.queue:
            raise Busy(f"the service is busy ({self.pending} requests); try again later")
        deadline = time.monotonic() + self.timeout

        def guarded() -> T:
            if time.monotonic() > deadline:
                raise Expired
            return work()

        future = asyncio.get_running_loop().run_in_executor(self.executor, guarded)
        self.pending += 1
        future.add_done_callback(self.finished)
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.timeout)
        except TimeoutError:
            raise Busy(f"no answer within {self.timeout:g} s; try again later") from None

    def finished(self, future: "asyncio.Future[Any]") -> None:
        self.pending -= 1
        if not future.cancelled():
            future.exception()  # retrieved: the answer may have been given up on

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def warm_up(enhancer: Enhancer) -> None:
    """Start the detector, and run the model once per batch size on CUDA (where ONNX Runtime
    picks the cuDNN algorithms of each new size, about 20 ms each, PR 5.2), once on CPU."""
    enhancer.enhance(np.full((64, 64, 3), 128, dtype=np.uint8))
    size = enhancer.lr_size
    for n in range(1, enhancer.config.batch_size + 1) if enhancer.device == "cuda" else (1,):
        enhancer.restorer(np.zeros((n, 3, size, size), dtype=np.float32))


def check_image(data: bytes, scale: int, max_output_pixels: int) -> None:
    """Refuse unsupported formats and too large outputs from the image header alone."""
    try:
        with Image.open(io.BytesIO(data)) as picture:
            kind, (width, height) = picture.format, picture.size
    except Image.DecompressionBombError:
        raise HTTPException(413, "the image has too many pixels") from None
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            415, f"not an image in a supported format ({', '.join(ACCEPTED)})"
        ) from None
    if kind not in ACCEPTED:
        raise HTTPException(415, f"{kind} images are not supported; use one of {ACCEPTED}")
    pixels = width * height * scale**2
    if pixels > max_output_pixels:
        raise HTTPException(
            413,
            f"a {width} x {height} photo at x{scale} gives {pixels:,} output pixels, more than "
            f"the limit of {max_output_pixels:,}",
        )


async def read_body(request: Request, limit: int) -> bytes:
    """The request body, refused as soon as it exceeds ``limit`` bytes."""
    if request.headers.get("content-type", "").startswith("multipart/"):
        raise HTTPException(
            415, "send the photo's bytes as the body (curl --data-binary @photo.jpg), not a form"
        )
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, f"the upload is larger than {limit:,} bytes")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(413, f"the upload is larger than {limit:,} bytes")
    if not body:
        raise HTTPException(400, "the request body must be the photo")
    return bytes(body)


def runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def create_app(enhancers: Mapping[int, Enhancer], config: ServeConfig) -> FastAPI:
    """The API over loaded enhancers, one per scale."""
    if not enhancers:
        raise ValueError("no models to serve")
    for scale, enhancer in enhancers.items():
        if enhancer.scale != scale:
            raise ValueError(f"the model for x{scale} is a x{enhancer.scale} model")
    worker = Worker(config.queue, config.timeout)
    settings = {scale: parameters(e.config) for scale, e in enhancers.items()}
    upload_limit = int(config.max_upload_mb * 1024 * 1024)
    default_scale = min(enhancers)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if config.warmup:
            started = time.perf_counter()
            for enhancer in enhancers.values():
                await worker.run(partial(warm_up, enhancer))
            logger.info("models warmed up in %.1f s", time.perf_counter() - started)
        logger.info(marking.NOTICE)
        yield
        worker.close()

    app = FastAPI(
        title="facehall",
        version=__version__,
        description="Face super-resolution for local use. "
        + marking.NOTICE
        + " The photo is processed in memory and not stored.",
        lifespan=lifespan,
        telemetry=TELEMETRY_OFF,
    )

    @app.middleware("http")
    async def log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:  # no query string: it may hold face boxes
        started = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s %d %.2f s",
            request.method,
            request.url.path,
            response.status_code,
            time.perf_counter() - started,
        )
        return response

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "scales": sorted(enhancers), "pending": worker.pending}

    @app.get("/version")
    async def version() -> dict[str, Any]:
        return {
            "facehall": __version__,
            "runtimes": runtime_versions(),
            "models": {
                str(scale): {**e.info, "device": e.device} for scale, e in sorted(enhancers.items())
            },
            "limits": {
                "max_upload_bytes": upload_limit,
                "max_output_pixels": config.max_output_pixels,
                "queue": config.queue,
                "timeout_s": config.timeout,
            },
            "notice": marking.NOTICE,
        }

    @app.post(
        "/enhance",
        response_class=Response,
        responses={
            200: {
                "description": "The enhanced photo, marked as AI-enhanced.",
                "content": {media: {} for media in MEDIA_TYPES.values()},
            },
            400: {"description": "Bad parameters, or a photo that cannot be decoded."},
            413: {"description": "The upload or its output is too large."},
            415: {"description": "Not a photo in a supported format, or a form."},
            503: {"description": "Busy: the queue is full or the answer took too long."},
        },
        openapi_extra=BODY,
    )
    async def enhance(
        request: Request,
        scale: Annotated[
            int | None, Query(description="4 or 8 (default: the smallest served)")
        ] = None,
        output_format: Annotated[
            Literal["png", "jpeg", "webp"], Query(alias="format", description="output format")
        ] = "png",
        quality: Annotated[int, Query(ge=1, le=100, description="JPEG and WebP quality")] = 95,
        watermark: Annotated[bool, Query(description="a visible 'AI-enhanced' mark")] = False,
        box: Annotated[
            list[str] | None,
            Query(
                description="LEFT,TOP,WIDTH,HEIGHT of a face the detector misses: tight, like a "
                "detector's, in photo pixels (repeatable); without it, faces are detected"
            ),
        ] = None,
    ) -> Response:
        chosen = default_scale if scale is None else scale
        if chosen not in enhancers:
            raise HTTPException(
                400, f"x{chosen} is not served; this service has {sorted(enhancers)}"
            )
        try:
            boxes = [to_box(numbers(b, 4, "box")) for b in box] if box else None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        data = await read_body(request, upload_limit)
        check_image(data, chosen, config.max_output_pixels)
        enhancer = enhancers[chosen]

        def work() -> tuple[bytes, str]:
            image, digest = decode_image(data)
            result = enhancer.enhance(image, boxes=boxes)
            out, rec = marked(
                enhancer,
                image,
                result,
                digest=digest,
                faces_from="boxes" if boxes else "detector",
                visible_mark=watermark,
                settings=settings[chosen],
            )
            encoded = marking.encode(out, rec, output_format.upper(), quality=quality)
            return encoded, summary(result.faces)

        try:
            encoded, faces = await worker.run(work)
        except Busy as error:
            raise HTTPException(503, str(error), headers={"Retry-After": "10"}) from None
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            raise HTTPException(400, str(error)) from None
        return Response(
            content=encoded,
            media_type=MEDIA_TYPES[output_format],
            headers={
                "X-AI-Enhanced": "true",
                "X-Facehall-Notice": marking.NOTICE,
                "X-Facehall-Model": f"{enhancer.info.get('name', '')} (x{chosen})",
                "X-Facehall-Faces": faces,
                "Cache-Control": "no-store",
            },
        )

    app.state.worker = worker  # for /health and the tests
    return app
