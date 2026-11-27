"""The web interface of facehall (``facehall ui``, D41): a page and its local JSON API.

The page (``static/``) lets the user choose photos, videos or folders on this computer, run
the enhancement and look at the results, which are written to an output folder. The service
reads the chosen files from the disk itself, so nothing is uploaded.

**Access** (local use only, D13)
- It binds to 127.0.0.1. Every ``/api`` request needs the session token, which
  ``facehall ui`` prints and opens in the browser as ``http://127.0.0.1:8100/#token=…``.
  The fragment is never sent to a server; the page keeps it for the tab
  (``sessionStorage``) and sends it in the ``X-Facehall-Token`` header, or as ``token`` in
  the address of an image or video.
- A request whose ``Host`` is not this machine's loopback address is refused, so a web page
  that rebinds its domain to 127.0.0.1 cannot reach the API; without the token it could not
  use it anyway.
- The page has a strict content security policy: its own scripts and styles only.

**Privacy** (``CLAUDE.md`` §11): the log holds the method, path, status and time of each
request, never its query string (file paths) or pixels. Nothing but the outputs the user
asked for is written; previews are made in memory.
"""

import logging
import secrets
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from facehall import __version__, marking
from facehall.pipeline import UiConfig
from facehall.serve.app import TELEMETRY_OFF
from facehall.ui import media
from facehall.ui.batches import Batches, ImageFormat, Options, VideoFormat

logger = logging.getLogger("facehall.ui")

STATIC = resources.files("facehall.ui") / "static"
ASSETS = {
    "app.js": "text/javascript",
    "app.css": "text/css",
    "i18n.json": "application/json",
    "icon.svg": "image/svg+xml",
}
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
PAGE_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; media-src 'self'; "
    "style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}
MAX_PREVIEW_PIXELS = 4096 * 4096
MAX_CROP = 1024


@dataclass(frozen=True)
class Settings:
    """What the page needs to know about this installation."""

    home: Path
    output: Path  # the default output folder
    scales: Mapping[int, bool]  # whether a model of each scale is configured
    backend: str
    device: str
    ffmpeg: bool


class BatchRequest(BaseModel):
    inputs: list[str] = Field(min_length=1)
    output: str
    scale: Literal[4, 8] = 4
    image_format: ImageFormat = "png"
    video_format: VideoFormat = "mp4"
    quality: int = Field(95, ge=1, le=100)
    watermark: bool = False
    overwrite: bool = False
    subfolder: bool = True


class OpenRequest(BaseModel):
    path: str


def host_allowed(header: str, bound: str) -> bool:
    name = header.rsplit(":", 1)[0] if not header.endswith("]") else header
    return name in LOOPBACK or name == bound


def opener() -> list[str]:
    """The command that opens a folder in the desktop's file manager."""
    if sys.platform == "darwin":
        return ["open"]
    if sys.platform == "win32":
        return ["explorer"]
    return ["xdg-open"]


def create_app(
    batches: Batches,
    settings: Settings,
    config: UiConfig,
    *,
    token: str,
    on_start: Callable[[], None] | None = None,
) -> FastAPI:
    """The page and its API over a queue of batches."""
    if len(token) < 16:
        raise ValueError("the session token is too short")
    pictures = media.Pictures(config.cache_mb * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        logger.info(marking.NOTICE)
        if on_start is not None:
            on_start()
        yield
        batches.close()

    app = FastAPI(
        title="facehall ui",
        version=__version__,
        lifespan=lifespan,
        telemetry=TELEMETRY_OFF,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:  # no query string in the log: it holds file paths
        started = time.perf_counter()
        if not host_allowed(request.headers.get("host", ""), config.host):
            response: Response = JSONResponse({"detail": "unknown host"}, status_code=400)
        elif request.url.path.startswith("/api/") and not secrets.compare_digest(
            request.headers.get("x-facehall-token") or request.query_params.get("token") or "",
            token,
        ):
            response = JSONResponse(
                {"detail": "open the address that facehall ui printed"}, status_code=401
            )
        else:
            response = await call_next(request)
        logger.info(
            "%s %s %d %.2f s",
            request.method,
            request.url.path,
            response.status_code,
            time.perf_counter() - started,
        )
        return response

    def asset(name: str) -> Response:
        data = (STATIC / name).read_bytes()
        return Response(data, media_type=ASSETS.get(name, "text/html"), headers=PAGE_HEADERS)

    @app.get("/", include_in_schema=False)
    def page() -> Response:
        return asset("index.html")

    @app.get("/static/{name}", include_in_schema=False)
    def static(name: str) -> Response:
        if name not in ASSETS:
            raise HTTPException(404, "not found")
        return asset(name)

    # --- information and files -------------------------------------------------------------

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {
            "version": __version__,
            "notice": marking.NOTICE,
            "home": str(settings.home),
            "output": str(settings.output),
            "places": media.places(settings.home),
            "scales": {str(s): ok for s, ok in sorted(settings.scales.items())},
            "backend": settings.backend,
            "device": settings.device,
            "ffmpeg": settings.ffmpeg,
            "image_formats": ["png", "jpg", "webp"],
            "video_formats": ["mp4", "mkv", "mov"],
            "thumbnail_size": config.thumbnail_size,
            "preview_size": config.preview_size,
        }

    @app.get("/api/folder")
    def folder(path: str = "", hidden: bool = False) -> dict[str, Any]:
        try:
            return media.listing(Path(path) if path else settings.home, hidden=hidden)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from None
        except PermissionError:
            raise HTTPException(403, f"{path}: no permission to read this folder") from None
        except (OSError, ValueError) as error:
            raise HTTPException(400, str(error)) from None

    def picture(path: str) -> tuple[Path, media.Kind]:
        try:
            return media.media_file(path)
        except media.NotMedia as error:
            raise HTTPException(415, str(error)) from None
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None

    def made(make: Callable[[], bytes], media_type: str) -> Response:
        try:
            data = make()
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from None
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise HTTPException(422, f"cannot show this file: {error}") from None
        return Response(data, media_type=media_type, headers={"Cache-Control": "no-store"})

    @app.get("/api/thumbnail")
    def thumbnail(path: str) -> Response:
        file, found = picture(path)
        return made(lambda: pictures.thumbnail(file, found, config.thumbnail_size), "image/jpeg")

    @app.get("/api/preview")
    def preview(
        path: str,
        width: Annotated[int, Query(ge=1)],
        height: Annotated[int, Query(ge=1)],
        interpolation: media.Interpolation = "bicubic",
    ) -> Response:
        if width * height > MAX_PREVIEW_PIXELS:
            raise HTTPException(400, "the preview is too large")
        file, found = picture(path)
        return made(
            lambda: pictures.preview(file, found, width, height, interpolation), "image/jpeg"
        )

    @app.get("/api/crop")
    def crop(
        path: str,
        x: float,
        y: float,
        w: Annotated[float, Query(gt=0)],
        h: Annotated[float, Query(gt=0)],
        size: Annotated[int, Query(ge=8, le=MAX_CROP)] = 256,
        interpolation: media.Interpolation = "bicubic",
    ) -> Response:
        file, found = picture(path)
        return made(
            lambda: pictures.crop(file, found, (x, y, w, h), size, interpolation), "image/png"
        )

    @app.get("/api/file")
    def file(path: str, download: bool = False) -> Response:
        found, _ = picture(path)
        return FileResponse(
            found,
            media_type=media.MEDIA_TYPES.get(found.suffix.lower(), "application/octet-stream"),
            filename=found.name if download else None,
            content_disposition_type="attachment" if download else "inline",
            headers={"Cache-Control": "no-store"},
        )

    # --- batches ---------------------------------------------------------------------------

    @app.post("/api/batches", status_code=201)
    def submit(request: BatchRequest) -> dict[str, Any]:
        options = Options(
            inputs=tuple(Path(p).expanduser() for p in request.inputs),
            output=Path(request.output).expanduser(),
            scale=request.scale,
            image_format=request.image_format,
            video_format=request.video_format,
            quality=request.quality,
            watermark=request.watermark,
            overwrite=request.overwrite,
            subfolder=request.subfolder,
        )
        try:
            batch = batches.submit(options)
        except (OSError, ValueError) as error:
            raise HTTPException(400, str(error)) from None
        return batches.get(batch.id) or {}

    @app.get("/api/batches")
    def summaries() -> list[dict[str, Any]]:
        return batches.summaries()

    @app.get("/api/batches/{batch_id}")
    def batch(batch_id: int) -> dict[str, Any]:
        found = batches.get(batch_id)
        if found is None:
            raise HTTPException(404, f"no batch {batch_id}")
        return found

    @app.post("/api/batches/{batch_id}/cancel")
    def cancel(batch_id: int) -> dict[str, Any]:
        if not batches.cancel(batch_id):
            raise HTTPException(404, f"no batch {batch_id}")
        return batches.get(batch_id) or {}

    @app.post("/api/open", status_code=204)
    def open_folder(request: OpenRequest) -> Response:
        """Show an output folder of a batch in the desktop's file manager."""
        target = Path(request.path).expanduser().resolve()
        if target not in batches.folders() or not target.is_dir():
            raise HTTPException(403, "only the output folders of this session can be opened")
        try:
            subprocess.Popen(  # a fixed program and a checked output folder
                [*opener(), str(target)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            raise HTTPException(501, f"cannot open the file manager: {error}") from None
        return Response(status_code=204)

    app.state.batches = batches  # for the tests
    return app
