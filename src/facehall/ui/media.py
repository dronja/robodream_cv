"""Local files for the web interface: folder listings, thumbnails, previews and face crops.

Everything is read from the disk on request and kept in memory only: nothing is written.
Images are decoded like the pipeline decodes its inputs (EXIF orientation applied), so the
face regions of a record match the pixels shown.
"""

import io
import os
import shutil
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps

from facehall import video
from facehall.enhance import INPUT_SUFFIXES, load_image

Kind = Literal["image", "video"]
Interpolation = Literal["bicubic", "nearest"]
RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "nearest": Image.Resampling.NEAREST}
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}


class NotMedia(ValueError):
    """A path that is not an image or video file (HTTP 415)."""


def kind(path: Path) -> Kind | None:
    """ "image" or "video" by the file suffix, or None."""
    suffix = path.suffix.lower()
    if suffix in INPUT_SUFFIXES:
        return "image"
    if suffix in video.INPUT_SUFFIXES:
        return "video"
    return None


def media_file(text: str) -> tuple[Path, Kind]:
    """An existing image or video file given by an absolute path."""
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{text}: give an absolute path")
    found = kind(path)
    if found is None:
        raise NotMedia(f"{path.name}: not an image or video file")
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such file")
    return path, found


# --- folders ------------------------------------------------------------------------------


def listing(folder: Path, *, hidden: bool = False) -> dict[str, Any]:
    """The subfolders, images and videos of ``folder``: folders first, then by name."""
    folder = folder.expanduser()
    if not folder.is_absolute():
        raise ValueError(f"{folder}: give an absolute path")
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder}: no such folder")
    entries: list[dict[str, Any]] = []
    with os.scandir(folder) as scan:
        for entry in scan:
            if not hidden and entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    entries.append({"name": entry.name, "path": entry.path, "kind": "dir"})
                    continue
                found = kind(Path(entry.name)) if entry.is_file() else None
                if found is not None:
                    stat = entry.stat()
                    entries.append(
                        {
                            "name": entry.name,
                            "path": entry.path,
                            "kind": found,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                        }
                    )
            except OSError:  # a broken link, or no permission to look
                continue
    entries.sort(key=lambda e: (e["kind"] != "dir", e["name"].casefold(), e["name"]))
    parent = folder.parent
    return {
        "path": str(folder),
        "parent": None if parent == folder else str(parent),
        "entries": entries,
    }


def user_dir(name: str, home: Path) -> Path | None:
    """An XDG user folder (PICTURES, VIDEOS, …), if it exists and is not the home folder.

    ``xdg-user-dir`` answers with the home folder for a folder that is not set up; then the
    usual English name under the home folder is tried.
    """
    command = shutil.which("xdg-user-dir")
    found: Path | None = None
    if command is not None:
        try:
            text = subprocess.run(
                [command, name], capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            text = ""
        found = Path(text) if text and Path(text) != home else None
    if found is None or not found.is_dir():
        found = home / ("Downloads" if name == "DOWNLOAD" else name.capitalize())
    return found if found.is_dir() and found != home else None


def places(home: Path) -> list[dict[str, str]]:
    """Shortcuts for the file dialog: the home folder and the XDG media folders."""
    found = [{"id": "home", "path": str(home)}]
    for name in ("PICTURES", "VIDEOS", "DESKTOP", "DOWNLOAD"):
        path = user_dir(name, home)
        if path is not None:
            found.append({"id": name.lower(), "path": str(path)})
    return found


def default_output(home: Path) -> Path:
    """Where results go unless the user picks another folder: <Pictures>/facehall."""
    return (user_dir("PICTURES", home) or home) / "facehall"


# --- pixels -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Key:
    path: str
    modified: int
    size: int


def key_of(path: Path) -> Key:
    stat = path.stat()
    return Key(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def first_frame(path: Path) -> Image.Image:
    capture, _ = video.open_video(path)
    try:
        frames = video.read_frames(capture)
        frame = next(frames, None)
    finally:
        capture.release()
    if frame is None:
        raise ValueError(f"{path.name}: no frames could be read")
    return Image.fromarray(frame)


def decode(path: Path, found: Kind) -> Image.Image:
    """RGB pixels of an image (EXIF orientation applied) or of a video's first frame."""
    if found == "video":
        return first_frame(path)
    pixels, _ = load_image(path)
    return Image.fromarray(pixels)


class Pictures:
    """Decoded images, the most recently used kept up to ``max_bytes``, and small encoded
    results (thumbnails, previews, crops) up to a tenth of that. Thread-safe."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.decoded: OrderedDict[Key, Image.Image] = OrderedDict()
        self.encoded: OrderedDict[tuple[Any, ...], bytes] = OrderedDict()
        self.lock = threading.Lock()

    def picture(self, path: Path, found: Kind) -> Image.Image:
        key = key_of(path)
        with self.lock:
            if key in self.decoded:
                self.decoded.move_to_end(key)
                return self.decoded[key]
        picture = decode(path, found)
        size = picture.width * picture.height * 3
        with self.lock:
            if size <= self.max_bytes:
                self.decoded[key] = picture
                while sum(p.width * p.height * 3 for p in self.decoded.values()) > self.max_bytes:
                    self.decoded.popitem(last=False)
        return picture

    def remember(self, key: tuple[Any, ...], make: Any) -> bytes:
        with self.lock:
            if key in self.encoded:
                self.encoded.move_to_end(key)
                return self.encoded[key]
        data: bytes = make()
        with self.lock:
            self.encoded[key] = data
            while sum(len(d) for d in self.encoded.values()) > self.max_bytes // 10:
                self.encoded.popitem(last=False)
        return data

    def thumbnail(self, path: Path, found: Kind, size: int) -> bytes:
        """A JPEG whose longest side is at most ``size``. Images are not kept decoded."""

        def make() -> bytes:
            if found == "video":
                picture = first_frame(path)
            else:
                with Image.open(path) as opened:
                    opened.draft("RGB", (size, size))  # JPEG: decode at a reduced scale
                    picture = ImageOps.exif_transpose(opened).convert("RGB")
            picture.thumbnail((size, size), Image.Resampling.BICUBIC, reducing_gap=2.0)
            return jpeg(picture)

        return self.remember(("thumbnail", key_of(path), size), make)

    def preview(
        self, path: Path, found: Kind, width: int, height: int, interpolation: Interpolation
    ) -> bytes:
        """A JPEG of the whole picture resized to ``width`` x ``height``."""

        def make() -> bytes:
            picture = self.picture(path, found)
            resized = picture.resize(
                (width, height),
                RESAMPLE[interpolation],
                reducing_gap=2.0 if interpolation == "bicubic" else None,
            )
            return jpeg(resized)

        return self.remember(("preview", key_of(path), width, height, interpolation), make)

    def crop(
        self,
        path: Path,
        found: Kind,
        box: tuple[float, float, float, float],
        size: int,
        interpolation: Interpolation,
    ) -> bytes:
        """A PNG of the region ``box`` (x, y, width, height in the file's pixels, clipped to
        the picture) resized so that its longest side is ``size``."""

        def make() -> bytes:
            picture = self.picture(path, found)
            x, y, w, h = box
            left, top = max(0.0, x), max(0.0, y)
            right, bottom = min(float(picture.width), x + w), min(float(picture.height), y + h)
            if right <= left or bottom <= top:
                raise ValueError("the region lies outside the picture")
            scale = size / max(right - left, bottom - top)
            out = (max(1, round((right - left) * scale)), max(1, round((bottom - top) * scale)))
            cropped = picture.resize(out, RESAMPLE[interpolation], box=(left, top, right, bottom))
            buffer = io.BytesIO()
            cropped.save(buffer, "PNG")
            return buffer.getvalue()

        return self.remember(("crop", key_of(path), box, size, interpolation), make)


def jpeg(picture: Image.Image) -> bytes:
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()
