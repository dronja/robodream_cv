"""Third-party weights: pinned download sources, checksums and the local weights directory.

Licenses and the reason for every asset are in ``docs/THIRD_PARTY_LICENSES.md``; nothing here is
committed to the repository. ``scripts/download_weights.py`` downloads an asset, checks its size
and sha256, extracts the files that are used (or keeps a single downloaded file) and writes
``SOURCE.json`` next to them.
"""

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from facehall.provenance import file_sha256

DEFAULT_ROOT = "/mnt/storage/AI/Face/weights"


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int  # bytes of the downloaded file
    sha256: str | None  # of the downloaded file; None until pinned
    members: tuple[str, ...]  # files kept from the zip archive; empty: keep the downloaded file
    license: str  # short note; details in docs/THIRD_PARTY_LICENSES.md


ASSETS = {
    "buffalo_l": Asset(
        name="buffalo_l",
        url="https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip",
        size=288_621_354,
        sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",  # 2026-11-24
        members=("w600k_r50.onnx",),
        license="InsightFace pretrained models: non-commercial research only (D18)",
    ),
    "inception": Asset(
        name="inception",
        url=(
            "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
            "weights-inception-2015-12-05-6726825d.pth"
        ),
        size=95_628_359,
        sha256="6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",  # 2026-11-24
        members=(),
        license=(
            "TensorFlow Inception v3 (2015-12-05) converted by torch-fidelity: Apache-2.0 per the "
            "torch-fidelity source; ImageNet terms need verification (Q6)"
        ),
    ),
}
INCEPTION_FILE = "weights-inception-2015-12-05-6726825d.pth"


def inception_weights(root: Path | None = None) -> Path:
    """Path of the InceptionV3 weights used for FID and KID (``download_weights.py inception``)."""
    return asset_dir("inception", root) / INCEPTION_FILE


def weights_root() -> Path:
    """``$FACEHALL_WEIGHTS``, or the default directory of the development machine."""
    return Path(os.environ.get("FACEHALL_WEIGHTS", DEFAULT_ROOT))


def asset_dir(name: str, root: Path | None = None) -> Path:
    return (root if root is not None else weights_root()) / name


def download(asset: Asset, root: Path, *, chunk_size: int = 1 << 20) -> Path:
    """Download and check ``asset``, extract its members into ``root / asset.name``.

    Without members, the downloaded file itself is kept under its URL file name. Returns the
    asset directory. An archive is removed after extraction; ``SOURCE.json`` keeps the URL, the
    download checksum and the checksums of the kept files.
    """
    target = asset_dir(asset.name, root)
    target.mkdir(parents=True, exist_ok=True)
    archive = target / (Path(asset.url).name + ".partial")
    digest = hashlib.sha256()
    with urllib.request.urlopen(asset.url) as response, archive.open("wb") as out:  # noqa: S310
        while chunk := response.read(chunk_size):
            digest.update(chunk)
            out.write(chunk)
    sha256 = digest.hexdigest()
    size = archive.stat().st_size
    if size != asset.size or (asset.sha256 is not None and sha256 != asset.sha256):
        archive.unlink()
        raise ValueError(
            f"{asset.url}: got {size} bytes with sha256 {sha256}, expected {asset.size} bytes"
            + (f" with sha256 {asset.sha256}" if asset.sha256 else "")
        )
    if asset.members:
        files = extract_members(archive, asset.members, target)
        archive.unlink()
    else:
        files = [target / Path(asset.url).name]
        archive.replace(files[0])
    source = {
        "asset": asset.name,
        "url": asset.url,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "archive_size": size,
        "archive_sha256": sha256,
        "files": {path.name: file_sha256(path) for path in files},
        "license": asset.license,
    }
    (target / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    return target


def extract_members(archive: Path, members: tuple[str, ...], target: Path) -> list[Path]:
    """Copy the archive entries whose file name is in ``members`` into ``target`` (flat)."""
    found: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or name not in members:
                continue
            path = target / name
            with zf.open(info) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            found.append(path)
    missing = set(members) - {p.name for p in found}
    if missing:
        raise ValueError(f"{archive.name} lacks {sorted(missing)}")
    return found
