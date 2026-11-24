"""CelebA aligned images from the Hugging Face parquet mirror ``eurecom-ds/celeba``.

The mirror stores ``img_align_celeba`` (RGB, 178x218) with the identity, the aligned 5-point
landmarks and the 40 binary attributes of every image, in parquet shards named
``{train,validation,test}-NNNNN-of-NNNNN.parquet`` (the official partition). Validation and
test images are the original JPEG files; train images are stored as PNG (decoded JPEG pixels,
docs/DATA.md §4). The ``bbox`` column holds in-the-wild coordinates and is not used.

``prepare_celeba`` writes the stored image bytes as files without re-encoding, plus the
manifest and ``prepare_info.json``. Layout and columns: docs/DATA.md.
"""

import hashlib
import io
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import facehall
from facehall.data.manifest import (
    MANIFEST_FILE,
    REQUIRED_FIELDS,
    SPLITS,
    content_sha256,
    write_manifest,
)
from facehall.provenance import file_sha256, git_commit, package_versions

log = logging.getLogger(__name__)

#: Attribute names in the order of the official ``list_attr_celeba.txt``. The mirror stores no
#: names; the order was checked on the data (docs/DATA.md §4).
ATTRIBUTE_NAMES: tuple[str, ...] = (
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
)

IMAGE_SIZE = (178, 218)  # (width, height) of img_align_celeba
IMAGE_MODE = "RGB"
IMAGE_SUFFIXES = {"JPEG": ".jpg", "PNG": ".png"}  # accepted PIL formats -> file suffix
IMAGE_DIR = "img_align"
INFO_FILE = "prepare_info.json"
SOURCE = "celeba-aligned"
LICENSE = "CelebA: non-commercial research only, no redistribution"
LANDMARK_VALUES = 10  # lx, ly, rx, ry, nose_x, nose_y, lmouth_x, lmouth_y, rmouth_x, rmouth_y

# Mirror split names -> manifest split names, in processing order.
MIRROR_SPLITS = {"train": "train", "validation": "val", "test": "test"}

_SHARD_NAME = re.compile(
    r"^(?P<split>train|validation|test)-(?P<index>\d{5})-of-(?P<total>\d{5})\.parquet$"
)
_BATCH_ROWS = 2048


@dataclass(frozen=True)
class SplitCount:
    images: int
    identities: int


#: Official partition, as found in the mirror on 2026-11-23 (docs/DATA.md §3).
OFFICIAL_COUNTS: dict[str, SplitCount] = {
    "train": SplitCount(images=162_770, identities=8_192),
    "val": SplitCount(images=19_867, identities=985),
    "test": SplitCount(images=19_962, identities=1_000),
}


@dataclass(frozen=True)
class Shard:
    path: Path
    split: str  # manifest split name
    index: int
    total: int


def attribute_column(name: str) -> str:
    """Manifest column for a CelebA attribute, e.g. ``Male`` -> ``attr_male``."""
    return "attr_" + name.lower()


def manifest_schema() -> pa.Schema:
    """Schema of the CelebA manifest: the required columns plus CelebA annotations."""
    return pa.schema(
        [
            *REQUIRED_FIELDS,
            pa.field("landmarks", pa.list_(pa.int32(), LANDMARK_VALUES), nullable=False),
            pa.field("source_file", pa.string(), nullable=False),
            pa.field("source_row", pa.int64(), nullable=False),
            *(pa.field(attribute_column(n), pa.bool_(), nullable=False) for n in ATTRIBUTE_NAMES),
        ]
    )


def find_shards(src: Path) -> list[Shard]:
    """All mirror shards in ``src``, ordered train, val, test and by shard index.

    Raises ValueError for unexpected parquet files or incomplete shard sets.
    """
    files = sorted(src.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {src}")
    shards = []
    for path in files:
        match = _SHARD_NAME.match(path.name)
        if match is None:
            raise ValueError(f"unexpected parquet file: {path.name}")
        shards.append(
            Shard(
                path=path,
                split=MIRROR_SPLITS[match["split"]],
                index=int(match["index"]),
                total=int(match["total"]),
            )
        )
    for split in SPLITS:
        group = [s for s in shards if s.split == split]
        totals = {s.total for s in group}
        complete = len(totals) == 1 and sorted(s.index for s in group) == list(range(totals.pop()))
        if not complete:
            raise ValueError(f"incomplete shard set for split {split!r} in {src}")
    return sorted(shards, key=lambda s: (SPLITS.index(s.split), s.index))


def check_splits(
    shards: list[Shard], expected: Mapping[str, SplitCount] | None = OFFICIAL_COUNTS
) -> dict[str, SplitCount]:
    """Count images and identities per split; check they are disjoint and as expected."""
    identities = {
        split: np.concatenate(
            [_read_column(s.path, "identity") for s in shards if s.split == split]
        )
        for split in SPLITS
    }
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared = np.intersect1d(identities[a], identities[b]).size
            if shared:
                raise ValueError(f"{shared} identities appear in both {a!r} and {b!r}")
    counts = {
        split: SplitCount(images=int(ids.size), identities=int(np.unique(ids).size))
        for split, ids in identities.items()
    }
    if expected is not None and counts != dict(expected):
        raise ValueError(f"split counts {counts} differ from the expected {dict(expected)}")
    return counts


def prepare_celeba(
    src: Path,
    out: Path,
    *,
    expected: Mapping[str, SplitCount] | None = OFFICIAL_COUNTS,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write images, manifest and ``prepare_info.json`` for the mirror in ``src`` to ``out``.

    Images go to ``out/img_align/{split}/{index:06d}{.jpg|.png}``, where ``index`` counts rows
    within the split in shard order and the suffix follows the stored format. Every image is
    decoded once and must be a 178x218 RGB JPEG or PNG. Pass
    ``expected=None`` to skip the comparison with the official counts (partial copies).
    Returns the content of ``prepare_info.json``.
    """
    shards = find_shards(src)
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; use overwrite to write into it")
    counts = check_splits(shards, expected)
    log.info("splits: %s", counts)

    rows = _Rows()
    next_index = dict.fromkeys(SPLITS, 0)
    shard_info = []
    for shard in shards:
        start = time.monotonic()
        written = _convert_shard(shard, out, next_index[shard.split], rows)
        next_index[shard.split] += written
        shard_info.append(
            {
                "file": shard.path.name,
                "split": shard.split,
                "rows": written,
                "sha256": file_sha256(shard.path),
            }
        )
        log.info("%s: %d images in %.1f s", shard.path.name, written, time.monotonic() - start)

    table = rows.to_table()
    manifest_path = write_manifest(table, out)
    info: dict[str, object] = {
        "dataset": SOURCE,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_dir": str(src),
        "git_commit": git_commit(),
        "versions": {
            "facehall": facehall.__version__,
            **package_versions(["numpy", "pillow", "pyarrow"]),
        },
        "shards": shard_info,
        "splits": _split_summary(table, counts),
        "manifest": {
            "file": MANIFEST_FILE,
            "rows": table.num_rows,
            "content_sha256": content_sha256(table),
            "file_sha256": file_sha256(manifest_path),
        },
    }
    (out / INFO_FILE).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def _split_summary(table: pa.Table, counts: Mapping[str, SplitCount]) -> dict[str, object]:
    """Per split: images, identities, ``Blurry`` images and images per stored format."""
    split = table.column("split").to_numpy(zero_copy_only=False)
    blurry = table.column(attribute_column("Blurry")).to_numpy(zero_copy_only=False)
    suffix = np.array([Path(p).suffix for p in table.column("path").to_pylist()])
    summary: dict[str, object] = {}
    for name, count in counts.items():
        rows = split == name
        summary[name] = {
            "images": count.images,
            "identities": count.identities,
            "blurry": int(blurry[rows].sum()),
            "formats": {fmt: int((suffix[rows] == s).sum()) for fmt, s in IMAGE_SUFFIXES.items()},
        }
    return summary


@dataclass
class _Rows:
    """Manifest columns, accumulated shard by shard."""

    path: list[str] = field(default_factory=list)
    sha256: list[str] = field(default_factory=list)
    split: list[str] = field(default_factory=list)
    identity_id: list[int] = field(default_factory=list)
    width: list[int] = field(default_factory=list)
    height: list[int] = field(default_factory=list)
    source_file: list[str] = field(default_factory=list)
    source_row: list[int] = field(default_factory=list)
    landmarks: list[npt.NDArray[np.int32]] = field(default_factory=list)
    attributes: list[npt.NDArray[np.bool_]] = field(default_factory=list)

    def to_table(self) -> pa.Table:
        n = len(self.path)
        landmarks = np.concatenate(self.landmarks).reshape(-1)
        attributes = np.concatenate(self.attributes)
        columns = {
            "path": self.path,
            "sha256": self.sha256,
            "split": self.split,
            "identity_id": self.identity_id,
            "width": self.width,
            "height": self.height,
            "source": [SOURCE] * n,
            "license": [LICENSE] * n,
            "landmarks": pa.FixedSizeListArray.from_arrays(
                pa.array(landmarks, pa.int32()), LANDMARK_VALUES
            ),
            "source_file": self.source_file,
            "source_row": self.source_row,
            **{attribute_column(name): attributes[:, i] for i, name in enumerate(ATTRIBUTE_NAMES)},
        }
        return pa.Table.from_pydict(columns, schema=manifest_schema())


def _convert_shard(shard: Shard, out: Path, first_index: int, rows: _Rows) -> int:
    """Write the images of one shard and append its manifest rows; return the row count."""
    image_dir = out / IMAGE_DIR / shard.split
    image_dir.mkdir(parents=True, exist_ok=True)
    columns = ["image", "identity", "landmarks", "attributes"]
    shard_row = 0
    for batch in pq.ParquetFile(shard.path).iter_batches(batch_size=_BATCH_ROWS, columns=columns):
        for name in columns:
            if batch.column(name).null_count:
                raise ValueError(f"{shard.path.name}: column {name!r} has missing values")
        attributes = _fixed_size_values(batch.column("attributes"), len(ATTRIBUTE_NAMES))
        if not np.isin(attributes, (0, 1)).all():
            raise ValueError(f"{shard.path.name}: attribute values outside {{0, 1}}")
        rows.attributes.append(attributes.astype(np.bool_))
        rows.landmarks.append(
            _fixed_size_values(batch.column("landmarks"), LANDMARK_VALUES).astype(np.int32)
        )
        rows.identity_id.extend(batch.column("identity").to_pylist())

        for data in batch.column("image").field("bytes").to_pylist():
            index = first_index + shard_row
            where = f"{shard.path.name} row {shard_row}"
            suffix, (width, height) = _check_image(data, where)
            relative = f"{IMAGE_DIR}/{shard.split}/{index:06d}{suffix}"
            (out / relative).write_bytes(data)
            rows.path.append(relative)
            rows.sha256.append(hashlib.sha256(data).hexdigest())
            rows.split.append(shard.split)
            rows.width.append(width)
            rows.height.append(height)
            rows.source_file.append(shard.path.name)
            rows.source_row.append(shard_row)
            shard_row += 1
    return shard_row


def _check_image(data: bytes, where: str) -> tuple[str, tuple[int, int]]:
    """Decode the image fully and check format, mode and size; return (suffix, (w, h))."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            fmt, mode, size = image.format, image.mode, image.size
    except OSError as exc:  # PIL raises OSError subclasses for unreadable or truncated data
        raise ValueError(f"{where}: cannot decode image: {exc}") from exc
    if fmt not in IMAGE_SUFFIXES or mode != IMAGE_MODE or size != IMAGE_SIZE:
        raise ValueError(
            f"{where}: expected a {IMAGE_SIZE} {IMAGE_MODE} JPEG or PNG, got {fmt} {mode} {size}"
        )
    return IMAGE_SUFFIXES[fmt], size


def _fixed_size_values(column: pa.Array, width: int) -> npt.NDArray[np.int64]:
    """Values of a fixed-size list column as an (n, width) array."""
    values = column.flatten().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.int64).reshape(-1, width)


def _read_column(path: Path, name: str) -> npt.NDArray[np.int64]:
    values = pq.read_table(path, columns=[name]).column(name).to_numpy()
    return np.asarray(values, dtype=np.int64)
