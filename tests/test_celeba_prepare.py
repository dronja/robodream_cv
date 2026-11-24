"""Tests for facehall.data.celeba on a synthetic parquet mirror."""

import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from facehall.data.celeba import (
    ATTRIBUTE_NAMES,
    IMAGE_SIZE,
    INFO_FILE,
    OFFICIAL_COUNTS,
    SplitCount,
    attribute_column,
    find_shards,
    prepare_celeba,
)
from facehall.data.manifest import MANIFEST_FILE, content_sha256, read_manifest

# Same schema as the eurecom-ds/celeba shards.
MIRROR_SCHEMA = pa.schema(
    [
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("attributes", pa.list_(pa.int8(), 40)),
        ("identity", pa.int64()),
        ("bbox", pa.list_(pa.int32(), 4)),
        ("landmarks", pa.list_(pa.int32(), 10)),
    ]
)

# Shard file -> identities of its rows.
LAYOUT = {
    "train-00000-of-00002.parquet": [1, 2, 3],
    "train-00001-of-00002.parquet": [3, 4, 5],
    "validation-00000-of-00001.parquet": [10, 11],
    "test-00000-of-00001.parquet": [20, 20],
}
COUNTS = {
    "train": SplitCount(images=6, identities=5),
    "val": SplitCount(images=2, identities=2),
    "test": SplitCount(images=2, identities=1),
}

FORMATS = {
    "train": {"JPEG": 0, "PNG": 6},
    "val": {"JPEG": 2, "PNG": 0},
    "test": {"JPEG": 2, "PNG": 0},
}

Row = dict[str, object]


def encode(
    seed: int, size: tuple[int, int] = IMAGE_SIZE, fmt: str = "JPEG", channels: int = 3
) -> bytes:
    rng = np.random.default_rng(seed)
    shape = (size[1], size[0], channels) if channels > 1 else (size[1], size[0])
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(buffer, format=fmt)
    return buffer.getvalue()


def make_row(identity: int, seed: int, fmt: str = "JPEG") -> Row:
    return {
        "image": {"bytes": encode(seed, fmt=fmt), "path": ""},
        "attributes": [(seed + i) % 2 for i in range(40)],
        "identity": identity,
        "bbox": [0, 0, 1, 1],
        "landmarks": [seed * 10 + i for i in range(10)],
    }


def write_shard(path: Path, rows: list[Row]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=MIRROR_SCHEMA), path)


@pytest.fixture
def mirror(tmp_path: Path) -> tuple[Path, list[Row]]:
    """A complete synthetic mirror; returns its directory and all rows in processing order.

    As in the real mirror, train images are PNG and validation/test images are JPEG.
    """
    src = tmp_path / "mirror"
    src.mkdir()
    rows: list[Row] = []
    seed = 0
    for name, identities in LAYOUT.items():
        shard_rows = []
        fmt = "PNG" if name.startswith("train") else "JPEG"
        for identity in identities:
            shard_rows.append(make_row(identity, seed, fmt))
            seed += 1
        write_shard(src / name, shard_rows)
        rows.extend(shard_rows)
    return src, rows


def test_prepare_writes_original_bytes_and_manifest(
    mirror: tuple[Path, list[Row]], tmp_path: Path
) -> None:
    src, rows = mirror
    out = tmp_path / "out"
    info = prepare_celeba(src, out, expected=COUNTS)

    table = read_manifest(out)
    assert table.num_rows == len(rows)
    assert table.column("path").to_pylist()[:7] == [
        *(f"img_align/train/{i:06d}.png" for i in range(6)),
        "img_align/val/000000.jpg",
    ]
    assert table.column("split").to_pylist() == ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    assert table.column("source_row").to_pylist() == [0, 1, 2, 0, 1, 2, 0, 1, 0, 1]

    for record, row in zip(table.to_pylist(), rows, strict=True):
        data = (out / record["path"]).read_bytes()
        assert data == row["image"]["bytes"]  # type: ignore[index]
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
        assert record["identity_id"] == row["identity"]
        assert record["landmarks"] == row["landmarks"]
        assert (record["width"], record["height"]) == IMAGE_SIZE
        attributes = [record[attribute_column(name)] for name in ATTRIBUTE_NAMES]
        assert attributes == [bool(v) for v in row["attributes"]]  # type: ignore[attr-defined]

    assert info["splits"] == {
        name: {
            "images": c.images,
            "identities": c.identities,
            "blurry": _blurry(rows, name),
            "formats": FORMATS[name],
        }
        for name, c in COUNTS.items()
    }
    saved = json.loads((out / INFO_FILE).read_text())
    assert saved["manifest"]["content_sha256"] == content_sha256(table)
    assert [s["file"] for s in saved["shards"]] == list(LAYOUT)


def _blurry(rows: list[Row], split: str) -> int:
    first = {"train": 0, "val": 6, "test": 8}[split]
    count = {"train": 6, "val": 2, "test": 2}[split]
    blurry = ATTRIBUTE_NAMES.index("Blurry")
    return sum(r["attributes"][blurry] for r in rows[first : first + count])  # type: ignore[index]


def test_official_counts_are_checked(mirror: tuple[Path, list[Row]], tmp_path: Path) -> None:
    src, _ = mirror
    with pytest.raises(ValueError, match="differ from the expected"):
        prepare_celeba(src, tmp_path / "out", expected=OFFICIAL_COUNTS)
    assert not (tmp_path / "out").exists()


def test_identity_shared_between_splits_is_rejected(
    mirror: tuple[Path, list[Row]], tmp_path: Path
) -> None:
    src, _ = mirror
    write_shard(src / "validation-00000-of-00001.parquet", [make_row(1, 100)])
    with pytest.raises(ValueError, match="identities appear in both 'train' and 'val'"):
        prepare_celeba(src, tmp_path / "out", expected=None)


def test_non_empty_output_needs_overwrite(mirror: tuple[Path, list[Row]], tmp_path: Path) -> None:
    src, _ = mirror
    out = tmp_path / "out"
    first = prepare_celeba(src, out, expected=COUNTS)
    with pytest.raises(FileExistsError):
        prepare_celeba(src, out, expected=COUNTS)
    second = prepare_celeba(src, out, expected=COUNTS, overwrite=True)
    assert second["manifest"] == first["manifest"]


@pytest.mark.parametrize(
    "image",
    [
        encode(0, fmt="BMP"),
        encode(0, channels=1),
        encode(0, size=(128, 128)),
        encode(0)[:500],  # truncated JPEG
        b"not an image",
    ],
    ids=["bmp", "grayscale", "wrong-size", "truncated", "garbage"],
)
def test_bad_image_is_rejected(
    mirror: tuple[Path, list[Row]], tmp_path: Path, image: bytes
) -> None:
    src, _ = mirror
    row = make_row(20, 200)
    row["image"] = {"bytes": image, "path": ""}
    write_shard(src / "test-00000-of-00001.parquet", [row])
    with pytest.raises(ValueError, match="test-00000-of-00001.parquet row 0"):
        prepare_celeba(src, tmp_path / "out", expected=None)
    assert not (tmp_path / "out" / MANIFEST_FILE).exists()


def test_non_binary_attribute_is_rejected(mirror: tuple[Path, list[Row]], tmp_path: Path) -> None:
    src, _ = mirror
    row = make_row(20, 200)
    row["attributes"] = [-1] * 40
    write_shard(src / "test-00000-of-00001.parquet", [row])
    with pytest.raises(ValueError, match="attribute values"):
        prepare_celeba(src, tmp_path / "out", expected=None)


def test_incomplete_shard_set_is_rejected(mirror: tuple[Path, list[Row]]) -> None:
    src, _ = mirror
    (src / "train-00001-of-00002.parquet").unlink()
    with pytest.raises(ValueError, match="incomplete shard set for split 'train'"):
        find_shards(src)


def test_unexpected_parquet_file_is_rejected(mirror: tuple[Path, list[Row]]) -> None:
    src, _ = mirror
    (src / "extra.parquet").write_bytes(b"")
    with pytest.raises(ValueError, match="unexpected parquet file"):
        find_shards(src)


def test_shards_are_ordered_by_split_then_index(mirror: tuple[Path, list[Row]]) -> None:
    src, _ = mirror
    assert [s.path.name for s in find_shards(src)] == list(LAYOUT)


def test_attribute_names_follow_the_official_order() -> None:
    assert len(set(ATTRIBUTE_NAMES)) == 40
    assert ATTRIBUTE_NAMES.index("Blurry") == 10
    assert ATTRIBUTE_NAMES.index("Male") == 20
    assert ATTRIBUTE_NAMES.index("Pale_Skin") == 26
    assert ATTRIBUTE_NAMES.index("Young") == 39
    assert attribute_column("5_o_Clock_Shadow") == "attr_5_o_clock_shadow"


def test_script_prepares_dataset(
    mirror: tuple[Path, list[Row]],
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    src, rows = mirror
    out = tmp_path / "out"
    script = load_script("prepare_dataset")
    assert script.main(["--src", str(src), "--out", str(out), "--no-count-check"]) == 0
    assert json.loads(capsys.readouterr().out)["manifest"]["rows"] == len(rows)
    assert (out / MANIFEST_FILE).is_file()
