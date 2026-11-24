"""Tests for facehall.data.manifest."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from facehall.data.manifest import (
    MANIFEST_FILE,
    REQUIRED_FIELDS,
    content_sha256,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def make_table(**overrides: list[object]) -> pa.Table:
    columns: dict[str, list[object]] = {
        "path": ["img/train/0.jpg", "img/val/0.jpg"],
        "sha256": ["a" * 64, "b" * 64],
        "split": ["train", "val"],
        "identity_id": [1, 2],
        "width": [178, 178],
        "height": [218, 218],
        "source": ["synthetic", "synthetic"],
        "license": ["test", "test"],
        "extra": [0.5, 1.5],
    }
    columns.update(overrides)
    schema = pa.schema([*REQUIRED_FIELDS, pa.field("extra", pa.float64())])
    return pa.Table.from_pydict(columns, schema=schema)


def test_valid_manifest_round_trips(tmp_path: Path) -> None:
    table = make_table()
    write_manifest(table, tmp_path)
    assert read_manifest(tmp_path).equals(table)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"path": ["img/a.jpg", "img/a.jpg"]}, "duplicate paths"),
        ({"path": ["/abs/a.jpg", "img/b.jpg"]}, "relative"),
        ({"path": ["../a.jpg", "img/b.jpg"]}, "relative"),
        ({"split": ["train", "validation"]}, "unknown split"),
        ({"identity_id": [1, 1]}, "identities shared between splits"),
    ],
)
def test_invalid_manifest_is_rejected(overrides: dict[str, list[object]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_manifest(make_table(**overrides))


def test_missing_column_and_wrong_type_are_rejected() -> None:
    table = make_table()
    with pytest.raises(ValueError, match="no column 'license'"):
        validate_manifest(table.drop_columns(["license"]))
    wrong = table.set_column(
        table.schema.get_field_index("width"), "width", pa.array([178, 178], pa.int64())
    )
    with pytest.raises(ValueError, match="'width' has type int64"):
        validate_manifest(wrong)


def test_content_sha256_ignores_parquet_encoding(tmp_path: Path) -> None:
    table = make_table()
    pq.write_table(table, tmp_path / "a.parquet", compression="zstd", row_group_size=1)
    pq.write_table(table, tmp_path / "b.parquet", compression="none")
    a = pq.read_table(tmp_path / "a.parquet")
    b = pq.read_table(tmp_path / "b.parquet")
    assert (tmp_path / "a.parquet").read_bytes() != (tmp_path / "b.parquet").read_bytes()
    assert content_sha256(a) == content_sha256(b) == content_sha256(table)
    assert content_sha256(make_table(extra=[0.5, 2.5])) != content_sha256(table)


def test_read_manifest_validates(tmp_path: Path) -> None:
    pq.write_table(make_table(split=["train", "dev"]), tmp_path / MANIFEST_FILE)
    with pytest.raises(ValueError, match="unknown split"):
        read_manifest(tmp_path)
