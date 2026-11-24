"""Dataset manifest: one row per image, stored as Parquet next to the images.

Manifests are never committed: they carry dataset annotations (identities, landmarks), which
the CelebA terms do not allow to redistribute (docs/DECISIONS.md, D14). Every manifest has the
columns in ``REQUIRED_FIELDS``; a source adds its own columns (CelebA: docs/DATA.md).
"""

import hashlib
import json
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MANIFEST_FILE = "manifest.parquet"
SPLITS = ("train", "val", "test")
UNKNOWN_IDENTITY = -1

REQUIRED_FIELDS = (
    # Image file relative to the dataset root, POSIX separators.
    pa.field("path", pa.string(), nullable=False),
    # Hex sha256 of the image file bytes.
    pa.field("sha256", pa.string(), nullable=False),
    # One of SPLITS. Splits are identity-disjoint.
    pa.field("split", pa.string(), nullable=False),
    # UNKNOWN_IDENTITY (-1) for sources without identities; those rows are not checked for
    # identity overlap, and such sources split by another key (docs/DATA.md).
    pa.field("identity_id", pa.int64(), nullable=False),
    pa.field("width", pa.int32(), nullable=False),
    pa.field("height", pa.int32(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("license", pa.string(), nullable=False),
)


def validate_manifest(table: pa.Table) -> None:
    """Raise ValueError unless ``table`` is a valid manifest."""
    for field in REQUIRED_FIELDS:
        if field.name not in table.schema.names:
            raise ValueError(f"manifest has no column {field.name!r}")
        actual = table.schema.field(field.name).type
        if actual != field.type:
            raise ValueError(f"column {field.name!r} has type {actual}, expected {field.type}")
        if table.column(field.name).null_count:
            raise ValueError(f"column {field.name!r} has missing values")

    unknown = set(pc.unique(table.column("split")).to_pylist()) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")

    paths = table.column("path").to_pylist()
    if len(set(paths)) != len(paths):
        raise ValueError("manifest has duplicate paths")
    for path in paths:
        parts = PurePosixPath(path)
        if parts.is_absolute() or ".." in parts.parts or "\\" in path:
            raise ValueError(f"path must be relative to the dataset root: {path!r}")

    overlap = _identity_overlap(table)
    if overlap:
        raise ValueError(f"identities shared between splits: {overlap}")


def write_manifest(table: pa.Table, root: Path) -> Path:
    """Validate ``table`` and write it to ``root / MANIFEST_FILE`` (via a temporary file)."""
    validate_manifest(table)
    path = root / MANIFEST_FILE
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path


def read_manifest(root: Path) -> pa.Table:
    """Read and validate the manifest of the dataset at ``root``."""
    table = pq.read_table(root / MANIFEST_FILE)
    validate_manifest(table)
    return table


def content_sha256(table: pa.Table) -> str:
    """Hex sha256 of the manifest content, independent of the Parquet encoding.

    Hashes the column names, then every row as compact JSON, in table order.
    """
    digest = hashlib.sha256()
    digest.update(_json_line(table.schema.names))
    for batch in table.to_batches(max_chunksize=8192):
        for row in batch.to_pylist():
            digest.update(_json_line(list(row.values())))
    return digest.hexdigest()


def _json_line(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def _identity_overlap(table: pa.Table) -> dict[str, int]:
    """Number of known identities shared by each pair of splits (only non-empty pairs)."""
    ids = {
        split: set(table.filter(pc.equal(table.column("split"), split))["identity_id"].to_pylist())
        - {UNKNOWN_IDENTITY}
        for split in SPLITS
    }
    overlap: dict[str, int] = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared = len(ids[a] & ids[b])
            if shared:
                overlap[f"{a}/{b}"] = shared
    return overlap
