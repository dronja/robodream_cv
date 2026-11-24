"""Tests for facehall.provenance."""

import hashlib
import re
from pathlib import Path

from facehall.provenance import file_sha256, git_commit, package_versions


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    data = bytes(range(256)) * 1000
    path.write_bytes(data)
    assert file_sha256(path, chunk_size=1000) == hashlib.sha256(data).hexdigest()


def test_git_commit_outside_a_checkout_is_none(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None


def test_git_commit_of_this_checkout() -> None:
    commit = git_commit(Path(__file__).resolve().parent)
    assert commit is not None
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", commit)


def test_package_versions() -> None:
    versions = package_versions(["numpy", "no-such-distribution-facehall"])
    assert versions["numpy"]
    assert versions["no-such-distribution-facehall"] is None
