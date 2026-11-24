"""Provenance of produced artifacts: file checksums, git commit, library versions."""

import hashlib
import subprocess
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path | None = None) -> str | None:
    """HEAD commit of ``repo`` (default: the checkout that contains this package).

    A ``-dirty`` suffix marks uncommitted changes. Returns None outside a git checkout.
    """
    cwd = repo if repo is not None else Path(__file__).resolve().parent
    try:
        commit = _git(cwd, "rev-parse", "HEAD")
        changes = _git(cwd, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if changes else "")


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Installed versions of the given distributions; None for missing ones."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()
