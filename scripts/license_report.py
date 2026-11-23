"""Print a Markdown table with the licenses of the packages in a requirements list.

The license is read from the installed package metadata, so run it in the project
environment and review the result by hand before updating THIRD_PARTY_LICENSES.md:

    uv export --no-dev --no-hashes --no-emit-project --extra cuda --format requirements.txt \\
        | uv run python scripts/license_report.py
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from importlib import metadata
from typing import Protocol

UNKNOWN = "UNKNOWN"
_MAX_FIELD_LEN = 60
_PINNED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s;\\]+)")


class Metadata(Protocol):
    """The part of the package metadata interface used here."""

    def get_all(self, name: str) -> list[str] | None: ...


def parse_requirements(lines: Iterable[str]) -> list[tuple[str, str]]:
    """Return (name, version) for every pinned requirement (``name==version``)."""
    pinned: list[tuple[str, str]] = []
    for line in lines:
        match = _PINNED.match(line.strip())
        if match:
            pinned.append((match.group(1), match.group(2)))
    return pinned


def _first(meta: Metadata, field: str) -> str:
    values = meta.get_all(field) or []
    return values[0].strip() if values else ""


def license_of(meta: Metadata) -> str:
    """Pick the most specific license statement available in the metadata.

    Order: ``License-Expression``, a short one-line ``License`` field, license
    classifiers, the first line of a long ``License`` field, ``UNKNOWN``.
    """
    expression = _first(meta, "License-Expression")
    if expression:
        return expression

    field = _first(meta, "License")
    if field.upper() == UNKNOWN:
        field = ""
    if field and "\n" not in field and len(field) <= _MAX_FIELD_LEN:
        return field

    classifiers = [
        value.split("::")[-1].strip()
        for value in meta.get_all("Classifier") or []
        if value.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(classifiers)

    if field:
        first_line = field.splitlines()[0].strip()
        return first_line[: _MAX_FIELD_LEN - 3] + "..."
    return UNKNOWN


def report(pinned: Iterable[tuple[str, str]]) -> str:
    """Render the license table for the given packages."""
    rows = ["| Package | Version | License (from package metadata) |", "|---|---|---|"]
    for name, version in sorted(pinned, key=lambda item: item[0].lower()):
        try:
            license_text = license_of(metadata.metadata(name))
        except metadata.PackageNotFoundError:
            license_text = "not installed in this environment"
        rows.append(f"| {name} | {version} | {license_text.replace('|', '/')} |")
    return "\n".join(rows)


def main() -> int:
    print(report(parse_requirements(sys.stdin)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
