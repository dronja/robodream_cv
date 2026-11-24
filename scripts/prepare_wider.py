"""Prepare the real-world low-resolution face set from WIDER FACE and the RetinaFace landmarks.

Reads the three raw archives (docs/DATA.md §7) and writes the aligned LR crops (PNG),
``manifest.parquet`` and ``prepare_info.json`` into the output directory. The output is derived
from WIDER FACE (CC BY-NC-ND): keep it outside the repository and do not share it.

    uv run python scripts/prepare_wider.py \\
        --raw /mnt/storage/AI/Face/wider_face/raw --out /mnt/storage/AI/Face/wider_face/realworld
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from facehall.data.wider import ARCHIVES, prepare_wider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="directory with the raw archives")
    parser.add_argument("--out", type=Path, required=True, help="output dataset directory")
    parser.add_argument(
        "--overwrite", action="store_true", help="write into a non-empty output directory"
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="skip the comparison with the pinned archive checksums",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    info = prepare_wider(
        args.raw,
        args.out,
        expected=None if args.no_checksum else ARCHIVES,
        overwrite=args.overwrite,
    )
    summary = {key: info[key] for key in ("annotations", "selection", "splits", "manifest")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
