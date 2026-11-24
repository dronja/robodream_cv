"""Prepare CelebA aligned from the Hugging Face parquet mirror ``eurecom-ds/celeba``.

Writes the original JPEG bytes as files, ``manifest.parquet`` and ``prepare_info.json`` into
the output directory; see docs/DATA.md. The output contains CelebA data: keep it outside the
repository (the CelebA terms forbid redistribution).

    uv run python scripts/prepare_dataset.py \\
        --src /mnt/storage/AI/Face/celeba_eurecom/data --out /mnt/storage/AI/Face/celeba
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from facehall.data.celeba import OFFICIAL_COUNTS, prepare_celeba


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="directory with the parquet shards")
    parser.add_argument("--out", type=Path, required=True, help="output dataset directory")
    parser.add_argument(
        "--overwrite", action="store_true", help="write into a non-empty output directory"
    )
    parser.add_argument(
        "--no-count-check",
        action="store_true",
        help="skip the comparison with the official CelebA counts (partial copies)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    info = prepare_celeba(
        args.src,
        args.out,
        expected=None if args.no_count_check else OFFICIAL_COUNTS,
        overwrite=args.overwrite,
    )
    print(json.dumps({"splits": info["splits"], "manifest": info["manifest"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
