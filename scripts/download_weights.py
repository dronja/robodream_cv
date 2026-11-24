"""Download third-party weights into the local weights directory (not into the repository).

Licenses: docs/THIRD_PARTY_LICENSES.md. Assets: ``facehall.weights.ASSETS``.

    uv run python scripts/download_weights.py buffalo_l [--root /mnt/storage/AI/Face/weights]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from facehall.weights import ASSETS, download, weights_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("assets", nargs="+", choices=sorted(ASSETS), help="assets to download")
    parser.add_argument(
        "--root", type=Path, default=None, help="weights directory (default: $FACEHALL_WEIGHTS)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = args.root if args.root is not None else weights_root()
    for name in args.assets:
        asset = ASSETS[name]
        logging.info(
            "downloading %s (%s bytes) from %s; %s", name, asset.size, asset.url, asset.license
        )
        target = download(asset, root)
        print(json.dumps(json.loads((target / "SOURCE.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
