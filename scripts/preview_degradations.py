"""Preview a degradation config on dataset crops: a grid image and the drawn parameters.

Rows are random crops of one split; columns are the HR crop and several LR variants. The grid
shows dataset faces, so keep it local: the default output directory is gitignored (D19).

    uv run python scripts/preview_degradations.py --degrade configs/degrade/realistic.yaml \
        --scale 4 [--data configs/data/celeba_128.yaml] [--split val] [--out outputs/degrade]
"""

import argparse
import json
import sys
from pathlib import Path

from facehall.config import load_config
from facehall.data.config import DataConfig, DegradeConfig
from facehall.data.preview import preview


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrade", type=Path, required=True, help="degradation config")
    parser.add_argument("--data", type=Path, default=Path("configs/data/celeba_128.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--count", type=int, default=8, help="crops (rows)")
    parser.add_argument("--variants", type=int, default=6, help="LR variants per crop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/degrade"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = preview(
        load_config(args.data, DataConfig),
        load_config(args.degrade, DegradeConfig),
        args.split,
        args.scale,
        count=args.count,
        variants=args.variants,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.degrade.stem}_x{args.scale}_{args.split}"
    result.grid.save(args.out / f"{stem}.png")
    (args.out / f"{stem}.json").write_text(json.dumps(result.records, indent=1) + "\n")
    print(args.out / f"{stem}.png")
    print(f"{result.seconds_per_call * 1e3:.2f} ms per degradation call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
