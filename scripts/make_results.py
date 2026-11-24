"""Generate docs/RESULTS.md from the ``facehall eval`` outputs, and local comparison grids.

The grids show CelebA faces, so they go to ``outputs/`` (gitignored), never into docs (D19).

    uv run python scripts/make_results.py [--eval-root outputs/eval] [--out docs/RESULTS.md]
"""

import argparse
import sys
from pathlib import Path

from facehall.eval.report import collect, comparison_grid, render


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-root", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--out", type=Path, default=Path("docs/RESULTS.md"))
    parser.add_argument("--grids", type=Path, default=Path("outputs/results"))
    parser.add_argument("--split", default="test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = collect(args.eval_root)
    if not summaries:
        print(f"no metrics.json under {args.eval_root}", file=sys.stderr)
        return 1
    args.out.write_text(render(summaries, split=args.split))
    print(args.out)
    args.grids.mkdir(parents=True, exist_ok=True)
    for protocol in sorted({s["protocol"] for s in summaries if s["split"] == args.split}):
        path = args.grids / f"{protocol}-{args.split}.png"
        comparison_grid(summaries, protocol, args.split).save(path)
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
