"""Print the inputs digest of evaluation protocols (no model needed).

The digest is the sha256 of every LR and HR crop of the split (``facehall.eval.runner``). Put it
under ``inputs_sha256`` in the protocol config to freeze the inputs: later evaluations then fail
if the inputs change, e.g. after a library update.

    uv run python scripts/protocol_digest.py configs/eval/celeba128_light_x4.yaml [...]
"""

import argparse
import sys
from pathlib import Path

from facehall.config import load_config
from facehall.eval.config import EvalConfig
from facehall.eval.runner import input_digest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+", type=Path, help="evaluation protocol configs")
    parser.add_argument("--split", default=None, help="split (default: the protocol's)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    changed = 0
    for path in args.configs:
        overrides = [f"split={args.split}"] if args.split else []
        config = load_config(path, EvalConfig, overrides)
        digest = input_digest(config)
        frozen = config.inputs_sha256.get(config.split)
        status = "new" if not frozen else "matches" if frozen == digest else "CHANGED"
        print(f"{path}\t{config.name}\t{config.split}\t{digest}\t{status}")
        changed += status == "CHANGED"
    if changed:
        print(f"{changed} protocol(s) no longer reproduce their frozen inputs", file=sys.stderr)
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
