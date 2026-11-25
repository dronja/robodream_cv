"""Check a model against the quality gate (D23): prints a table; exit code 1 on failure.

uv run python scripts/quality_gate.py --scale 4 --candidate srresnet_x4-<stamp>-best
"""

import argparse
import sys
from pathlib import Path

from facehall.config import load_config
from facehall.eval.gate import GateConfig, check, render


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--candidate", required=True, help="model label in the eval outputs")
    parser.add_argument("--reference", help="model label (default: the configured reference)")
    parser.add_argument("--config", type=Path, default=Path("configs/gate.yaml"))
    parser.add_argument("--eval-root", type=Path, default=Path("outputs/eval"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, GateConfig)
    reference = args.reference or config.references.get(args.scale, "")
    checks = check(config, args.eval_root, args.scale, args.candidate, reference)
    print(render(checks, args.candidate, reference))
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
