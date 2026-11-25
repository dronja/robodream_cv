"""Score degradation configs by how close their synthetic LR crops are to real LR crops.

Runs the search spec (``facehall.eval.match``) for the scale of a real-world protocol and
writes ``results.json``, ``results.md`` and ``winner.yaml`` to
``<out>/<spec>_x<scale>_<real split>/``. These are local outputs: an accepted winner is copied
into ``configs/degrade/`` by hand, with the evidence in its header.

    uv run python scripts/match_degradations.py --spec configs/degrade/search/realistic.yaml \\
        --real configs/eval/wider_real_x4.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from facehall.config import load_config
from facehall.data.config import DataConfig
from facehall.eval.match import load_spec, render, run_search, winner_yaml
from facehall.eval.noref import InceptionFeatures
from facehall.eval.realworld import RealWorldConfig, protocol_dataset, validate_realworld
from facehall.train.trainer import resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True, help="search spec (YAML)")
    parser.add_argument("--real", type=Path, required=True, help="real-world protocol (YAML)")
    parser.add_argument("--real-split", default="val", help="split of the real crops")
    parser.add_argument("--data", type=Path, default=Path("configs/data/celeba_128.yaml"))
    parser.add_argument("--data-split", default="val", help="split of the synthetic HR crops")
    parser.add_argument("--samples", type=int, default=5000, help="synthetic crops per config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirm", type=int, default=5, help="finalists re-scored on seed + 1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("outputs/degrade_match"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cv2.setNumThreads(1)  # before the DataLoader workers fork (docs/PROJECT.md §9)
    protocol = load_config(args.real, RealWorldConfig, [f"split={args.real_split}"])
    validate_realworld(protocol)
    if not protocol.metrics.inception:
        raise SystemExit(f"{args.real} sets no Inception weights (metrics.inception)")
    device = resolve_device(args.device)
    spec = load_spec(args.spec)
    results = run_search(
        spec,
        protocol_dataset(protocol),
        load_config(args.data, DataConfig),
        data_split=args.data_split,
        extractor=InceptionFeatures(protocol.metrics.inception, device),
        samples=args.samples,
        seed=args.seed,
        confirm=args.confirm,
        kid_subsets=protocol.metrics.kid_subsets,
        kid_subset_size=protocol.metrics.kid_subset_size,
        workers=args.workers,
        device=device,
    )
    out = args.out / f"{spec.name}_x{protocol.scale}_{args.real_split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "results.md").write_text(render(results))
    if results["winner"]:
        verdict = "accepted" if results["accepted"] else "NOT accepted: the base stays"
        header = (
            f"Winner of {args.spec} for x{protocol.scale} against {args.real} "
            f"({args.real_split}), git {results['git_commit']}: {results['winner']} ({verdict})"
        )
        (out / "winner.yaml").write_text(winner_yaml(results, header))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
