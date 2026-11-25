"""Command-line interface for facehall."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from facehall import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="facehall", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    train = commands.add_parser(
        "train",
        help="train a model from a config",
        description="Train a model from a config (configs/train/*.yaml) and write a run "
        "directory with the config, provenance, metrics and checkpoints; or continue an "
        "interrupted run with --resume.",
    )
    start = train.add_mutually_exclusive_group(required=True)
    start.add_argument("--config", type=Path, help="training config (YAML)")
    start.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="continue an interrupted run after its last completed epoch, with its own config",
    )
    train.add_argument(
        "--smoke",
        action="store_true",
        help="short run on random subsets (config section 'smoke'), to check the pipeline",
    )
    train.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.batch_size=32 (repeatable)",
    )

    evaluate = commands.add_parser(
        "eval",
        help="score a model on an evaluation protocol",
        description="Score a checkpoint (--ckpt) or a parameter-free baseline (--model) on an "
        "evaluation protocol (configs/eval/*.yaml). Writes metrics.json and per_image.parquet "
        "to <output>/<protocol>/<split>/<model label>/.",
    )
    evaluate.add_argument("--config", type=Path, required=True, help="protocol config (YAML)")
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", type=Path, help="checkpoint (.safetensors) written by train")
    source.add_argument("--model", help="parameter-free baseline, e.g. bicubic")
    evaluate.add_argument("--split", help="override the protocol split (e.g. val)")
    evaluate.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a protocol value (repeatable)",
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    if args.command == "train":
        from facehall.train import load_experiment, resume
        from facehall.train import train as run_training

        if args.resume is not None:
            if args.smoke or args.overrides:
                parser.error("--resume takes the run's own config: no --smoke or --set")
            print(resume(args.resume))
        else:
            config = load_experiment(args.config, args.overrides)
            print(run_training(config, smoke=args.smoke))
    elif args.command == "eval":
        from facehall.eval.cli import run_eval

        overrides = [*args.overrides, *([f"split={args.split}"] if args.split else [])]
        print(run_eval(args.config, overrides, checkpoint=args.ckpt, baseline=args.model))
    return 0
