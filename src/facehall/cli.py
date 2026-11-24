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
        "directory with the config, provenance, metrics and checkpoints.",
    )
    train.add_argument("--config", type=Path, required=True, help="training config (YAML)")
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

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    if args.command == "train":
        from facehall.train import load_experiment
        from facehall.train import train as run_training

        config = load_experiment(args.config, args.overrides)
        print(run_training(config, smoke=args.smoke))
    return 0
