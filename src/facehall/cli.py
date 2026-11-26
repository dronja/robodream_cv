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

    enhance = commands.add_parser(
        "enhance",
        help="restore the faces of photos and videos",
        description="Restore the faces of photos, videos or folders of them with a restoration "
        "model and write marked outputs (AI-enhanced metadata; videos need the ffmpeg program). "
        "The restored faces are a plausible "
        "reconstruction, not a recovery of the real details: do not use them to identify a "
        "person or as evidence.",
    )
    enhance.add_argument(
        "inputs", type=Path, nargs="+", metavar="INPUT", help="image, video or folder"
    )
    enhance.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output image or video (one input file) or folder (mirrors the inputs)",
    )
    model = enhance.add_mutually_exclusive_group()
    model.add_argument(
        "--scale", type=int, choices=(4, 8), help="model scale from the pipeline config (4)"
    )
    model.add_argument("--ckpt", type=Path, help="restoration checkpoint (.safetensors)")
    enhance.add_argument(
        "--pipeline",
        type=Path,
        default=Path("configs/pipeline/default.yaml"),
        help="pipeline config (YAML; default: %(default)s)",
    )
    enhance.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a pipeline value, e.g. --set color_correction=true (repeatable)",
    )
    faces = enhance.add_mutually_exclusive_group()
    faces.add_argument(
        "--landmarks",
        action="append",
        default=[],
        metavar="X1,Y1,...,X5,Y5",
        help="five landmarks of a face instead of detection (eye centres, nose tip, mouth "
        "corners; image pixels; one input image; repeatable)",
    )
    faces.add_argument(
        "--box",
        action="append",
        default=[],
        metavar="LEFT,TOP,WIDTH,HEIGHT",
        help="a face box for a face the detector misses: tight, like a detector's (cheek to "
        "cheek, forehead to chin, about 2.2 eye distances wide; image pixels; one input image; "
        "repeatable)",
    )
    enhance.add_argument(
        "--watermark", action="store_true", help="add a visible 'AI-enhanced' mark"
    )
    enhance.add_argument(
        "--format",
        choices=("png", "jpg", "webp"),
        default="png",
        help="image format in an output folder (default: %(default)s)",
    )
    enhance.add_argument(
        "--video-format",
        choices=("mp4", "mkv", "mov"),
        default="mp4",
        help="video format in an output folder (default: %(default)s)",
    )
    enhance.add_argument("--quality", type=int, default=95, help="JPEG and WebP quality")
    enhance.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    enhance.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")

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
    elif args.command == "enhance":
        from facehall.enhance import run_enhance

        try:
            return run_enhance(
                args.inputs,
                args.output,
                pipeline=args.pipeline,
                overrides=args.overrides,
                checkpoint=args.ckpt,
                scale=args.scale,
                landmarks=args.landmarks,
                boxes=args.box,
                visible_mark=args.watermark,
                suffix="." + args.format,
                video_suffix="." + args.video_format,
                quality=args.quality,
                overwrite=args.overwrite,
                device=args.device,
            )
        except (OSError, ValueError) as error:
            parser.exit(2, f"facehall enhance: error: {error}\n")
    return 0
