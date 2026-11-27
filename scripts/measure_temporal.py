"""Measure flicker between video frames on synthetic CelebA clips (PR 4.5).

For one checkpoint, scores the variants of ``facehall.eval.temporal`` (bicubic, annotated
landmarks, detected landmarks, and face tracking settings) on every clip condition and writes
``results.json`` and ``results.md`` to ``<out>/<checkpoint label>/`` (local).

    uv run python scripts/measure_temporal.py --ckpt runs/<id>/best.safetensors \\
        [--clips 100 --frames 24 --tracking 1,0,2 0.5,0.1,2]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2

from facehall.config import load_config
from facehall.data.config import DataConfig
from facehall.eval.detection import sample_rows
from facehall.eval.temporal import CONDITIONS, Tracking, clips, evaluate_clip, mean_results
from facehall.pipeline import Enhancer, PipelineConfig
from facehall.provenance import file_sha256, git_commit
from facehall.train.trainer import resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True, help="restoration checkpoint")
    parser.add_argument("--pipeline", type=Path, default=Path("configs/pipeline/default.yaml"))
    parser.add_argument("--data", type=Path, default=Path("configs/data/celeba_128.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--clips", type=int, default=100)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument(
        "--tracking",
        nargs="*",
        type=Tracking.parse,
        default=[],
        metavar="ALPHA,BETA,HOLD[,LOW]",
        help="face tracker settings to compare (facehall.video.FaceTracker)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("outputs/temporal"))
    return parser.parse_args(argv)


def render(label: str, results: dict[str, Any]) -> str:
    lines = [f"### {label}", ""]
    for condition in results["conditions"]:
        r = results[condition]
        lines += [
            f"{condition}: {results['clips']} clips of {results['frames']} frames",
            "",
            "| Variant | Flicker (levels) | PSNR-Y (dB) | Restored |",
            "|---|---|---|---|",
        ]
        for variant, v in r.items():
            psnr = f"{v['psnr_y']:.2f}" if "psnr_y" in v else "—"
            restored = f"{v['restored']:.3f}" if "restored" in v else "—"
            lines.append(f"| {variant} | {v['flicker']:.3f} | {psnr} | {restored} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cv2.setNumThreads(1)
    device = resolve_device(args.device)
    data = load_config(args.data, DataConfig)
    config = load_config(args.pipeline, PipelineConfig)
    enhancer = Enhancer.from_checkpoint(args.ckpt, config, device)
    rows = sample_rows(Path(data.root), args.split, args.clips)
    label = f"{args.ckpt.parent.name}-{args.ckpt.stem}"
    results: dict[str, Any] = {
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": file_sha256(args.ckpt),
        "scale": enhancer.scale,
        "split": args.split,
        "pipeline": str(args.pipeline),
        "detector_threshold": config.detector.score_threshold,
        "track_threshold": config.video.track_threshold,
        "git_commit": git_commit(),
        "clips": len(rows),
        "frames": args.frames,
        "conditions": args.conditions,
        "settings": {k: str(v) for k, v in vars(args).items()},
    }
    for condition in args.conditions:
        start = time.perf_counter()
        per_clip = [
            evaluate_clip(enhancer, clip, args.tracking)
            for clip in clips(
                rows,
                scale=enhancer.scale,
                hr_size=enhancer.hr_size,
                frames=args.frames,
                condition=condition,
                seed=args.seed,
            )
        ]
        results[condition] = mean_results(per_clip)
        logging.info("%s %s: %.0f s", label, condition, time.perf_counter() - start)
    out = args.out / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    text = render(label, results)
    (out / "results.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
