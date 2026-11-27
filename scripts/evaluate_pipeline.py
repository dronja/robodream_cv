"""Evaluate the inference pipeline end to end on synthetic CelebA scenes (PR 4.2).

For one checkpoint, scores the variants of ``facehall.eval.endtoend`` (bicubic, annotated
landmarks, detected landmarks, the model output without paste-back) on clean and degraded scenes
and writes ``results.json`` and ``results.md`` to ``<out>/<checkpoint label>/`` (local).

    uv run python scripts/evaluate_pipeline.py --ckpt runs/<id>/best.safetensors [--samples 1000]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2

from facehall.align.template import CELEBA_TEMPLATE
from facehall.config import load_config
from facehall.data.config import DataConfig
from facehall.eval.detection import SceneDegradation, sample_rows
from facehall.eval.endtoend import VARIANTS, Metric, evaluate, samples
from facehall.eval.metrics import ArcFaceOnnx, IdentitySimilarity, Lpips
from facehall.pipeline import Enhancer, PipelineConfig
from facehall.provenance import file_sha256, git_commit
from facehall.train.trainer import resolve_device
from facehall.weights import asset_dir

METRICS = ("psnr_y", "ssim_y", "lpips", "identity")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt", type=Path, required=True, help="restoration checkpoint or ONNX export"
    )
    parser.add_argument("--pipeline", type=Path, default=Path("configs/pipeline/default.yaml"))
    parser.add_argument("--data", type=Path, default=Path("configs/data/celeba_128.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-perceptual", action="store_true", help="skip LPIPS and ID")
    parser.add_argument("--out", type=Path, default=Path("outputs/pipeline_eval"))
    return parser.parse_args(argv)


def render(label: str, results: dict[str, Any]) -> str:
    names = [m for m in METRICS if m in results["clean"]["variants"]["bicubic"]]
    lines = [f"### {label}", ""]
    for inputs in ("clean", "degraded"):
        r = results[inputs]
        lines += [
            f"{inputs}: {r['scenes']} scenes, face detected {r['detected']:.3f}, "
            f"restored {r['restored']:.3f}",
            "",
            "| Variant | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for variant in VARIANTS:
            v = r["variants"][variant]
            lines.append(f"| {variant} | " + " | ".join(f"{v[n]:.4f}" for n in names) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cv2.setNumThreads(1)
    device = resolve_device(args.device)
    data = load_config(args.data, DataConfig)
    config = load_config(args.pipeline, PipelineConfig)
    enhancer = Enhancer.load(args.ckpt, config, str(device))
    metrics: dict[str, Metric] = {}
    if not args.no_perceptual:
        metrics["lpips"] = Lpips("alex", device)
        embedder = ArcFaceOnnx(asset_dir("buffalo_l") / "w600k_r50.onnx")
        metrics["identity"] = IdentitySimilarity(embedder, CELEBA_TEMPLATE, enhancer.hr_size)
    rows = sample_rows(Path(data.root), args.split, args.samples)
    label = f"{args.ckpt.parent.name}-{args.ckpt.stem}"
    results: dict[str, Any] = {
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": file_sha256(args.ckpt),
        "scale": enhancer.scale,
        "split": args.split,
        "pipeline": str(args.pipeline),
        "git_commit": git_commit(),
        "settings": {k: str(v) for k, v in vars(args).items()},
    }
    for inputs, degradation in (("clean", None), ("degraded", SceneDegradation())):
        scenes = samples(rows, enhancer.scale, enhancer.hr_size, degradation, args.seed)
        results[inputs] = evaluate(enhancer, scenes, metrics)
        logging.info("%s %s: %s", label, inputs, results[inputs])
    out = args.out / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    text = render(label, results)
    (out / "results.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
