"""Measure YuNet face detection and alignment on synthetic CelebA scenes (Phase 4, D31).

Two parts; the results go to ``<out>/results.json`` and are printed as Markdown (local):

- ``grid``: the detection rate and landmark error (NME) of faces with a few pixels between the
  eyes, at several bicubic upscales before detection (val split);
- ``template``: for each scale, the YuNet template fitted on train scenes (32 / scale px between
  the eyes, detected at upsample = scale), and the alignment error on val scenes with
  ``CELEBA_TEMPLATE`` and with the fitted template, against the training jitter.

The scene construction and the measures are in ``facehall.eval.detection``.

    uv run python scripts/measure_detection.py [--grid-samples 300] [--fit 1500] [--val 800]
"""

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from facehall.align.detect import YuNet
from facehall.align.similarity import Points
from facehall.align.template import CELEBA_TEMPLATE, YUNET_TEMPLATES
from facehall.config import load_config
from facehall.data.config import DataConfig, JitterConfig
from facehall.eval.detection import (
    Row,
    SceneDegradation,
    alignment_error,
    fit_template,
    landmark_error,
    make_scenes,
    match,
    sample_rows,
    summarize,
)
from facehall.provenance import file_sha256, git_commit, package_versions
from facehall.weights import yunet_weights

HR_EYES = 32.0  # px between the eyes in the 128 px HR crop (CELEBA_TEMPLATE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("configs/data/celeba_128.yaml"))
    parser.add_argument("--weights", type=Path, default=None, help="default: $FACEHALL_WEIGHTS")
    parser.add_argument(
        "--parts", nargs="+", choices=("grid", "template"), default=["grid", "template"]
    )
    parser.add_argument("--grid-samples", type=int, default=300, help="val scenes per grid cell")
    parser.add_argument("--eyes", type=float, nargs="+", default=[4, 6, 8, 12])
    parser.add_argument("--upsample", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--scales", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--fit", type=int, default=1500, help="train scenes for the template")
    parser.add_argument("--val", type=int, default=800, help="val scenes for the alignment error")
    parser.add_argument("--score", type=float, default=0.6, help="detector score threshold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/detection"))
    return parser.parse_args(argv)


def scenes(
    rows: list[Row], eyes: float, degraded: bool, seed: int
) -> Iterator[tuple[npt.NDArray[np.uint8], Points]]:
    return make_scenes(rows, eyes, SceneDegradation() if degraded else None, seed)


def grid(args: argparse.Namespace, root: Path, weights: Path) -> list[dict[str, Any]]:
    rows = sample_rows(root, "val", args.grid_samples)
    results = []
    for eyes in args.eyes:
        for degraded in (False, True):
            cases = list(scenes(rows, eyes, degraded, args.seed))
            for upsample in args.upsample:
                detector = YuNet(weights, upsample=upsample, score_threshold=args.score)
                errors = []
                for scene, truth in cases:
                    face = match(detector.detect(scene), truth)
                    if face is not None:
                        errors.append(landmark_error(face.landmarks, truth))
                record = {
                    "eyes": eyes,
                    "degraded": degraded,
                    "upsample": upsample,
                    "scenes": len(cases),
                    "detected": len(errors) / len(cases),
                    "nme_mean": float(np.mean(errors)) if errors else None,
                    "nme_median": float(np.median(errors)) if errors else None,
                }
                logging.info("grid %s", record)
                results.append(record)
    return results


def template(
    args: argparse.Namespace, root: Path, weights: Path, jitter: JitterConfig
) -> list[dict[str, Any]]:
    train, val = sample_rows(root, "train", args.fit), sample_rows(root, "val", args.val)
    results = []
    for scale in args.scales:
        eyes = HR_EYES / scale
        detector = YuNet(weights, upsample=scale, score_threshold=args.score)
        pairs = []
        for scene, truth in scenes(train, eyes, False, args.seed):
            face = match(detector.detect(scene), truth)
            if face is not None:
                pairs.append((face.landmarks, truth))
        fitted = fit_template(pairs)
        logging.info("x%d: template fitted on %d of %d train scenes", scale, len(pairs), len(train))
        record: dict[str, Any] = {
            "scale": scale,
            "eyes": eyes,
            "fit_scenes": len(train),
            "fit_detected": len(pairs),
            "template": np.round(fitted, 5).tolist(),
            "template_in_code": YUNET_TEMPLATES[scale].tolist()
            if scale in YUNET_TEMPLATES
            else None,
            "val": {},
        }
        for degraded in (False, True):
            found = []
            for scene, truth in scenes(val, eyes, degraded, args.seed + 1):
                face = match(detector.detect(scene), truth)
                if face is not None:
                    found.append((face.landmarks, truth))
            key = "degraded" if degraded else "clean"
            record["val"][key] = {
                "scenes": len(val),
                "detected": len(found) / len(val),
                "celeba_template": summarize(
                    [alignment_error(d, t, CELEBA_TEMPLATE) for d, t in found], jitter
                ),
                "fitted_template": summarize(
                    [alignment_error(d, t, fitted) for d, t in found], jitter
                ),
            }
            logging.info("x%d %s: %s", scale, key, record["val"][key])
        results.append(record)
    return results


def render(results: dict[str, Any]) -> str:
    lines = []
    if "grid" in results:
        lines += [
            "| Eyes (px) | Upsample | Detected clean / degraded | NME clean / degraded |",
            "|---|---|---|---|",
        ]
        cells: dict[tuple[float, float], dict[bool, dict[str, Any]]] = {}
        for r in results["grid"]:
            cells.setdefault((r["eyes"], r["upsample"]), {})[r["degraded"]] = r
        for (eyes, upsample), pair in cells.items():
            rate = " / ".join(f"{pair[d]['detected']:.3f}" for d in (False, True))
            nme = " / ".join(
                "—" if pair[d]["nme_mean"] is None else f"{pair[d]['nme_mean']:.3f}"
                for d in (False, True)
            )
            lines.append(f"| {eyes:g} | ×{upsample:g} | {rate} | {nme} |")
        lines.append("")
    if "template" in results:
        lines += [
            "| Scale | Input | Template | Detected | Points (HR px) mean | Shift p50 / p95 | "
            "Rotation p50 / p95 | Scale p50 / p95 | Within jitter |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in results["template"]:
            for key, v in r["val"].items():
                for name in ("celeba_template", "fitted_template"):
                    s = v[name]
                    lines.append(
                        f"| ×{r['scale']} | {key} | {name.split('_')[0]} | {v['detected']:.3f} | "
                        f"{s['points_mean']:.2f} | {s['shift_p50']:.3f} / {s['shift_p95']:.3f} | "
                        f"{s['rotation_p50']:.2f} / {s['rotation_p95']:.2f} | "
                        f"{s['scale_p50']:.3f} / {s['scale_p95']:.3f} | {s['within_jitter']:.3f} |"
                    )
            lines.append(f"\n×{r['scale']} fitted template: {r['template']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = load_config(args.data, DataConfig)
    root = Path(data.root)
    weights = args.weights if args.weights is not None else yunet_weights()
    results: dict[str, Any] = {
        "settings": {k: str(v) for k, v in vars(args).items()},
        "provenance": {
            "git_commit": git_commit(),
            "weights": str(weights),
            "weights_sha256": file_sha256(weights),
            "versions": package_versions(["opencv-python-headless", "numpy"]),
        },
    }
    if "grid" in args.parts:
        results["grid"] = grid(args, root, weights)
    if "template" in args.parts:
        results["template"] = template(args, root, weights, data.jitter)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    text = render(results)
    (args.out / "results.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
