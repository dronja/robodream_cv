"""Measure YuNet face detection and alignment on synthetic CelebA scenes (Phase 4, D31).

The results go to ``<out>/results.json`` and are printed as Markdown (local). Parts:

- ``grid``: the detection rate and landmark error (NME) of faces with a few pixels between the
  eyes, at several bicubic upscales before detection (val split);
- ``template``: for each scale, the YuNet template fitted on train scenes (32 / scale px between
  the eyes, detected at upsample = scale), and the alignment error on val scenes with
  ``CELEBA_TEMPLATE`` and with the fitted template, against the training jitter;
- ``box``: landmarks from a face box (``YUNET_BOX_LANDMARKS``), fitted on train scenes, and
  their alignment error on val scenes against that of the detected landmarks (PR 4.4);
- ``wider``: on real photos (the WIDER FACE ``val`` part of D20, with the official face boxes),
  how many detections of a size the pipeline restores are faces, per score threshold (PR 4.4).
  YuNet was trained on WIDER FACE training photos, these included, so the numbers are
  optimistic.

The scene construction and the measures are in ``facehall.eval.detection``.

    uv run python scripts/measure_detection.py [--grid-samples 300] [--fit 1500] [--val 800]
"""

import argparse
import json
import logging
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from facehall.align.detect import YuNet, box_landmarks
from facehall.align.similarity import Points
from facehall.align.template import CELEBA_TEMPLATE, YUNET_BOX_LANDMARKS, YUNET_TEMPLATES
from facehall.config import load_config
from facehall.data.config import DataConfig, JitterConfig
from facehall.data.wider import (
    IMAGE_PREFIX,
    IMAGES_ARCHIVE,
    OFFICIAL_FILE,
    SPLIT_ARCHIVE,
    TEST_FRACTION,
    parse_official,
    split_of,
)
from facehall.eval.detection import (
    Detection,
    Row,
    SceneDegradation,
    alignment_error,
    box_lr_scale,
    fit_box_landmarks,
    fit_template,
    landmark_error,
    make_scenes,
    match,
    precision_table,
    sample_rows,
    score_detections,
    summarize,
)
from facehall.pipeline import PipelineConfig
from facehall.provenance import file_sha256, git_commit, package_versions
from facehall.weights import yunet_weights

HR_EYES = 32.0  # px between the eyes in the 128 px HR crop (CELEBA_TEMPLATE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("configs/data/celeba_128.yaml"))
    parser.add_argument("--weights", type=Path, default=None, help="default: $FACEHALL_WEIGHTS")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("grid", "template", "box", "wider"),
        default=["grid", "template"],
    )
    parser.add_argument("--grid-samples", type=int, default=300, help="val scenes per grid cell")
    parser.add_argument("--eyes", type=float, nargs="+", default=[4, 6, 8, 12])
    parser.add_argument("--upsample", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--scales", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--fit", type=int, default=1500, help="train scenes for the template")
    parser.add_argument("--val", type=int, default=800, help="val scenes for the alignment error")
    parser.add_argument("--score", type=float, default=0.6, help="detector score threshold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wider", type=Path, default=None, help="WIDER FACE raw archives")
    parser.add_argument("--photos", type=int, default=200, help="WIDER val photos")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
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


def box(
    args: argparse.Namespace, root: Path, weights: Path, jitter: JitterConfig
) -> list[dict[str, Any]]:
    """Landmarks from a face box (the fallback for a face the detector misses): fitted on train
    scenes from YuNet's own boxes; on val, the alignment error of landmarks from the detected
    box against that of the detected landmarks."""
    train, val = sample_rows(root, "train", args.fit), sample_rows(root, "val", args.val)
    results = []
    for scale in args.scales:
        eyes = HR_EYES / scale
        detector = YuNet(weights, upsample=scale, score_threshold=args.score)
        faces = []
        for scene, truth in scenes(train, eyes, False, args.seed):
            face = match(detector.detect(scene), truth)
            if face is not None:
                faces.append(face)
        fitted = fit_box_landmarks(faces)
        record: dict[str, Any] = {
            "scale": scale,
            "fit_detected": len(faces),
            "box_landmarks": np.round(fitted, 5).tolist(),
            "box_landmarks_in_code": YUNET_BOX_LANDMARKS[scale].tolist()
            if scale in YUNET_BOX_LANDMARKS
            else None,
            "val": {},
        }
        template = YUNET_TEMPLATES[scale]
        for degraded in (False, True):
            found = []
            for scene, truth in scenes(val, eyes, degraded, args.seed + 1):
                face = match(detector.detect(scene), truth)
                if face is not None:
                    found.append((face, truth))
            key = "degraded" if degraded else "clean"
            record["val"][key] = {
                "detected": len(found) / len(val),
                "landmarks": summarize(
                    [alignment_error(f.landmarks, t, template) for f, t in found], jitter
                ),
                "box": summarize(
                    [alignment_error(box_landmarks(f.box, fitted), t, template) for f, t in found],
                    jitter,
                ),
            }
            logging.info("box x%d %s: %s", scale, key, record["val"][key])
        results.append(record)
    return results


def wider(args: argparse.Namespace, weights: Path, root: Path) -> list[dict[str, Any]]:
    raw = args.wider if args.wider is not None else root.parent / "wider_face" / "raw"
    with zipfile.ZipFile(raw / SPLIT_ARCHIVE) as archive:
        official = parse_official(archive.read(OFFICIAL_FILE).decode())
    names = sorted(n for n in official if split_of(n, TEST_FRACTION) == "val")
    rng = np.random.default_rng(args.seed)
    picked = [names[i] for i in sorted(rng.choice(len(names), args.photos, replace=False))]
    pipeline = PipelineConfig()
    limits = (1.0 / pipeline.max_downscale, pipeline.max_upscale)
    results = []
    with zipfile.ZipFile(raw / IMAGES_ARCHIVE) as archive:
        for scale in args.scales:
            detector = YuNet(weights, upsample=scale, score_threshold=min(args.thresholds))
            lr_size = int(HR_EYES * 4) // scale
            detections: list[Detection] = []
            recalled: list[float | None] = []
            for name in picked:
                data = np.frombuffer(archive.read(IMAGE_PREFIX + name), dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"{name}: cannot decode the photo")
                image = np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)
                truth = [f.box for f in official[name] if f.box[2] > 0 and f.box[3] > 0]
                found, best = score_detections(
                    detector.detect(image), truth, YUNET_TEMPLATES[scale], lr_size
                )
                detections += found
                low, high = limits
                recalled += [
                    s
                    for box, s in zip(truth, best, strict=True)
                    if low <= box_lr_scale(box[2], lr_size) <= high
                ]
            table = precision_table(detections, recalled, args.thresholds, len(picked), limits)
            record = {"scale": scale, "photos": len(picked), "faces": len(recalled), "rows": table}
            logging.info("wider x%d: %s", scale, record)
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
    if "box" in results:
        lines += [
            "",
            "| Scale | Input | Landmarks from | Shift p50 / p95 | Rotation p50 / p95 | "
            "Scale p50 / p95 | Within jitter |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in results["box"]:
            for key, v in r["val"].items():
                for name in ("landmarks", "box"):
                    s = v[name]
                    lines.append(
                        f"| ×{r['scale']} | {key} | {name} | "
                        f"{s['shift_p50']:.3f} / {s['shift_p95']:.3f} | "
                        f"{s['rotation_p50']:.2f} / {s['rotation_p95']:.2f} | "
                        f"{s['scale_p50']:.3f} / {s['scale_p95']:.3f} | {s['within_jitter']:.3f} |"
                    )
            lines.append(f"\n×{r['scale']} box landmarks: {r['box_landmarks']}")
    if "wider" in results:
        lines += [
            "",
            "| Scale | Threshold | Restorable detections | Faces among them | False per photo | "
            "Faces of restorable size found |",
            "|---|---|---|---|---|---|",
        ]
        for r in results["wider"]:
            for row in r["rows"]:
                precision = "—" if row["precision"] is None else f"{row['precision']:.3f}"
                recall = "—" if row["recall"] is None else f"{row['recall']:.3f}"
                lines.append(
                    f"| ×{r['scale']} | {row['threshold']:g} | {row['restorable']} | {precision} | "
                    f"{row['false_per_photo']:.2f} | {recall} |"
                )
            lines.append(
                f"\n×{r['scale']}: {r['photos']} photos, {r['faces']} faces of restorable size\n"
            )
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
    if "box" in args.parts:
        results["box"] = box(args, root, weights, data.jitter)
    if "wider" in args.parts:
        results["wider"] = wider(args, weights, root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    text = render(results)
    (args.out / "results.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
