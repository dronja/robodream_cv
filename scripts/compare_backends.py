"""Compare restoration backends end to end: same outputs, and their speed (PR 5.2).

Two steps, so the second runs where PyTorch is not installed (an ONNX model, D34):

    # 1. synthetic CelebA scenes (local, D19): single faces and mosaics of many faces
    uv run python scripts/compare_backends.py scenes --scale 4 --out outputs/backends
    # 2. the pipeline on every scene with one model and device; with --reference, the outputs
    #    are compared pixel by pixel with those of another run
    uv run python scripts/compare_backends.py run --model models/x4.onnx --device cpu \\
        --scenes outputs/backends/scenes_x4 --label onnx-cpu \\
        --reference outputs/backends/runs_x4/torch-cpu

A run writes the restored images (PNG, unmarked, local) and ``run.json``: the faces per scene,
wall times per stage (``Enhanced.seconds``), the peak resident memory, and the comparison.
"""

import argparse
import json
import logging
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from facehall.config import load_config
from facehall.pipeline import Enhancer, PipelineConfig
from facehall.provenance import git_commit, package_versions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    scenes = steps.add_parser("scenes", help="write the synthetic scenes")
    scenes.add_argument("--scale", type=int, choices=(4, 8), required=True)
    scenes.add_argument("--data", type=Path, default=Path("configs/data/celeba_128.yaml"))
    scenes.add_argument("--split", default="val")
    scenes.add_argument("--single", type=int, default=100, help="single-face scenes per input")
    scenes.add_argument("--mosaics", type=int, default=5, help="mosaics of --grid² scenes")
    scenes.add_argument("--grid", type=int, default=8)
    scenes.add_argument("--seed", type=int, default=0)
    scenes.add_argument("--out", type=Path, default=Path("outputs/backends"))
    run = steps.add_parser("run", help="run the pipeline on the scenes")
    run.add_argument("--model", type=Path, required=True, help="checkpoint or ONNX export")
    run.add_argument("--device", default="auto")
    run.add_argument("--scenes", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--reference", type=Path, help="another run to compare the outputs with")
    run.add_argument("--pipeline", type=Path, default=Path("configs/pipeline/default.yaml"))
    return parser.parse_args(argv)


def write_scenes(args: argparse.Namespace) -> Path:
    from facehall.align.template import CELEBA_TEMPLATE
    from facehall.data.config import DataConfig
    from facehall.eval.detection import SceneDegradation, make_scenes, sample_rows

    data = load_config(args.data, DataConfig)
    eyes = float(CELEBA_TEMPLATE[1, 0] - CELEBA_TEMPLATE[0, 0]) * 128 / args.scale
    half, per = args.single // 2, args.grid**2
    rows = sample_rows(Path(data.root), args.split, half + args.mosaics * per)
    out: Path = args.out / f"scenes_x{args.scale}"
    out.mkdir(parents=True, exist_ok=True)
    for kind, degradation in (("clean", None), ("degraded", SceneDegradation())):
        pool: list[npt.NDArray[np.uint8]] = []
        for i, (scene, _) in enumerate(make_scenes(rows, eyes, degradation, args.seed)):
            if i < half:
                Image.fromarray(scene).save(out / f"single_{kind}_{i:04d}.png")
            else:
                pool.append(scene)
        for m in range(args.mosaics):  # even: clean, odd: degraded
            if m % 2 != (kind == "degraded"):
                continue
            tiles = pool[m * per : (m + 1) * per]
            side = max(max(t.shape[:2]) for t in tiles)
            lines = [
                np.concatenate(
                    [pad(t, side) for t in tiles[r * args.grid : (r + 1) * args.grid]], 1
                )
                for r in range(args.grid)
            ]
            Image.fromarray(np.concatenate(lines, 0)).save(out / f"mosaic_{m:02d}_{kind}.png")
    return out


def pad(scene: npt.NDArray[np.uint8], side: int) -> npt.NDArray[np.uint8]:
    """``scene`` in the middle of a ``side`` square filled with its border colour."""
    h, w = scene.shape[:2]
    out = np.empty((side, side, 3), dtype=np.uint8)
    out[:] = scene[0, 0]
    top, left = (side - h) // 2, (side - w) // 2
    out[top : top + h, left : left + w] = scene
    return out


def run_backend(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.pipeline, PipelineConfig)
    enhancer = Enhancer.load(args.model, config, args.device)
    scenes = sorted(args.scenes.glob("*.png"))
    out = args.scenes.parent / f"runs_x{enhancer.scale}" / args.label
    out.mkdir(parents=True, exist_ok=True)
    warm = np.asarray(Image.open(scenes[0]).convert("RGB"))
    start = time.perf_counter()
    enhancer.enhance(warm)  # first call: detector and session start-up
    first = time.perf_counter() - start
    per_scene: dict[str, Any] = {}
    for path in scenes:
        image = np.asarray(Image.open(path).convert("RGB"))
        start = time.perf_counter()
        result = enhancer.enhance(image)
        total = time.perf_counter() - start
        Image.fromarray(result.image).save(out / path.name, compress_level=1)
        per_scene[path.name] = {
            "faces": len(result.faces),
            "restored": sum(f.restored for f in result.faces),
            "seconds": {**result.seconds, "total": total},
        }
    results: dict[str, Any] = {
        "model": str(args.model),
        "device": args.device,
        "runs_on": enhancer.device,
        "info": enhancer.info,
        "git_commit": git_commit(),
        "versions": package_versions(("torch", "onnxruntime", "onnxruntime-gpu", "numpy")),
        "machine": platform.processor() or platform.machine(),
        "first_call_s": first,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "summary": summarize(per_scene),
        "scenes": per_scene,
    }
    if args.reference is not None:
        results["comparison"] = compare(out, args.reference, per_scene)
    (out / "run.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def summarize(per_scene: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for kind in ("single", "mosaic"):
        runs = [r for name, r in per_scene.items() if name.startswith(kind)]
        if not runs:
            continue
        faces = sum(r["restored"] for r in runs)
        summary[kind] = {
            "scenes": len(runs),
            "faces": sum(r["faces"] for r in runs),
            "restored": faces,
            **{
                f"median_{stage}_ms": 1000 * statistics.median(r["seconds"][stage] for r in runs)
                for stage in ("detect", "restore", "paste", "total")
            },
            "restore_ms_per_face": 1000
            * sum(r["seconds"]["restore"] for r in runs)
            / max(faces, 1),
        }
    return summary


def compare(out: Path, reference: Path, per_scene: dict[str, Any]) -> dict[str, Any]:
    """Pixel differences from the outputs of the reference run (8-bit levels)."""
    ref = json.loads((reference / "run.json").read_text())["scenes"]
    worst, changed, values, same_faces = 0, 0, 0, 0
    for name, r in per_scene.items():
        a = np.asarray(Image.open(out / name), dtype=np.int16)
        b = np.asarray(Image.open(reference / name), dtype=np.int16)
        diff = np.abs(a - b)
        worst = max(worst, int(diff.max()))
        changed += int((diff > 0).sum())
        values += diff.size
        same_faces += int(ref[name]["restored"] == r["restored"])
    return {
        "reference": str(reference),
        "max_level_diff": worst,
        "values_changed": changed / values,
        "scenes_with_the_same_restored_faces": same_faces / len(per_scene),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.step == "scenes":
        print(write_scenes(args))
        return 0
    results = run_backend(args)
    print(json.dumps({k: results[k] for k in ("summary", "peak_rss_mb", "first_call_s")}, indent=1))
    if "comparison" in results:
        print(json.dumps(results["comparison"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
