"""Check an exported ONNX model against PyTorch on real crops and time both (PR 5.1).

Exports the checkpoint to ``<out>/<label>/model.onnx`` (local), then, for each evaluation
protocol of its scale (bicubic, light, medium and heavy levels by default), runs the first
``--samples`` LR crops of the protocol's split through PyTorch (CPU, float32) and ONNX Runtime
(CPU) and records their differences (``facehall.export.compare``). Then it times both on a
batch of the pipeline's size, and PyTorch on CUDA when there is a GPU. Writes ``results.json``
and ``results.md`` next to the model.

    uv run python scripts/measure_onnx.py --ckpt runs/<id>/best.safetensors [--samples 1000]
"""

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from facehall.config import load_config
from facehall.eval.config import EvalConfig, validate_eval
from facehall.eval.runner import protocol_dataset
from facehall.export import compare, export_checkpoint, run, session
from facehall.provenance import git_commit, package_versions
from facehall.train.checkpoint import load_checkpoint

LEVELS = ("", "_light", "_medium", "_heavy")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True, help="restoration checkpoint")
    parser.add_argument(
        "--protocols", type=Path, nargs="+", help="eval configs (default: the scale's levels)"
    )
    parser.add_argument("--split", help="override the protocols' split")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32, help="timing batch (pipeline default)")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("outputs/onnx"))
    return parser.parse_args(argv)


def median_ms(step: Callable[[], object], repeats: int) -> float:
    step()  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        times.append(time.perf_counter() - start)
    return 1000.0 * statistics.median(times)


def crops(config: EvalConfig, count: int) -> npt.NDArray[np.float32]:
    dataset = protocol_dataset(config)
    count = min(count, len(dataset))
    return np.stack([dataset[i]["lr"].numpy() for i in range(count)]).astype(np.float32)


def timings(model: torch.nn.Module, onnx: Any, lr: npt.NDArray[np.float32], repeats: int) -> Any:
    batch = torch.from_numpy(lr)
    result: dict[str, float] = {}
    with torch.no_grad():
        result["torch_cpu_ms"] = median_ms(lambda: model.cpu()(batch), repeats)
        result["onnxruntime_cpu_ms"] = median_ms(lambda: run(onnx, lr), repeats)
        if torch.cuda.is_available():
            gpu = model.cuda()
            on_gpu = batch.cuda()

            def step() -> None:
                gpu(on_gpu)
                torch.cuda.synchronize()

            result["torch_cuda_ms"] = median_ms(step, repeats)
            model.cpu()
    return result


def cuda_diff(model: torch.nn.Module, lr: npt.NDArray[np.float32]) -> float | None:
    """Max abs difference of PyTorch on CUDA from PyTorch on CPU, for scale."""
    if not torch.cuda.is_available():
        return None
    with torch.no_grad():
        cpu = model.cpu()(torch.from_numpy(lr)).numpy()
        gpu = model.cuda()(torch.from_numpy(lr).cuda()).float().cpu().numpy()
    model.cpu()
    return float(np.abs(cpu - gpu).max())


def render(label: str, results: dict[str, Any]) -> str:
    lines = [
        f"### {label}",
        "",
        f"{results['samples']} crops per protocol ({results['split'] or 'protocol split'})",
        "",
        "| Protocol | Max abs diff | Mean abs diff | Max level diff | Values changed | "
        "CUDA − CPU (PyTorch) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results["protocols"].items():
        cuda = f"{r['cuda_vs_cpu']:.1e}" if r["cuda_vs_cpu"] is not None else "—"
        lines.append(
            f"| {name} | {r['max_abs_diff']:.1e} | {r['mean_abs_diff']:.1e} | "
            f"{r['max_level_diff']:.0f} | {100 * r['levels_changed']:.4f}% | {cuda} |"
        )
    t = results["timing"]
    lines += [
        "",
        f"Batch of {results['batch']} crops, median of {results['repeats']}: "
        + ", ".join(f"{k.removesuffix('_ms')} {v:.1f} ms" for k, v in t.items()),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cv2.setNumThreads(1)
    label = f"{args.ckpt.parent.name}-{args.ckpt.stem}"
    out = args.out / label
    exported = export_checkpoint(args.ckpt, out / "model.onnx", overwrite=True)
    model, _ = load_checkpoint(args.ckpt, "cpu")
    onnx = session(exported.path)
    scale = exported.record["scale"]
    protocols = args.protocols or [Path(f"configs/eval/celeba128{s}_x{scale}.yaml") for s in LEVELS]
    results: dict[str, Any] = {
        "checkpoint": str(args.ckpt),
        "record": exported.record,
        "split": args.split,
        "samples": args.samples,
        "batch": args.batch,
        "repeats": args.repeats,
        "git_commit": git_commit(),
        "versions": package_versions(("torch", "onnxruntime", "onnxscript", "numpy")),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "protocols": {},
    }
    lr = np.zeros((0,), dtype=np.float32)
    for path in protocols:
        overrides = [f"split={args.split}"] if args.split else []
        config = load_config(path, EvalConfig, overrides)
        validate_eval(config)
        if config.scale != scale:
            raise ValueError(f"{path} is a x{config.scale} protocol, the model x{scale}")
        lr = crops(config, args.samples)
        differences = compare(model, onnx, lr)
        results["protocols"][config.name] = {
            **differences,
            "split": config.split,
            "crops": len(lr),
            "cuda_vs_cpu": cuda_diff(model, lr),
        }
        logging.info("%s: %s", config.name, differences)
    results["timing"] = timings(model, onnx, lr[: args.batch], args.repeats)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    text = render(label, results)
    (out / "results.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
