"""Latency and memory of the REST API in the service images, or outside them (PR 5.5).

    uv run python scripts/benchmark_images.py --image facehall:cpu --label docker-cpu
    uv run python scripts/benchmark_images.py --image facehall:gpu --gpu --label docker-gpu
    uv run python scripts/benchmark_images.py --image facehall:cpu --cpuset 0-3 --label docker-cpu-4
    uv run python scripts/benchmark_images.py --local .venv/bin/facehall --label local-cpu

For every scale and workload a fresh service is started, serving that scale only, and the script
measures:

- **start:** from starting the container (or the process) to the first healthy ``/health``,
  which includes loading and warming up the model;
- **latency** of ``POST /enhance`` as the client sees it, one request at a time: the first
  request, then the median, p95 and maximum of the others;
- **memory:** the service process's peak resident memory (``VmHWM``); for a container also the
  cgroup's peak, which counts the page cache too; with ``--gpu``, the process's GPU memory,
  polled with ``nvidia-smi`` every 0.1 s.

Workloads, from the synthetic CelebA scenes of ``compare_backends.py scenes`` (local, D19):

- ``single``: the single-face scenes (about 100 x 120 px at x4, 50 x 60 at x8), PNG output;
- ``mosaic``: the mosaics of 64 faces, each sent ``--repeat`` times, PNG output;
- ``frame``: ``--frames`` 1920 x 1080 frames with 8 faces (single scenes on a grey canvas, sent
  as JPEG with quality 95), each sent once with PNG and once with JPEG output.

Writes ``<out>/<label>.json`` and prints a table.
"""

import argparse
import io
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from facehall.provenance import git_commit
from facehall.weights import yunet_weights

FRAME = (1080, 1920)
FACES_PER_FRAME = 8


@dataclass
class Workload:
    name: str
    uploads: list[bytes]
    query: str = ""
    repeat: int = 1


@dataclass
class Service:
    """A running service: its address, the host pid of its process and how to stop it."""

    url: str
    pid: int
    stop: Callable[[], None]
    cgroup_peak: Callable[[], int | None] = field(default=lambda: None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", help="service image, e.g. facehall:cpu")
    target.add_argument("--local", type=Path, help="a facehall executable, run without Docker")
    parser.add_argument("--label", required=True, help="name of the result file")
    parser.add_argument("--gpu", action="store_true", help="serve on cuda (--gpus all)")
    parser.add_argument("--cpuset", help="CPUs of the container, e.g. 0-3 (docker --cpuset-cpus)")
    parser.add_argument("--scales", default="4,8")
    parser.add_argument("--workloads", default="single,mosaic,frame")
    parser.add_argument("--scenes", type=Path, default=Path("outputs/backends"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--repeat", type=int, default=3, help="times each mosaic is sent")
    parser.add_argument("--frames", type=int, default=5, help="frames, each sent once per output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--docker", default="docker", help="docker command (default: docker)")
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark"))
    return parser.parse_args(argv)


def make_frame(
    scenes: list[npt.NDArray[np.uint8]], rng: np.random.Generator
) -> npt.NDArray[np.uint8]:
    """The scenes on a grey 1920 x 1080 canvas with a gentle gradient, one per cell of a 2 x 4
    grid, at a random position inside the cell."""
    h, w = FRAME
    y, x = np.mgrid[0:h, 0:w]
    gray = (110 + 30 * x / w + 20 * y / h).astype(np.uint8)
    canvas: npt.NDArray[np.uint8] = np.repeat(gray[..., None], 3, axis=2)
    rows, cols = 2, len(scenes) // 2
    cell_h, cell_w = h // rows, w // cols
    for i, scene in enumerate(scenes):
        sh, sw = scene.shape[:2]
        top = (i // cols) * cell_h + int(rng.integers(0, cell_h - sh))
        left = (i % cols) * cell_w + int(rng.integers(0, cell_w - sw))
        canvas[top : top + sh, left : left + sw] = scene
    return canvas


def encode(image: npt.NDArray[np.uint8], kind: str, quality: int = 95) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, kind, quality=quality)
    return buffer.getvalue()


def workloads(args: argparse.Namespace, scale: int) -> list[Workload]:
    folder = args.scenes / f"scenes_x{scale}"
    singles = sorted(folder.glob("single_*.png"))
    mosaics = sorted(folder.glob("mosaic_*.png"))
    if not singles or not mosaics:
        raise SystemExit(f"{folder}: no scenes; run scripts/compare_backends.py scenes first")
    wanted = args.workloads.split(",")
    out: list[Workload] = []
    if "single" in wanted:
        out.append(Workload("single", [p.read_bytes() for p in singles]))
    if "mosaic" in wanted:
        out.append(Workload("mosaic", [p.read_bytes() for p in mosaics], repeat=args.repeat))
    if "frame" in wanted:
        rng = np.random.default_rng(args.seed)
        faces = [np.asarray(Image.open(p).convert("RGB")) for p in singles]
        frames = [
            encode(make_frame(faces[k * FACES_PER_FRAME : (k + 1) * FACES_PER_FRAME], rng), "JPEG")
            for k in range(args.frames)
        ]
        out.append(Workload("frame", frames))
        out.append(Workload("frame", frames, query="format=jpeg"))
    return out


def docker(prefix: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*prefix.split(), *args], capture_output=True, text=True, check=check)


def start_container(args: argparse.Namespace, scale: int, port: int) -> Service:
    name = f"facehall-bench-{os.getpid()}-{port}"
    command = ["run", "-d", "--name", name, "--read-only", "--tmpfs", "/tmp"]
    command += ["-p", f"127.0.0.1:{port}:8000"]
    command += ["-v", f"{args.models.resolve()}:/models:ro"]
    command += ["-v", f"{yunet_weights().parent.resolve()}:/weights/yunet:ro"]
    command += ["--gpus", "all"] if args.gpu else []
    command += ["--cpuset-cpus", args.cpuset] if args.cpuset else []
    command += [args.image, "facehall", "serve", "--host", "0.0.0.0", "--port", "8000"]
    command += ["--device", "cuda" if args.gpu else "cpu", "--scale", str(scale)]
    docker(args.docker, *command)
    pid = int(docker(args.docker, "inspect", "-f", "{{.State.Pid}}", name).stdout)

    def cgroup_peak() -> int | None:
        result = docker(args.docker, "exec", name, "cat", "/sys/fs/cgroup/memory.peak", check=False)
        return int(result.stdout) if result.returncode == 0 else None

    def stop() -> None:
        docker(args.docker, "rm", "-f", name, check=False)

    return Service(f"http://127.0.0.1:{port}", pid, stop, cgroup_peak)


def start_local(args: argparse.Namespace, scale: int, port: int) -> Service:
    command = [str(args.local), "serve", "--port", str(port), "--scale", str(scale)]
    command += ["--device", "cuda" if args.gpu else "cpu"]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop() -> None:
        process.terminate()
        process.wait(timeout=30)

    return Service(f"http://127.0.0.1:{port}", process.pid, stop)


def post(url: str, data: bytes) -> tuple[float, int, dict[str, str], int]:
    """Seconds, status, headers and the size of the answer."""
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=300) as r:
            body = r.read()
            status, headers = r.status, dict(r.headers.items())
    except urllib.error.HTTPError as error:
        body, status, headers = error.read(), error.code, dict(error.headers.items())
    return time.perf_counter() - start, status, headers, len(body)


def wait_healthy(service: Service, timeout: float = 180.0) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            with urllib.request.urlopen(service.url + "/health", timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - start
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.05)
    raise SystemExit(f"{service.url}: not healthy after {timeout:.0f} s")


def resident_peak(pid: int) -> int | None:
    """``VmHWM`` of a process, in bytes."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


@contextmanager
def gpu_memory(pid: int, enabled: bool) -> Iterator[list[int]]:
    """Polls the GPU memory of ``pid`` (MiB) while the block runs; yields the list of samples."""
    samples: list[int] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == str(pid):
                    samples.append(int(parts[1].split()[0]))
            stop.wait(0.1)

    thread = threading.Thread(target=poll, daemon=True)
    if enabled:
        thread.start()
    try:
        yield samples
    finally:
        stop.set()
        if enabled:
            thread.join()


def restored_faces(summary: str) -> int:
    """The restored faces of an ``X-Facehall-Faces`` header ("8 faces, 7 restored, …")."""
    match = re.search(r"(\d+) restored", summary)
    return int(match.group(1)) if match else 0


def summarize(seconds: list[float]) -> dict[str, float]:
    first, rest = seconds[0], sorted(seconds[1:]) or seconds
    p95 = rest[min(len(rest) - 1, round(0.95 * (len(rest) - 1)))]
    return {
        "first": first,
        "median": statistics.median(rest),
        "p95": p95,
        "max": rest[-1],
        "n": len(seconds),
    }


def run_workload(args: argparse.Namespace, scale: int, work: Workload, port: int) -> dict[str, Any]:
    started = time.perf_counter()
    service = (start_local if args.local else start_container)(args, scale, port)
    try:
        wait_healthy(service)
        start = time.perf_counter() - started
        url = f"{service.url}/enhance?scale={scale}" + (f"&{work.query}" if work.query else "")
        seconds: list[float] = []
        faces: dict[str, int] = {}
        restored: list[int] = []
        sizes: list[int] = []
        with gpu_memory(service.pid, args.gpu) as gpu:
            for _ in range(work.repeat):
                for upload in work.uploads:
                    elapsed, status, headers, size = post(url, upload)
                    if status != 200:
                        raise SystemExit(f"{work.name}: {status} {headers}")
                    seconds.append(elapsed)
                    sizes.append(size)
                    count = {k.lower(): v for k, v in headers.items()}["x-facehall-faces"]
                    faces[count] = faces.get(count, 0) + 1
                    restored.append(restored_faces(count))
        return {
            "workload": work.name,
            "output": "jpeg" if "jpeg" in work.query else "png",
            "scale": scale,
            "start_seconds": start,
            "latency_seconds": summarize(seconds),
            "faces": faces,
            "restored_median": statistics.median(restored),
            "upload_bytes": statistics.median(len(u) for u in work.uploads),
            "answer_bytes": statistics.median(sizes),
            "resident_peak_bytes": resident_peak(service.pid),
            "cgroup_peak_bytes": service.cgroup_peak(),
            "gpu_peak_mib": max(gpu) if gpu else None,
        }
    finally:
        service.stop()


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Scale | Workload | Output | Start (s) | First (s) | Median (s) | p95 (s) | "
        "Peak RSS (GB) | GPU (GB) | Restored faces |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lat = r["latency_seconds"]
        rss = r["resident_peak_bytes"]
        gpu = r["gpu_peak_mib"]
        lines.append(
            f"| x{r['scale']} | {r['workload']} | {r['output']} | {r['start_seconds']:.1f} | "
            f"{lat['first']:.3f} | {lat['median']:.3f} | {lat['p95']:.3f} | "
            f"{rss / 2**30 if rss else float('nan'):.2f} | "
            f"{gpu / 1024 if gpu else float('nan'):.2f} | "
            f"{r['restored_median']:g} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    port = args.port
    for scale in map(int, args.scales.split(",")):
        for work in workloads(args, scale):
            rows.append(run_workload(args, scale, work, port))
            port += 1
            print(table(rows[-1:]).splitlines()[-1], flush=True)
    target: dict[str, Any] = {"image": args.image} if args.image else {"local": str(args.local)}
    if args.image:
        inspect = docker(args.docker, "image", "inspect", "-f", "{{.Id}}", args.image)
        target["image_id"] = inspect.stdout.strip()
    result = {
        "label": args.label,
        "target": target,
        "gpu": args.gpu,
        "cpuset": args.cpuset,
        "git_commit": git_commit(),
        "machine": {"cpu": platform.processor() or platform.machine(), "cpus": os.cpu_count()},
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
