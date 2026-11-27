"""Start a service image (docker/, PR 5.4) and check it end to end.

    uv run python scripts/smoke_image.py --image facehall:cpu --models models [--gpu]
    uv run python scripts/smoke_image.py --image facehall:cpu --tiny   # a tiny random model (CI)

Runs the container the way its Dockerfile documents (read-only root, ``/tmp`` in memory, the
port on 127.0.0.1, the models and YuNet mounted read-only), waits for ``/health``, then checks:

- ``/version``: the expected scales, the ONNX runtime and device;
- ``POST /enhance``: a marked answer (headers and record) for a synthetic photo with a face box;
- a bad request: a readable 415;
- the container: a non-root user, no PyTorch, nothing written outside ``/tmp``;
- ``docker logs``: no query string (face boxes) and no telemetry messages;
- ONNX Runtime telemetry is off (``ORT_DISABLE_TELEMETRY=1``).

Exits non-zero on the first failed check and always removes the container.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from facehall.marking import NOTICE, read_record
from facehall.weights import yunet_weights

BOX = "16,16,32,32"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True, help="image tag, e.g. facehall:cpu")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--models", type=Path, help="folder with x4.onnx (and x8.onnx)")
    source.add_argument("--tiny", action="store_true", help="export a tiny random x4 model")
    parser.add_argument("--gpu", action="store_true", help="run with --gpus all, expect CUDA")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--docker", default="docker", help="docker command (default: docker)")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for start")
    return parser.parse_args(argv)


def tiny_model(folder: Path) -> Path:
    from facehall.export import export_checkpoint
    from facehall.models import build_model
    from facehall.train.checkpoint import save_checkpoint
    from facehall.train.config import experiment_to_dict, load_experiment

    config = load_experiment(
        "configs/train/srresnet_x4.yaml", ["model.residual_blocks=1", "model.channels=8"]
    )
    model = build_model(config.model, config.scale, config.data.hr_size // config.scale)
    checkpoint = folder / "tiny" / "best.safetensors"
    checkpoint.parent.mkdir(parents=True)
    save_checkpoint(checkpoint, model, {"config": experiment_to_dict(config), "run": "tiny"})
    path = export_checkpoint(checkpoint, folder / "x4.onnx").path
    folder.chmod(0o755)  # the container runs as another user (uid 10001)
    path.chmod(0o644)
    return path


def docker(prefix: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*prefix.split(), *args], capture_output=True, text=True, check=check)


def request(url: str, data: bytes | None = None) -> tuple[int, dict[str, str], bytes]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=60) as r:
            return r.status, dict(r.headers.items()), r.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()


def photo() -> bytes:
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:64, 0:64]
    base = 128 + 60 * np.sin(x / 5.0)[..., None] * np.cos(y / 7.0)[..., None]
    pixels = np.clip(base + rng.normal(0, 5, (64, 64, 3)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "PNG")
    return buffer.getvalue()


def check(condition: bool, what: str) -> None:
    print(("ok    " if condition else "FAIL  ") + what)
    if not condition:
        raise SystemExit(1)


def run(args: argparse.Namespace, models: Path, scales: list[int]) -> dict[str, Any]:
    weights = yunet_weights().parent
    check(weights.is_dir(), f"YuNet weights in {weights} (scripts/download_weights.py yunet)")
    name = f"facehall-smoke-{os.getpid()}"
    options = ["--read-only", "--tmpfs", "/tmp"]
    options += [
        "-v",
        f"{models.resolve()}:/models:ro",
        "-v",
        f"{weights.resolve()}:/weights/yunet:ro",
    ]
    options += ["--gpus", "all"] if args.gpu else []
    # What the container runtime itself adds to the root (the NVIDIA runtime injects the driver
    # files and rebuilds the ldconfig cache): the same container running `true`.
    docker(args.docker, "run", "--name", f"{name}-base", *options, args.image, "true")
    try:
        baseline = set(docker(args.docker, "diff", f"{name}-base").stdout.splitlines())
    finally:
        docker(args.docker, "rm", "-f", f"{name}-base", check=False)
    command = ["run", "-d", "--name", name, *options, "-p", f"127.0.0.1:{args.port}:8000"]
    command += [args.image, "facehall", "serve", "--host", "0.0.0.0", "--port", "8000"]
    command += ["--device", "cuda" if args.gpu else "cpu"]
    command += [f"--scale={s}" for s in scales]
    docker(args.docker, *command)
    base = f"http://127.0.0.1:{args.port}"
    try:
        started = time.monotonic()
        while True:
            try:
                status, _, _ = request(base + "/health")
                if status == 200:
                    break
            except (urllib.error.URLError, ConnectionError):
                pass
            state = docker(args.docker, "inspect", "-f", "{{.State.Running}}", name).stdout
            if state.strip() != "true" or time.monotonic() - started > args.timeout:
                print(docker(args.docker, "logs", name, check=False).stderr[-3000:])
                check(False, "the service started")
            time.sleep(0.5)
        check(True, f"the service started in {time.monotonic() - started:.1f} s")
        _, _, body = request(base + "/version")
        version = json.loads(body)
        check(sorted(map(int, version["models"])) == scales, f"serves x{scales}")
        device = "cuda" if args.gpu else "cpu"
        for model in version["models"].values():
            check(model["device"] == device and model["format"] == "onnx", f"{model['runtime']}")
        upload = photo()
        status, headers, body = request(f"{base}/enhance?scale={scales[0]}&box={BOX}", upload)
        headers = {k.lower(): v for k, v in headers.items()}
        check(status == 200 and headers.get("content-type") == "image/png", "POST /enhance: 200")
        check(headers.get("x-ai-enhanced") == "true", "X-AI-Enhanced header")
        check(headers.get("x-facehall-notice") == NOTICE, "X-Facehall-Notice header")
        rec = read_record(io.BytesIO(body))
        check(rec is not None and rec["input"]["faces_from"] == "boxes", "marked output record")
        status, _, body = request(f"{base}/enhance", b"not an image")
        check(status == 415 and b"not an image" in body, "a bad request: 415")
        user = docker(args.docker, "exec", name, "id", "-u").stdout.strip()
        check(user != "0", f"runs as a non-root user (uid {user})")
        probe = "import importlib.util as u; print(u.find_spec('torch') is None)"
        torch_free = docker(args.docker, "exec", name, "python", "-c", probe).stdout.strip()
        check(torch_free == "True", "no PyTorch in the image")
        changes = set(docker(args.docker, "diff", name).stdout.splitlines()) - baseline
        changed = sorted(line.split(maxsplit=1)[-1] for line in changes)
        outside = [path for path in changed if not path.startswith("/tmp")]
        check(not outside, f"nothing written outside /tmp {outside[:5] if outside else ''}".strip())
        logs = docker(args.docker, "logs", name, check=False)
        text = logs.stdout + logs.stderr
        check(BOX not in text and "box=" not in text, "no face boxes in the logs")
        env = docker(args.docker, "exec", name, "printenv", "ORT_DISABLE_TELEMETRY").stdout
        check(env.strip() == "1", "ONNX Runtime telemetry off")
        check("telemetry" not in text.lower(), "no ONNX Runtime telemetry messages in the logs")
        size = docker(args.docker, "image", "inspect", "-f", "{{.Size}}", args.image).stdout
        return {"image": args.image, "bytes": int(size), "version": version}
    finally:
        docker(args.docker, "rm", "-f", name, check=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory() as folder:
        models = tiny_model(Path(folder)).parent if args.tiny else args.models
        scales = [s for s in (4, 8) if (models / f"x{s}.onnx").is_file()]
        check(bool(scales), f"ONNX models in {models}")
        result = run(args, models, scales)
    print(f"image {result['image']}: {result['bytes'] / 1e6:.0f} MB of compressed content")
    return 0


if __name__ == "__main__":
    sys.exit(main())
