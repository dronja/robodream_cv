"""REST API (PR 5.3): endpoints, marks, limits, the queue and privacy.

A numpy stand-in for the model and a stub detector, so no GPU, weights or dataset; the CLI and
the torch-free tests use a tiny ONNX export.
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from facehall import marking
from facehall.align import CELEBA_TEMPLATE, Face
from facehall.cli import main
from facehall.export import export_checkpoint
from facehall.models import build_model
from facehall.pipeline import Enhancer, PipelineConfig, ServeConfig
from facehall.serve import Busy, Worker, create_app, warm_up
from facehall.train.checkpoint import save_checkpoint
from facehall.train.config import experiment_to_dict, load_experiment

# A face with 8 px between the eyes (the x4 input size), centred in a 64 x 64 photo.
POINTS = CELEBA_TEMPLATE * 32 - 0.5 + 16.0
FACE = Face(box=(16.0, 16.0, 32.0, 32.0), landmarks=POINTS, score=0.9)


class StubDetector:
    def __init__(self, faces: Sequence[Face]) -> None:
        self.faces = list(faces)
        self.calls = 0

    def detect(self, image: npt.NDArray[np.uint8]) -> list[Face]:
        self.calls += 1
        return self.faces


class FakeRestorer:
    """Nearest-neighbour upscaling, recording the batch sizes."""

    def __init__(self, scale: int = 4, device: str = "cpu") -> None:
        self.scale, self.hr_size, self.lr_size = scale, 128, 128 // scale
        self.device = device
        self.info: dict[str, Any] = {"name": f"fake-x{scale}", "scale": scale, "format": "test"}
        self.batches: list[int] = []

    def __call__(self, lr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.batches.append(len(lr))
        out: npt.NDArray[np.float32] = lr.repeat(self.scale, axis=2).repeat(self.scale, axis=3)
        return out


def enhancer(scale: int = 4, faces: Sequence[Face] = (FACE,), device: str = "cpu") -> Enhancer:
    detector = StubDetector(faces)
    return Enhancer(
        FakeRestorer(scale, device), PipelineConfig(), detector=detector, box_detector=detector
    )


def photo(size: int = 64, seed: int = 0) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 128 + 60 * np.sin(x / 5.0)[..., None] * np.cos(y / 7.0)[..., None]
    pixels: npt.NDArray[np.uint8] = np.clip(
        base + rng.normal(0, 5, (size, size, 3)), 0, 255
    ).astype(np.uint8)
    return pixels


def encoded(image: npt.NDArray[np.uint8], kind: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, kind)
    return buffer.getvalue()


def client(
    enhancers: dict[int, Enhancer] | None = None, **settings: Any
) -> tuple[TestClient, FastAPI]:
    app = create_app(enhancers or {4: enhancer()}, ServeConfig(**settings))
    return TestClient(app), app


def test_enhance_returns_the_marked_photo() -> None:
    model = enhancer()
    upload = encoded(photo())
    with TestClient(create_app({4: model}, ServeConfig())) as api:
        response = api.post("/enhance", content=upload, headers={"Content-Type": "image/png"})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-ai-enhanced"] == "true"
    assert response.headers["x-facehall-notice"] == marking.NOTICE
    assert response.headers["x-facehall-model"] == "fake-x4 (x4)"
    assert response.headers["x-facehall-faces"] == "1 face, 1 restored"
    assert response.headers["cache-control"] == "no-store"
    picture = Image.open(io.BytesIO(response.content))
    assert picture.size == (256, 256)
    expected = model.enhance(photo()).image
    assert np.array_equal(np.asarray(picture.convert("RGB")), expected)
    rec = marking.read_record(io.BytesIO(response.content))
    assert rec is not None and rec["ai_enhanced"] and rec["model"]["name"] == "fake-x4"
    assert rec["input"]["sha256"] == hashlib.sha256(upload).hexdigest()
    assert rec["input"]["faces_from"] == "detector" and rec["faces"][0]["restored"]
    assert "serve" not in rec["parameters"] and "models" not in rec["parameters"]


def test_formats_watermark_and_face_boxes() -> None:
    api, _ = client()
    upload = encoded(photo(), "JPEG")
    jpeg = api.post("/enhance?format=jpeg&quality=80&watermark=true", content=upload)
    assert jpeg.status_code == 200 and jpeg.headers["content-type"] == "image/jpeg"
    rec = marking.read_record(io.BytesIO(jpeg.content))
    assert rec is not None and rec["watermark"] is True
    webp = api.post("/enhance?format=webp", content=upload)
    assert Image.open(io.BytesIO(webp.content)).format == "WEBP"
    boxed = api.post("/enhance?box=16,16,32,32&box=0,0,8,8", content=upload)
    rec = marking.read_record(io.BytesIO(boxed.content))
    assert rec is not None and rec["input"]["faces_from"] == "boxes"


def test_the_scale_picks_the_model() -> None:
    api, _ = client({8: enhancer(8), 4: enhancer(4)})
    upload = encoded(photo())
    assert api.post("/enhance", content=upload).headers["x-facehall-model"] == "fake-x4 (x4)"
    x8 = api.post("/enhance?scale=8", content=upload)
    assert x8.headers["x-facehall-model"] == "fake-x8 (x8)"
    assert Image.open(io.BytesIO(x8.content)).size == (512, 512)
    wrong = api.post("/enhance?scale=2", content=upload)
    assert wrong.status_code == 400 and "x2 is not served" in wrong.json()["detail"]
    with pytest.raises(ValueError, match="x8 model"):
        create_app({4: enhancer(8)}, ServeConfig())


def test_bad_requests_get_clear_answers() -> None:
    api, _ = client()
    gif = encoded(photo(), "GIF")
    cases = [
        (api.post("/enhance", content=b""), 400, "must be the photo"),
        (api.post("/enhance", content=b"hello"), 415, "not an image"),
        (api.post("/enhance", content=gif), 415, "GIF images are not supported"),
        (api.post("/enhance", files={"image": ("a.png", encoded(photo()))}), 415, "not a form"),
        (api.post("/enhance?box=1,2,3", content=encoded(photo())), 400, "4 comma-separated"),
    ]
    for response, status, message in cases:
        assert response.status_code == status, response.text
        assert message in response.json()["detail"]
    assert api.post("/enhance?format=gif", content=encoded(photo())).status_code == 422
    assert api.post("/enhance?quality=0", content=encoded(photo())).status_code == 422
    broken = encoded(photo())[:200]  # a valid header, truncated pixels
    assert api.post("/enhance", content=broken).status_code == 400


def test_limits_are_checked_before_decoding() -> None:
    model = enhancer()
    api, _ = client({4: model}, max_upload_mb=0.001, max_output_pixels=128 * 128)
    big = api.post("/enhance", content=b"x" * 2000)
    assert big.status_code == 413 and "larger than 1,048 bytes" in big.json()["detail"]
    api, _ = client({4: model}, max_output_pixels=128 * 128)
    many = api.post("/enhance", content=encoded(photo()))
    assert many.status_code == 413 and "64 x 64 photo at x4 gives 65,536" in many.json()["detail"]
    assert model._detector.calls == 0  # type: ignore[union-attr]


def test_health_and_version_hold_no_local_paths(tmp_path: Path) -> None:
    model = enhancer()
    model.info = {**model.info, "checkpoint": "best.safetensors"}
    api, _ = client({4: model}, queue=3)
    assert api.get("/health").json() == {"status": "ok", "scales": [4], "pending": 0}
    version = api.get("/version").json()
    assert version["models"]["4"]["name"] == "fake-x4" and version["models"]["4"]["device"] == "cpu"
    assert version["limits"]["queue"] == 3 and version["notice"] == marking.NOTICE
    assert "/" not in str(version["models"])


def test_a_busy_service_answers_503() -> None:
    api, app = client()

    async def busy(work: Any) -> Any:
        raise Busy("the service is busy (9 requests); try again later")

    app.state.worker.run = busy
    response = api.post("/enhance", content=encoded(photo()))
    assert response.status_code == 503 and response.headers["retry-after"] == "10"
    assert "busy" in response.json()["detail"]


def test_the_worker_queues_refuses_and_times_out() -> None:
    async def scenario() -> list[str]:
        worker = Worker(queue=1, timeout=0.5)
        release = threading.Event()
        ran: list[str] = []

        def slow() -> str:
            release.wait(5)
            ran.append("slow")
            return "slow"

        def quick() -> str:
            ran.append("quick")
            return "quick"

        first = asyncio.ensure_future(worker.run(slow))
        await asyncio.sleep(0.05)
        second = asyncio.ensure_future(worker.run(quick))  # waits in the queue
        await asyncio.sleep(0.05)
        assert worker.pending == 2
        with pytest.raises(Busy, match="busy"):
            await worker.run(quick)  # beyond 1 + queue
        with pytest.raises(Busy, match="no answer within 0.5 s"):
            await first
        with pytest.raises(Busy, match="no answer"):
            await second
        release.set()
        for _ in range(100):
            if worker.pending == 0:
                break
            await asyncio.sleep(0.01)
        assert worker.pending == 0
        assert await worker.run(quick) == "quick"
        worker.close()
        return ran

    # the slow work finished; the queued one was skipped, as its request had been answered
    assert asyncio.run(scenario()) == ["slow", "quick"]


def test_warm_up_runs_each_batch_size_on_cuda() -> None:
    gpu = enhancer(faces=(), device="cuda")
    warm_up(gpu)
    assert gpu.restorer.batches == list(range(1, 33))  # type: ignore[attr-defined]
    cpu = enhancer(faces=())
    warm_up(cpu)
    assert cpu.restorer.batches == [1]  # type: ignore[attr-defined]
    with TestClient(create_app({4: enhancer(faces=())}, ServeConfig())) as api:
        assert api.get("/health").status_code == 200


def test_logs_hold_no_query_and_uploads_stay_in_memory(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_disk(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an upload went to a temporary file")

    for name in ("SpooledTemporaryFile", "TemporaryFile", "NamedTemporaryFile", "mkstemp"):
        monkeypatch.setattr(tempfile, name, no_disk)
    api, _ = client()
    with caplog.at_level(logging.INFO, logger="facehall.serve"):
        response = api.post("/enhance?box=17,19,31,33", content=encoded(photo()))
    assert response.status_code == 200
    text = caplog.text
    assert "POST /enhance 200" in text
    assert "box" not in text and "17,19" not in text


def test_fastapi_telemetry_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastapi.telemetry._asgi as telemetry

    _, app = client()
    # as if an OpenTelemetry SDK were configured in the process
    monkeypatch.setattr(telemetry, "_unconfigured", lambda provider: False)
    assert FastAPI()._native_telemetry.enabled()  # the default would record requests
    assert not app._native_telemetry.enabled()


def test_serve_config_validation() -> None:
    cases: list[dict[str, Any]] = [
        {"scales": [2]},
        {"scales": []},
        {"port": 0},
        {"queue": -1},
        {"timeout": 0},
    ]
    for settings in cases:
        with pytest.raises(ValueError, match="serve"):
            ServeConfig(**settings)


@pytest.fixture(scope="module")
def onnx_x4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("models")
    config = load_experiment(
        "configs/train/srresnet_x4.yaml", ["model.residual_blocks=1", "model.channels=8"]
    )
    model = build_model(config.model, config.scale, config.data.hr_size // config.scale)
    checkpoint = folder / "tiny-test" / "best.safetensors"
    checkpoint.parent.mkdir()
    save_checkpoint(checkpoint, model, {"config": experiment_to_dict(config), "run": "tiny-test"})
    return export_checkpoint(checkpoint, folder / "x4.onnx").path


def test_cli_serves_the_onnx_models(
    onnx_x4: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import uvicorn

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append({"app": app, **kwargs}))
    monkeypatch.setenv("FACEHALL_MODELS", str(onnx_x4.parent))
    assert main(["serve", "--scale", "4", "--port", "9999", "--device", "cpu"]) == 0
    (call,) = calls
    assert call["host"] == "127.0.0.1" and call["port"] == 9999 and call["access_log"] is False
    version = TestClient(call["app"]).get("/version").json()
    assert version["models"]["4"]["format"] == "onnx"
    assert version["models"]["4"]["name"] == "tiny-test"
    with caplog.at_level(logging.WARNING, logger="facehall.serve"):
        main(["serve", "--scale", "4", "--host", "0.0.0.0", "--device", "cpu"])
    assert "no authentication" in caplog.text


def test_cli_says_how_to_get_a_missing_model(
    onnx_x4: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FACEHALL_MODELS", str(onnx_x4.parent))
    with pytest.raises(SystemExit):
        main(["serve", "--device", "cpu"])  # x4 and x8 by default; there is no x8.onnx
    assert "facehall export onnx --scale 8" in capsys.readouterr().err


def requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    assert match is not None, requirement
    return match.group(0).lower()


def image_groups() -> tuple[list[str], dict[str, list[str]]]:
    """The runtime dependencies of the package, and the packages of each image group."""
    project = tomllib.loads(Path("pyproject.toml").read_text())
    groups = project["dependency-groups"]

    def resolve(name: str) -> list[str]:
        out: list[str] = []
        for item in groups[name]:
            out += resolve(item["include-group"]) if isinstance(item, dict) else [item]
        return out

    return project["project"]["dependencies"], {g: resolve(g) for g in ("image-cpu", "image-gpu")}


def test_the_image_groups_follow_the_package_dependencies() -> None:
    dependencies, groups = image_groups()
    bounds = {requirement_name(d): d for d in dependencies}
    for group, requirements in groups.items():
        for requirement in requirements:
            name = requirement_name(requirement)
            if name == "onnxruntime-gpu":  # instead of onnxruntime, with the same version (D35)
                assert requirement.endswith(bounds["onnxruntime"].removeprefix("onnxruntime"))
                continue
            assert bounds.get(name) == requirement, (group, requirement)


SERVE_WITH_THE_IMAGE_PACKAGES = """
import importlib.abc, io, json, sys
from pathlib import Path

BLOCKED = set(json.loads(sys.argv[1]))

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is not in the service images")
        return None

sys.meta_path.insert(0, Block())
import numpy as np
from fastapi.testclient import TestClient
from facehall.align import CELEBA_TEMPLATE, Face
from facehall.cli import build_parser
from facehall.config import load_config
from facehall.pipeline import PipelineConfig
from facehall.serve import create_app, load_enhancers

class Stub:
    def detect(self, image):
        return [Face((16.0, 16.0, 32.0, 32.0), CELEBA_TEMPLATE * 32 - 0.5 + 16.0, 0.9)]

build_parser().parse_args(["serve", "--scale", "4"])
pipeline = Path("configs/pipeline/default.yaml")
config = load_config(pipeline, PipelineConfig, ["models.backend=onnx"])
enhancers = load_enhancers(config, [4], pipeline, "cpu")
enhancers[4]._detector = enhancers[4]._box_detector = Stub()
with TestClient(create_app(enhancers, config.serve)) as api:
    response = api.post("/enhance", content=sys.stdin.buffer.read())
assert response.status_code == 200, response.text
assert response.headers["x-facehall-faces"] == "1 face, 1 restored", response.headers
loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
assert not loaded, loaded
"""


def test_the_service_runs_with_the_image_packages_only(onnx_x4: Path) -> None:
    """The service imports none of the package's runtime dependencies that the images leave
    out (PyTorch, the ONNX exporter, pyarrow, safetensors, …)."""
    dependencies, groups = image_groups()
    image = {requirement_name(r) for r in groups["image-cpu"]}
    blocked = sorted(
        requirement_name(d).replace("-", "_")
        for d in dependencies
        if requirement_name(d) not in image
    )
    assert {"torch", "torchvision", "onnx", "onnxscript", "pyarrow", "safetensors"} <= set(blocked)
    result = subprocess.run(
        [sys.executable, "-c", SERVE_WITH_THE_IMAGE_PACKAGES, json.dumps(blocked)],
        input=encoded(photo()),
        capture_output=True,
        env={**os.environ, "FACEHALL_MODELS": str(onnx_x4.parent)},
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()[-2000:]
