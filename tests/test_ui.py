"""The web interface (``facehall ui``, D41): access, files, previews, batches and the page.

A numpy stand-in for the model and a stub detector, so no GPU, weights or dataset.
"""

import io
import json
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from facehall import marking
from facehall.align import CELEBA_TEMPLATE, Face
from facehall.cli import build_parser
from facehall.enhance import parameters, plan
from facehall.pipeline import Enhancer, PipelineConfig, UiConfig
from facehall.ui import Batches, Options, Settings, create_app
from facehall.ui.app import host_allowed
from facehall.ui.batches import run_folder
from facehall.ui.media import Pictures, default_output, listing, places

STATIC = Path("src/facehall/ui/static")
TOKEN = "t" * 32
POINTS = CELEBA_TEMPLATE * 32 - 0.5 + 16.0  # 8 px between the eyes in a 64 x 64 photo
FACE = Face(box=(16.0, 16.0, 32.0, 32.0), landmarks=POINTS, score=0.9)
FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg")


class StubDetector:
    def __init__(self, faces: Sequence[Face], gate: threading.Event | None = None) -> None:
        self.faces, self.gate = list(faces), gate

    def detect(self, image: npt.NDArray[np.uint8]) -> list[Face]:
        if self.gate is not None:
            assert self.gate.wait(10), "the test did not open the gate"
        return self.faces


class FakeRestorer:
    """Nearest-neighbour upscaling."""

    def __init__(self, scale: int = 4) -> None:
        self.scale, self.hr_size, self.lr_size = scale, 128, 128 // scale
        self.device = "cpu"
        self.info: dict[str, Any] = {"name": f"fake-x{scale}", "scale": scale, "sha256": "0" * 64}

    def __call__(self, lr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        out: npt.NDArray[np.float32] = lr.repeat(self.scale, axis=2).repeat(self.scale, axis=3)
        return out


def make_enhancer(scale: int = 4, gate: threading.Event | None = None) -> Enhancer:
    detector = StubDetector([FACE] if scale == 4 else [], gate)
    return Enhancer(FakeRestorer(scale), PipelineConfig(), detector=detector, box_detector=detector)


def write_photo(path: Path, size: tuple[int, int] = (64, 64), seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def make_video(path: Path, frames: int = 5) -> Path:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"]
    command += ["-i", f"testsrc=size=64x64:rate=10:duration={frames / 10}"]
    subprocess.run([*command, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def wait(client: TestClient, batch_id: int, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        batch: dict[str, Any] = client.get(f"/api/batches/{batch_id}").json()
        if batch["status"] not in ("queued", "loading", "running"):
            return batch
        time.sleep(0.02)
    raise AssertionError(f"batch {batch_id} did not finish")


@pytest.fixture
def ui(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    """A client with the token, over the stand-in models, with tmp_path as the home folder."""
    home = tmp_path / "home"
    home.mkdir()
    settings = Settings(
        home=home,
        output=home / "out",
        scales={4: True, 8: False},
        backend="test",
        device="cpu",
        ffmpeg=FFMPEG,
    )
    batches = Batches(make_enhancer, lambda scale: scale == 4)
    app = create_app(batches, settings, UiConfig(), token=TOKEN)
    with TestClient(app, base_url="http://127.0.0.1:8100") as client:
        client.headers["X-Facehall-Token"] = TOKEN
        yield client, home


# --- access ------------------------------------------------------------------------------


def test_the_api_needs_the_token_and_a_local_host(ui: tuple[TestClient, Path]) -> None:
    client, _ = ui
    assert client.get("/api/info").status_code == 200
    assert client.get("/api/info", headers={"X-Facehall-Token": "wrong"}).status_code == 401
    bare = TestClient(client.app, base_url="http://127.0.0.1:8100")
    assert bare.get("/api/info").status_code == 401
    assert bare.get("/api/info", params={"token": TOKEN}).status_code == 200
    # a page that rebinds its own domain to 127.0.0.1 sends its own Host
    assert client.get("/api/info", headers={"Host": "evil.example:8100"}).status_code == 400
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    assert host_allowed("localhost:8100", "127.0.0.1") and host_allowed("[::1]:8100", "::1")
    assert not host_allowed("127.0.0.1.evil.example", "127.0.0.1")


def test_the_page_and_its_assets(ui: tuple[TestClient, Path]) -> None:
    client, _ = ui
    page = TestClient(client.app, base_url="http://127.0.0.1:8100").get("/")  # no token
    assert page.status_code == 200 and "facehall" in page.text
    policy = page.headers["content-security-policy"]
    assert "script-src 'self'" in policy and "frame-ancestors 'none'" in policy
    for name, kind in [("app.js", "javascript"), ("app.css", "css"), ("i18n.json", "json")]:
        response = client.get(f"/static/{name}")
        assert response.status_code == 200 and kind in response.headers["content-type"]
    assert client.get("/static/../app.py").status_code == 404
    assert client.get("/static/secret.txt").status_code == 404
    assert client.get("/docs").status_code == 404  # no API pages


def test_short_tokens_are_refused() -> None:
    settings = Settings(Path("/"), Path("/out"), {4: True}, "test", "cpu", False)
    with pytest.raises(ValueError, match="too short"):
        create_app(Batches(make_enhancer), settings, UiConfig(), token="short")


def test_info(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    info = client.get("/api/info").json()
    assert info["scales"] == {"4": True, "8": False}
    assert info["home"] == str(home) and info["output"] == str(home / "out")
    assert info["notice"] == marking.NOTICE
    assert info["places"][0] == {"id": "home", "path": str(home)}


# --- files -------------------------------------------------------------------------------


def test_folder_listing(tmp_path: Path) -> None:
    write_photo(tmp_path / "b.PNG")
    write_photo(tmp_path / "a.jpg")
    (tmp_path / "clip.mp4").write_bytes(b"not decoded here")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden.png").write_bytes(b"x")
    (tmp_path / "Sub").mkdir()
    (tmp_path / ".cache").mkdir()
    found = listing(tmp_path)
    assert [(e["name"], e["kind"]) for e in found["entries"]] == [
        ("Sub", "dir"),
        ("a.jpg", "image"),
        ("b.PNG", "image"),
        ("clip.mp4", "video"),
    ]
    assert found["parent"] == str(tmp_path.parent)
    assert {e["name"] for e in listing(tmp_path, hidden=True)["entries"]} >= {
        ".cache",
        ".hidden.png",
    }
    assert listing(Path("/"))["parent"] is None


def test_folder_endpoint_errors(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    write_photo(home / "p.png")
    assert client.get("/api/folder").json()["path"] == str(home)
    assert client.get("/api/folder", params={"path": str(home / "nope")}).status_code == 404
    assert client.get("/api/folder", params={"path": "relative/path"}).status_code == 400
    assert client.get("/api/folder", params={"path": str(home / "p.png")}).status_code == 404


def test_thumbnail_preview_and_crop(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    path = str(write_photo(home / "p.png", (80, 40)))
    thumb = client.get("/api/thumbnail", params={"path": path})
    assert thumb.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(thumb.content)).size == (80, 40)  # never enlarged
    preview = client.get("/api/preview", params={"path": path, "width": 320, "height": 160})
    assert Image.open(io.BytesIO(preview.content)).size == (320, 160)
    crop = client.get(
        "/api/crop",
        params={"path": path, "x": -10, "y": 10, "w": 40, "h": 20, "size": 90},
    )
    assert crop.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(crop.content)).size == (90, 60)  # clipped to x in [0, 30)
    outside = {"path": path, "x": 500, "y": 0, "w": 10, "h": 10}
    assert client.get("/api/crop", params=outside).status_code == 422
    assert (
        client.get("/api/preview", params={"path": path, "width": 9000, "height": 9000}).status_code
        == 400
    )
    (home / "a.txt").write_text("x")
    assert client.get("/api/thumbnail", params={"path": str(home / "a.txt")}).status_code == 415
    assert client.get("/api/thumbnail", params={"path": str(home / "gone.png")}).status_code == 404
    (home / "broken.png").write_bytes(b"not a png")
    assert (
        client.get("/api/thumbnail", params={"path": str(home / "broken.png")}).status_code == 422
    )


def test_crops_match_the_pixels_with_nearest_interpolation(tmp_path: Path) -> None:
    pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    path = tmp_path / "p.png"
    Image.fromarray(pixels).save(path)
    data = Pictures(1 << 20).crop(path, "image", (4, 2, 8, 8), 32, "nearest")
    crop = np.asarray(Image.open(io.BytesIO(data)))
    assert np.array_equal(crop[::4, ::4], pixels[2:10, 4:12])


def test_the_picture_cache_keeps_to_its_budget(tmp_path: Path) -> None:
    pictures = Pictures(max_bytes=2 * 64 * 64 * 3)
    paths = [write_photo(tmp_path / f"{i}.png", seed=i) for i in range(3)]
    for path in paths:
        pictures.picture(path, "image")
    assert len(pictures.decoded) == 2
    write_photo(paths[2], seed=9)  # a changed file is decoded again
    assert not np.array_equal(
        np.asarray(pictures.picture(paths[2], "image")),
        np.asarray(Image.open(paths[1]).convert("RGB")),
    )


def test_file_download_and_range(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    path = write_photo(home / "p.png")
    whole = client.get("/api/file", params={"path": str(path)})
    assert whole.content == path.read_bytes() and whole.headers["content-type"] == "image/png"
    assert "content-disposition" not in whole.headers
    download = client.get("/api/file", params={"path": str(path), "download": True})
    assert download.headers["content-disposition"].startswith("attachment")
    part = client.get("/api/file", params={"path": str(path)}, headers={"Range": "bytes=0-9"})
    assert part.status_code == 206 and part.content == path.read_bytes()[:10]


# --- batches -----------------------------------------------------------------------------


def test_a_batch_writes_marked_outputs_into_a_new_subfolder(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    write_photo(home / "in" / "a.png")
    write_photo(home / "in" / "sub" / "b.jpg", seed=1)
    out = home / "out"
    response = client.post(
        "/api/batches",
        json={"inputs": [str(home / "in")], "output": str(out), "image_format": "webp"},
    )
    assert response.status_code == 201, response.text
    batch = wait(client, response.json()["id"])
    assert batch["status"] == "done" and batch["counts"]["done"] == 2
    folder = Path(batch["folder"])
    assert folder.parent == out and re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_x4", folder.name)
    assert sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.webp")) == [
        "a.webp",
        "sub/b.webp",
    ]
    item = batch["items"][0]
    assert item["output"] == {"width": 256, "height": 256}
    assert item["faces"][0]["restored"] and item["faces"][0]["region"]
    record = marking.read_record(Path(item["target"]))
    assert record is not None and record["model"]["name"] == "fake-x4"
    assert batch["model"]["name"] == "fake-x4" and batch["progress"] == 1.0
    listed = client.get("/api/batches").json()
    assert listed[0]["id"] == batch["id"] and "items" not in listed[0]


def test_batch_options_are_checked_before_queueing(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    photo = str(write_photo(home / "a.png"))
    out = str(home / "out")

    def submit(**changes: Any) -> Any:
        return client.post("/api/batches", json={"inputs": [photo], "output": out, **changes})

    assert submit(scale=8).status_code == 400  # no x8 model set up
    assert "no x8 model" in submit(scale=8).json()["detail"]
    assert submit(scale=2).status_code == 422
    assert submit(quality=0).status_code == 422
    assert submit(inputs=[]).status_code == 422
    assert submit(inputs=[str(home / "gone.png")]).status_code == 400
    assert submit(inputs=["a.png"]).status_code == 400  # relative
    assert submit(output=str(home / "out.png")).status_code == 400
    assert submit(output=photo).status_code == 400
    assert client.get("/api/batches").json() == []


def test_existing_outputs_need_overwrite_without_a_subfolder(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    photo = str(write_photo(home / "a.png"))
    out = home / "out"
    body = {"inputs": [photo], "output": str(out), "subfolder": False}
    first = wait(client, client.post("/api/batches", json=body).json()["id"])
    assert first["folder"] == str(out) and (out / "a.png").is_file()
    again = client.post("/api/batches", json=body)
    assert again.status_code == 400 and "exists" in again.json()["detail"]
    replaced = client.post("/api/batches", json={**body, "overwrite": True})
    assert wait(client, replaced.json()["id"])["status"] == "done"


def test_earlier_outputs_inside_an_input_folder_are_left_out(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    photos = home / "photos"
    write_photo(photos / "a.png")
    out = photos / "enhanced"
    body = {"inputs": [str(photos)], "output": str(out)}
    first = wait(client, client.post("/api/batches", json=body).json()["id"])
    second = wait(client, client.post("/api/batches", json=body).json()["id"])
    assert first["total"] == second["total"] == 1
    assert second["items"][0]["source"] == str(photos / "a.png")


def test_run_folders_never_collide(tmp_path: Path) -> None:
    options = Options(inputs=(tmp_path,), output=tmp_path / "out", scale=8)
    now = datetime(2026, 11, 27, 14, 30, 5)
    first = run_folder(options, now)
    assert first == tmp_path / "out" / "2026-11-27_14-30-05_x8"
    first.mkdir(parents=True)
    assert run_folder(options, now).name == "2026-11-27_14-30-05_x8-2"
    assert run_folder(Options(inputs=(tmp_path,), output=tmp_path, subfolder=False)) == tmp_path
    with pytest.raises(ValueError, match="folder"):
        run_folder(Options(inputs=(tmp_path,), output=tmp_path / "x.mp4"))


def test_cancel_stops_the_remaining_inputs(tmp_path: Path) -> None:
    gate = threading.Event()
    batches = Batches(lambda scale: make_enhancer(scale, gate))
    try:
        inputs = tuple(write_photo(tmp_path / f"{i}.png", seed=i) for i in range(3))
        batch = batches.submit(Options(inputs=inputs, output=tmp_path / "out"))
        later = batches.submit(Options(inputs=inputs[:1], output=tmp_path / "out"))
        deadline = time.monotonic() + 10
        while batches.get(batch.id)["items"][0]["status"] != "running":  # type: ignore[index]
            assert time.monotonic() < deadline
            time.sleep(0.01)
        batches.cancel(batch.id)
        batches.cancel(later.id)
        gate.set()
        while batches.busy():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        statuses = [i["status"] for i in batches.get(batch.id)["items"]]  # type: ignore[index]
        assert statuses == ["done", "cancelled", "cancelled"]  # the running image finishes
        assert batches.get(batch.id)["status"] == "cancelled"  # type: ignore[index]
        assert batches.get(later.id)["status"] == "cancelled"  # type: ignore[index]
    finally:
        gate.set()
        batches.close()


def test_a_failing_model_load_fails_the_batch_only(tmp_path: Path) -> None:
    calls = []

    def load(scale: int) -> Enhancer:
        calls.append(scale)
        if len(calls) == 1:
            raise FileNotFoundError("no model here")
        return make_enhancer(scale)

    batches = Batches(load)
    try:
        photo = write_photo(tmp_path / "a.png")
        first = batches.submit(Options(inputs=(photo,), output=tmp_path / "out"))
        second = batches.submit(Options(inputs=(photo,), output=tmp_path / "out"))
        deadline = time.monotonic() + 10
        while batches.busy():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        failed = batches.get(first.id)
        assert failed is not None and failed["status"] == "failed"
        assert failed["message"] == "no model here"
        assert batches.get(second.id)["status"] == "done"  # type: ignore[index]
    finally:
        batches.close()


def test_a_broken_input_does_not_stop_the_batch(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    (home / "in").mkdir()
    (home / "in" / "a.png").write_bytes(b"broken")
    write_photo(home / "in" / "b.png")
    batch_id = client.post(
        "/api/batches", json={"inputs": [str(home / "in")], "output": str(home / "out")}
    ).json()["id"]
    batch = wait(client, batch_id)
    assert [i["status"] for i in batch["items"]] == ["failed", "done"]
    assert batch["items"][0]["message"] and batch["status"] == "done"
    assert not list((home / "out").rglob("*.partial*"))


@needs_ffmpeg
def test_a_video_batch_reports_progress_and_its_record(ui: tuple[TestClient, Path]) -> None:
    client, home = ui
    clip = make_video(home / "clip.mp4", frames=6)
    batch_id = client.post(
        "/api/batches",
        json={"inputs": [str(clip)], "output": str(home / "out"), "video_format": "mkv"},
    ).json()["id"]
    batch = wait(client, batch_id, timeout=60)
    item = batch["items"][0]
    assert item["status"] == "done" and item["kind"] == "video"
    assert item["target"].endswith(".mkv") and Path(item["target"]).is_file()
    assert item["video"]["frames"] == 6 and item["faces"]["detected"] == 6
    thumb = client.get("/api/thumbnail", params={"path": item["target"]})
    assert thumb.status_code == 200 and Image.open(io.BytesIO(thumb.content)).size == (160, 160)


@needs_ffmpeg
def test_cancelling_a_video_writes_nothing(tmp_path: Path) -> None:
    clip = make_video(tmp_path / "clip.mp4", frames=30)
    out = tmp_path / "out"
    batches = Batches(make_enhancer)
    try:
        batch = batches.submit(Options(inputs=(clip,), output=out, subfolder=False))
        deadline = time.monotonic() + 30
        while not (batches.get(batch.id)["items"][0]["progress"] or 0) > 0:  # type: ignore[index]
            assert time.monotonic() < deadline
            time.sleep(0.005)
        batches.cancel(batch.id)
        while batches.busy():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert batches.get(batch.id)["items"][0]["status"] == "cancelled"  # type: ignore[index]
        assert not out.exists() or not any(out.iterdir())
    finally:
        batches.close()


def test_open_only_the_output_folders_of_the_session(
    ui: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, home = ui
    opened: list[list[str]] = []

    class Popen:
        def __init__(self, command: list[str], **_: Any) -> None:
            opened.append(command)

    monkeypatch.setattr("facehall.ui.app.subprocess.Popen", Popen)
    assert client.post("/api/open", json={"path": str(home)}).status_code == 403
    photo = str(write_photo(home / "a.png"))
    batch = wait(
        client,
        client.post("/api/batches", json={"inputs": [photo], "output": str(home / "out")}).json()[
            "id"
        ],
    )
    assert client.post("/api/open", json={"path": batch["folder"]}).status_code == 204
    assert opened and opened[0][-1] == str(Path(batch["folder"]).resolve())


# --- shared code, the command and the page ------------------------------------------------


def test_plan_excludes_a_folder(tmp_path: Path) -> None:
    write_photo(tmp_path / "in" / "a.png")
    write_photo(tmp_path / "in" / "out" / "old.png")
    jobs = plan([tmp_path / "in"], tmp_path / "in" / "out" / "new", exclude=tmp_path / "in" / "out")
    assert [j.source.name for j in jobs] == ["a.png"]
    assert len(plan([tmp_path / "in"], tmp_path / "elsewhere")) == 2


def test_the_record_has_no_ui_settings() -> None:
    assert "ui" not in parameters(PipelineConfig())
    with pytest.raises(ValueError, match="port"):
        UiConfig(port=0)


def test_the_ui_command() -> None:
    args = build_parser().parse_args(["ui", "--port", "9100", "--no-browser"])
    assert args.command == "ui" and args.port == 9100 and args.no_browser


def test_every_text_has_both_languages() -> None:
    strings = json.loads((STATIC / "i18n.json").read_text(encoding="utf-8"))
    assert set(strings) == {"en", "uk"}
    assert set(strings["en"]) == set(strings["uk"])
    for key, value in strings["en"].items():
        assert type(value) is type(strings["uk"][key]), key
        if isinstance(value, dict):
            assert {"one", "other"} <= set(value) and {"one", "few", "many", "other"} <= set(
                strings["uk"][key]
            ), key


def test_every_text_the_page_uses_exists() -> None:
    strings = json.loads((STATIC / "i18n.json").read_text(encoding="utf-8"))["en"]
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    used = set(re.findall(r'data-i18n(?:-title)?="([^"]+)"', page))
    used |= set(re.findall(r'\bt\("([^"]+)"', script))
    used |= {f"scale.help.{s}" for s in (4, 8)}
    used |= {f"status.{s}" for s in ("queued", "loading", "running", "done", "failed", "cancelled")}
    used |= {f"theme.{m}" for m in ("auto", "light", "dark")}
    used |= {f"reason.{k}" for k in ("large", "small", "outside", "other")}
    used |= {f"place.{p}" for p in ("home", "pictures", "videos", "desktop", "download")}
    used |= {"picker.files", "picker.folder", "picker.output"}
    assert used - set(strings) == set()


def test_the_page_loads_nothing_from_elsewhere() -> None:
    for path in STATIC.iterdir():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", text), path.name
        if path.suffix == ".html":
            assert not re.search(r"\son\w+=|<style|\sstyle=", text), "inline code or styles"


def test_places_fall_back_to_the_usual_folder_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Pictures").mkdir()
    (tmp_path / "Downloads").mkdir()

    class Answer:
        stdout = f"{tmp_path}\n"  # xdg-user-dir: the home folder for a folder not set up

    monkeypatch.setattr("facehall.ui.media.shutil.which", lambda _: "/usr/bin/xdg-user-dir")
    monkeypatch.setattr("facehall.ui.media.subprocess.run", lambda *a, **k: Answer())
    assert [p["id"] for p in places(tmp_path)] == ["home", "pictures", "download"]
    assert default_output(tmp_path) == tmp_path / "Pictures" / "facehall"
    monkeypatch.setattr("facehall.ui.media.shutil.which", lambda _: None)
    assert default_output(tmp_path / "nowhere") == tmp_path / "nowhere" / "facehall"
