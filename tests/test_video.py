"""Video (facehall.video), video marks and the flicker measure (facehall.eval.temporal).

The model is a tiny untrained SRResNet saved as a checkpoint and the detector a stand-in, so no
weights or dataset are needed. Tests that encode video need the ffmpeg program and are skipped
without it.
"""

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from facehall import marking
from facehall.align import CELEBA_TEMPLATE, Face
from facehall.cli import main
from facehall.enhance import plan
from facehall.eval.temporal import (
    CONDITIONS,
    Tracking,
    evaluate_clip,
    flicker,
    make_clip,
    mean_results,
    motion,
    render,
)
from facehall.models import build_model
from facehall.pipeline import Enhancer, PipelineConfig, VideoConfig
from facehall.train import load_experiment
from facehall.train.checkpoint import save_checkpoint
from facehall.train.config import experiment_to_dict
from facehall.video import FaceTracker, mux_command, read_record, tracker

FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="needs the ffmpeg and ffprobe programs")

# A face with 8 px between the eyes (the x4 input size), centred in a 64 x 64 frame.
POINTS = CELEBA_TEMPLATE * 32 - 0.5 + 16.0


def face(offset: tuple[float, float] = (0.0, 0.0), score: float = 0.9) -> Face:
    points = POINTS + np.asarray(offset)
    return Face(box=(16.0 + offset[0], 16.0 + offset[1], 32.0, 32.0), landmarks=points, score=score)


class StubDetector:
    def __init__(self, faces: Sequence[Face]) -> None:
        self.faces = list(faces)

    def detect(self, image: npt.NDArray[np.uint8]) -> list[Face]:
        return self.faces


def stub(enhancer: Enhancer, detector: object) -> Enhancer:
    """``enhancer`` with ``detector`` for photos, face boxes and video tracks (no weights)."""
    enhancer._detector = enhancer._box_detector = enhancer._track_detector = detector  # type: ignore[assignment]
    return enhancer


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = load_experiment(
        "configs/train/srresnet_x4.yaml", ["model.residual_blocks=1", "model.channels=8"]
    )
    model = build_model(config.model, config.scale, config.data.hr_size // config.scale)
    path = tmp_path_factory.mktemp("run") / "best.safetensors"
    save_checkpoint(path, model, {"config": experiment_to_dict(config), "run": "tiny_x4-test"})
    return path


# --- face tracks -------------------------------------------------------------------------


def test_tracker_without_smoothing_passes_detections_through() -> None:
    tracks = FaceTracker(1.0, 0.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        detected = face(tuple(rng.normal(0, 0.5, 2)))
        (out,) = tracks.update([detected])
        assert np.allclose(out.landmarks, detected.landmarks)
        assert np.allclose(out.box, detected.box) and out.score == detected.score


def test_smoothing_damps_jitter_and_follows_constant_motion() -> None:
    rng = np.random.default_rng(0)
    raw, smoothed = [], []
    tracks = FaceTracker(0.3, 0.05)
    for t in range(200):
        truth = (0.5 * t, 0.2 * t)
        noisy = face((truth[0] + rng.normal(0, 0.5), truth[1] + rng.normal(0, 0.5)))
        (out,) = tracks.update([noisy])
        if t >= 100:  # steady state
            raw.append(noisy.landmarks - (POINTS + truth))
            smoothed.append(out.landmarks - (POINTS + truth))
    assert np.std(smoothed) < 0.6 * np.std(raw)
    assert abs(float(np.mean(smoothed))) < 0.1  # no lag behind the motion


def test_hold_keeps_a_missed_face_then_ends_the_track() -> None:
    tracks = FaceTracker(0.5, 0.1, hold=2, keep=3)
    for t in range(4):
        tracks.update([face((0.5 * t, 0.0))])
    counts = [len(tracks.update([])) for _ in range(4)]
    assert counts == [1, 1, 0, 0]
    assert tracks.tracks == []
    # without hold, a missed face is not returned, but its track continues
    tracks = FaceTracker(0.5, 0.1)
    tracks.update([face()])
    assert tracks.update([]) == []
    (out,) = tracks.update([face()])
    assert len(tracks.tracks) == 1


def test_weak_detections_only_continue_tracks() -> None:
    tracks = FaceTracker(1.0, 0.0, start=0.8)
    assert tracks.update([face(score=0.5)]) == []  # too weak to begin a track
    assert tracks.tracks == []
    (strong,) = tracks.update([face(score=0.9)])
    (weak,) = tracks.update([face((0.5, 0.0), score=0.4)])  # continues it
    assert weak.score == 0.4 and len(tracks.tracks) == 1
    # a weak detection elsewhere begins nothing
    (kept,) = tracks.update([face((0.5, 0.0), score=0.4), face((100.0, 0.0), score=0.4)])
    assert kept.box[0] < 20 and len(tracks.tracks) == 1


def test_faces_far_apart_get_their_own_tracks() -> None:
    tracks = FaceTracker(0.5, 0.0)
    tracks.update([face(), face((100.0, 0.0))])
    a, b = tracks.update([face((100.5, 0.0)), face((0.5, 0.0))])
    assert a.box[0] > 90 and b.box[0] < 20
    assert len(tracks.tracks) == 2


def test_tracker_settings_and_validation() -> None:
    off = VideoConfig(smooth=False, hold=0, track_threshold=0.8)
    assert tracker(off, 0.8) is None
    default = tracker(VideoConfig(), 0.8)
    assert default is not None and (default.alpha, default.beta, default.start) == (0.5, 0.1, 0.8)
    held = tracker(VideoConfig(smooth=False, hold=3, track_threshold=0.8), 0.8)
    assert held is not None and (held.alpha, held.beta, held.hold, held.keep) == (1.0, 0.0, 3, 3)
    assert held.start == 0.0  # no hysteresis
    smooth = tracker(VideoConfig(alpha=0.3, beta=0.05, hold=0), 0.8)
    assert smooth is not None and (smooth.alpha, smooth.beta, smooth.hold) == (0.3, 0.05, 0)
    for kwargs in ({"alpha": 0.0}, {"beta": 1.5}, {"iou": 0.0}, {"hold": 3, "keep": 2}):
        with pytest.raises(ValueError):
            FaceTracker(**{"alpha": 0.5, "beta": 0.1, **kwargs})  # type: ignore[arg-type]
    for config in ({"codec": "mpeg4"}, {"crf": 52}, {"hold": -1}, {"track_threshold": 0.0}):
        with pytest.raises(ValueError):
            VideoConfig(**config)


# --- planning and marks ------------------------------------------------------------------


def test_plan_sends_videos_to_video_outputs(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    (folder / "sub").mkdir(parents=True)
    for name in ("a.jpg", "sub/clip.mov", "notes.txt"):
        (folder / name).write_bytes(b"x")
    jobs = plan([folder], tmp_path / "out", video_suffix=".mkv")
    assert sorted(j.target.relative_to(tmp_path / "out").as_posix() for j in jobs) == [
        "a.png",
        "sub/clip.mkv",
    ]
    video = folder / "sub" / "clip.mov"
    assert plan([video], tmp_path / "x.mp4")[0].target == tmp_path / "x.mp4"
    with pytest.raises(ValueError, match="input's kind"):
        plan([video], tmp_path / "x.png")
    with pytest.raises(ValueError, match="input's kind"):
        plan([folder / "a.jpg"], tmp_path / "x.mp4")
    with pytest.raises(ValueError, match="video format"):
        plan([folder], tmp_path / "out", video_suffix=".webm")


def test_video_record_and_tags() -> None:
    rec = marking.video_record(
        model={"name": "run-1", "sha256": "ab" * 32},
        parameters={"video": {"smooth": True}},
        input_sha256="cd" * 32,
        input_size=(64, 48),
        output_size=(256, 192),
        video={"fps": 25.0, "frames": 5, "audio": True},
        faces={"detected": 5, "restored": 4, "frames_with_faces": 5},
        watermark=False,
        created="2026-11-26T00:00:00Z",
    )
    assert rec["faces"] == {"detected": 5, "restored": 4, "frames_with_faces": 5}
    assert rec["input"]["faces_from"] == "detector" and rec["video"]["frames"] == 5
    tags = marking.video_tags(rec)
    assert tags["ai_enhanced"] == "true" and tags["comment"] == marking.NOTICE
    assert tags["digital_source_type"] == marking.DIGITAL_SOURCE_TYPE
    assert json.loads(tags["facehall"]) == rec


def test_mux_command_per_container() -> None:
    if not FFMPEG:
        pytest.skip("needs ffmpeg")
    tags = {"comment": "x"}
    mp4 = mux_command(Path("v.mp4"), Path("o.mp4"), audio_from=Path("in.mov"), tags=tags)
    assert mp4[mp4.index("-c:a") : mp4.index("-c:a") + 2] == ["-c:a", "aac"]
    assert "+use_metadata_tags" in mp4 and "comment=x" in mp4
    mkv = mux_command(Path("v.mkv"), Path("o.mkv"), audio_from=None, tags=tags)
    assert "-c:a" not in mkv and "+use_metadata_tags" not in mkv


# --- video end to end --------------------------------------------------------------------


def make_video(path: Path, frames: int = 5, audio: bool = True) -> None:
    """A 64 x 64 test video at 10 fps, H.264, with a sine tone."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    command += ["-f", "lavfi", "-i", f"testsrc=size=64x64:rate=10:duration={frames / 10}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / 10}"]
        command += ["-c:a", "aac", "-shortest"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True)


def streams(path: Path) -> list[dict[str, str]]:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries"]
        + ["stream=codec_type,width,height,nb_read_frames", "-of", "json", str(path)],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    result: list[dict[str, str]] = json.loads(output)["streams"]
    return result


@needs_ffmpeg
def test_cli_enhances_a_video_with_audio_and_marks(
    tmp_path: Path,
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "clip.mp4"
    make_video(source)
    target = tmp_path / "out" / "clip.mp4"
    detector = StubDetector([face()])
    original = Enhancer.from_checkpoint

    def with_stub(*args: object, **kwargs: object) -> Enhancer:
        enhancer = original(*args, **kwargs)  # type: ignore[arg-type]
        stub(enhancer, detector)
        return enhancer

    monkeypatch.setattr(Enhancer, "from_checkpoint", with_stub)
    code = main(
        ["enhance", str(source), "-o", str(target), "--ckpt", str(checkpoint), "--device", "cpu"]
        + ["--set", "video.smooth=false", "--set", "video.hold=0"]
    )
    output = capsys.readouterr().out
    assert code == 0, output
    assert "5 frames, 5 faces in 5 frames, 5 restored" in output
    kinds = {s["codec_type"]: s for s in streams(target)}
    assert set(kinds) == {"video", "audio"}
    assert (int(kinds["video"]["width"]), int(kinds["video"]["height"])) == (256, 256)
    assert int(kinds["video"]["nb_read_frames"]) == 5
    rec = read_record(target)
    assert rec is not None and rec["faces"]["restored"] == 5
    assert rec["parameters"]["video"]["smooth"] is False and rec["model"]["name"] == "tiny_x4-test"
    assert not list(target.parent.glob("tmp*")), "no scratch files left"


@needs_ffmpeg
def test_video_without_audio_and_a_broken_video(tmp_path: Path, checkpoint: Path) -> None:
    from facehall.enhance import enhance_files

    silent, broken = tmp_path / "silent.mp4", tmp_path / "broken.mp4"
    make_video(silent, frames=3, audio=False)
    broken.write_bytes(b"not a video")
    enhancer = Enhancer.from_checkpoint(checkpoint, PipelineConfig())
    stub(enhancer, StubDetector([]))
    jobs = plan([silent, broken], tmp_path / "out")
    failures = enhance_files(enhancer, jobs, report=lambda _: None)
    assert [f[0] for f in failures] == [broken]
    out = tmp_path / "out" / "silent.mp4"
    assert [s["codec_type"] for s in streams(out)] == ["video"]
    rec = read_record(out)
    assert rec is not None and rec["faces"]["detected"] == 0
    assert (
        "models" not in rec["parameters"] and rec["parameters"]["video"]["track_threshold"] == 0.3
    )


@needs_ffmpeg
def test_rotated_video_is_written_upright(tmp_path: Path, checkpoint: Path) -> None:
    from facehall.video import enhance_video

    plain, rotated = tmp_path / "plain.mp4", tmp_path / "rotated.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run(
        command
        + ["-f", "lavfi", "-i", "testsrc=size=48x32:rate=10:duration=0.2"]
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(plain)],
        check=True,
    )
    subprocess.run(
        command + ["-display_rotation", "90", "-i", str(plain), "-c", "copy", str(rotated)],
        check=True,
    )
    enhancer = Enhancer.from_checkpoint(checkpoint, PipelineConfig())
    stub(enhancer, StubDetector([]))
    out = tmp_path / "out.mp4"
    enhance_video(enhancer, rotated, out, parameters={}, report=lambda _: None)
    (video,) = streams(out)
    assert (int(video["width"]), int(video["height"])) == (128, 192)


# --- the flicker measure -----------------------------------------------------------------


def test_render_moves_the_landmarks_with_the_content() -> None:
    image = np.zeros((40, 30, 3), np.uint8)
    image[19:22, 14:17] = 255  # a bright dot at (15, 20)
    points = np.tile([[15.0, 20.0]], (5, 1))
    for offset in ((0.0, 0.0), (1.3, -0.6)):
        scene, truth = render(image, points, 0.5, offset)
        assert scene.shape == (60, 45, 3)
        weight = scene[..., 0].astype(np.float64)
        weight = np.where(weight > 20, weight, 0.0)  # the dot, not the padding
        y, x = np.mgrid[: scene.shape[0], : scene.shape[1]]
        centroid = np.array([(x * weight).sum(), (y * weight).sum()]) / weight.sum()
        assert np.allclose(centroid, truth[0], atol=0.2)
    scene, truth = render(image, points, 0.5, (0.0, 0.0), pad=3)
    assert scene.shape == (140, 105, 3) and np.allclose(truth[0], (52.25, 69.75))


def test_motion_and_flicker() -> None:
    assert motion("static_degraded", 5) == (0.0, 0.0)
    assert motion("drift_clean", 2) == pytest.approx((0.74, 0.26))
    assert motion("sway_degraded", 4) == pytest.approx((2.0, 0.0))
    assert motion("fast_degraded", 2) == pytest.approx((3.0, 1.0))
    weights = np.ones((4, 4), np.float32)
    still = [np.full((4, 4, 3), 100, np.uint8)] * 3
    assert flicker(still, weights) == 0.0
    steps = [np.full((4, 4, 3), v, np.uint8) for v in (100, 102, 100)]
    assert flicker(steps, weights) == pytest.approx(2.0)
    assert Tracking.parse("0.3,0.05,2") == Tracking(0.3, 0.05, 2)
    assert Tracking(0.3, 0.05, 2).name == "track:0.3,0.05,2"
    assert Tracking.parse("0.5,0.1,0,0.3") == Tracking(0.5, 0.1, 0, 0.3)
    assert Tracking(0.5, 0.1, 0, 0.3).name == "track:0.5,0.1,0,0.3"
    for text in ("0.3,0.05", "1,0,0,0.3,1"):
        with pytest.raises(ValueError):
            Tracking.parse(text)


def test_clip_and_evaluation_with_a_stand_in_detector(checkpoint: Path) -> None:
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:218, 0:178]
    pattern = 128 + 60 * np.sin(x / 9.0) * np.cos(y / 11.0)
    image = np.repeat(pattern[..., None], 3, axis=2).astype(np.uint8)
    points = np.array([[70.0, 112.0], [108.0, 112.0], [89.0, 135.0], [73.0, 152.0], [105.0, 152.0]])
    for condition in CONDITIONS:
        clip = make_clip(
            image, points, scale=4, hr_size=128, frames=3, condition=condition, rng=rng
        )
        assert len(clip.frames) == len(clip.truth) == len(clip.hr) == 3
        assert clip.hr[0].shape == (128, 128, 3)
    enhancer = Enhancer.from_checkpoint(checkpoint, PipelineConfig())

    class TruthDetector:
        index = 0

        def detect(self, frame: npt.NDArray[np.uint8]) -> list[Face]:
            truth = clip.truth[self.index]
            self.index += 1
            left, top = truth.min(axis=0) - 4
            right, bottom = truth.max(axis=0) + 4
            return [Face((left, top, right - left, bottom - top), truth, 0.9)]

    stub(enhancer, TruthDetector())
    result = evaluate_clip(enhancer, clip, [Tracking(1.0, 0.0, 0)])
    assert set(result) == {"hr", "bicubic", "annotated", "detected", "track:1,0,0"}
    with pytest.raises(ValueError, match="below"):
        evaluate_clip(enhancer, clip, [Tracking(1.0, 0.0, 0, 0.1)])
    # a tracker that neither smooths nor holds changes nothing
    assert result["track:1,0,0"] == result["detected"]
    assert result["hr"]["flicker"] == pytest.approx(0.0, abs=1.0)
    assert result["detected"]["restored"] == 1.0
    assert mean_results([result, result]) == result
