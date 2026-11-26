"""Video (Phase 4, PR 4.5): restore the faces of every frame and write a marked video.

- **Frames** are decoded by OpenCV and restored one by one by ``facehall.pipeline.Enhancer``.
- **Encoding:** the system ``ffmpeg`` encodes the frames through a pipe, H.264 by default
  (``VideoConfig``). A second ffmpeg pass copies the video stream, adds the input's audio and
  writes the marks as container tags (``facehall.marking.video_tags``). OpenCV's own writer has
  no H.264, no audio and no tags.
- **Face tracks:** faces are detected in every frame (D31, D33) and matched to the tracks of the
  previous frames by box overlap (``FaceTracker``). With ``smooth``, each track's landmarks and
  box go through an alpha-beta filter, which damps the detector's frame-to-frame jitter and
  follows constant motion without lag. With ``hold``, a face the detector misses for up to that
  many frames stays restored at its predicted place.
- **Timing:** constant frame rate, taken from the input. Variable-rate inputs are written at
  their mean rate.
"""

import contextlib
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from facehall import marking
from facehall.align.detect import Face, box_iou
from facehall.pipeline import Enhancer, VideoConfig

INPUT_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
OUTPUT_SUFFIXES = (".mp4", ".mov", ".mkv")

Image = npt.NDArray[np.uint8]


def is_video(path: Path) -> bool:
    return path.suffix.lower() in INPUT_SUFFIXES


# --- face tracks -------------------------------------------------------------------------


def face_state(face: Face) -> npt.NDArray[np.float64]:
    """(7, 2): the five landmarks, then the box's top-left and bottom-right corners."""
    x, y, w, h = face.box
    return np.vstack([np.asarray(face.landmarks, dtype=np.float64), [[x, y], [x + w, y + h]]])


def state_face(state: npt.NDArray[np.float64], score: float) -> Face:
    (x0, y0), (x1, y1) = state[5], state[6]
    box = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
    return Face(box=box, landmarks=state[:5].copy(), score=score)


@dataclass
class Track:
    state: npt.NDArray[np.float64]
    score: float
    velocity: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((7, 2)))
    seen: int = 1
    missing: int = 0


class FaceTracker:
    """Matches the faces of consecutive frames, smooths each track and bridges short gaps.

    - **Filter:** an alpha-beta filter per track. The prediction is ``p = x + v``; a new
      detection ``z`` gives ``x = p + alpha (z - p)`` and ``v = v + beta (z - p)``. A new track
      starts at its first detection with zero velocity: a velocity taken from the difference
      of two detections carries their jitter (a growing box pushed a face over the face-size
      limit). ``alpha = 1, beta = 0`` passes the detections through unchanged.
    - **Matching:** detections are matched to the predicted tracks greedily by box IoU (at
      least ``iou``).
    - **Gaps:** a track without a detection moves on to its prediction. For ``hold`` frames it
      is still returned, so its face stays restored; after ``keep`` frames it ends.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        *,
        iou: float = 0.3,
        hold: int = 0,
        keep: int = 2,
    ) -> None:
        if not (0.0 < alpha <= 1.0 and 0.0 <= beta <= 1.0 and 0.0 < iou <= 1.0):
            raise ValueError("tracker alpha and iou must be in (0, 1], beta in [0, 1]")
        if not 0 <= hold <= keep:
            raise ValueError(f"tracker hold must be in [0, keep={keep}], got {hold}")
        self.alpha, self.beta, self.iou, self.hold, self.keep = alpha, beta, iou, hold, keep
        self.tracks: list[Track] = []

    def update(self, faces: Sequence[Face]) -> list[Face]:
        """The tracked faces: one per detection, in the order of ``faces``, then the held
        tracks that had no detection in this frame."""
        predicted = [t.state + t.velocity for t in self.tracks]
        pairs = sorted(
            (
                (box_iou(state_face(p, 0.0).box, f.box), i, j)
                for i, p in enumerate(predicted)
                for j, f in enumerate(faces)
            ),
            reverse=True,
        )
        matched: dict[int, int] = {}  # face -> track
        used: set[int] = set()
        for overlap, i, j in pairs:
            if overlap < self.iou:
                break
            if i not in used and j not in matched:
                matched[j] = i
                used.add(i)
        out: list[Face] = []
        kept: list[Track] = []
        for j, face in enumerate(faces):
            z = face_state(face)
            if j not in matched:
                track = Track(z, face.score)
            else:
                track = self.tracks[matched[j]]
                p = predicted[matched[j]]
                residual = z - p
                track.state = p + self.alpha * residual
                track.velocity = track.velocity + self.beta * residual
                track.seen, track.missing, track.score = track.seen + 1, 0, face.score
            kept.append(track)
            out.append(state_face(track.state, face.score))
        for i, track in enumerate(self.tracks):
            if i in used or track.missing >= self.keep:
                continue
            track.state, track.missing = predicted[i], track.missing + 1
            kept.append(track)
            if track.missing <= self.hold:
                out.append(state_face(track.state, track.score))
        self.tracks = kept
        return out


def tracker(config: VideoConfig) -> FaceTracker | None:
    """The face tracker of ``config``, or None when it neither smooths nor holds."""
    if not config.smooth and config.hold == 0:
        return None
    alpha, beta = (config.alpha, config.beta) if config.smooth else (1.0, 0.0)
    return FaceTracker(
        alpha, beta, iou=config.track_iou, hold=config.hold, keep=max(2, config.hold)
    )


# --- reading and writing ---------------------------------------------------------------


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int  # as the container states it; may be approximate


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"{path}: cannot open the video")
    info = VideoInfo(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.width <= 0 or info.height <= 0 or not info.fps > 0:
        capture.release()
        raise ValueError(f"{path}: no video stream with a frame size and rate")
    return capture, info


def read_frames(capture: cv2.VideoCapture) -> Iterator[Image]:
    """RGB frames until the end of the stream."""
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield np.ascontiguousarray(frame[:, :, ::-1], dtype=np.uint8)


def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FileNotFoundError("video needs the ffmpeg program on PATH (e.g. apt install ffmpeg)")
    return path


def encode_command(
    target: Path, size: tuple[int, int], fps: float, config: VideoConfig
) -> list[str]:
    """ffmpeg reading RGB frames from stdin and encoding them to ``target`` (video only)."""
    return [
        ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        repr(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        config.codec,
        "-crf",
        str(config.crf),
        "-preset",
        config.preset,
        "-pix_fmt",
        "yuv420p",
        str(target),
    ]


def mux_command(
    video: Path, target: Path, *, audio_from: Path | None, tags: dict[str, str]
) -> list[str]:
    """ffmpeg copying the encoded ``video``, the audio of ``audio_from`` and the ``tags``."""
    command = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video)]
    if audio_from is not None:
        command += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a?"]
    command += ["-c:v", "copy"]
    if audio_from is not None:
        # MP4 and MOV take a few audio codecs only, so audio is re-encoded there.
        command += ["-c:a", "copy"] if target.suffix.lower() == ".mkv" else ["-c:a", "aac"]
    if target.suffix.lower() in (".mp4", ".mov"):
        command += ["-movflags", "+use_metadata_tags"]
    for key, value in tags.items():
        command += ["-metadata", f"{key}={value}"]
    return [*command, str(target)]


def run(command: list[str], stdin: Iterator[bytes] | None = None) -> None:
    """Run ffmpeg, feeding it ``stdin``. An error raised while producing the input propagates
    after ffmpeg has been stopped."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE if stdin is not None else None, stderr=errors
        )
        try:
            # BrokenPipeError: ffmpeg stopped reading; its exit code says why
            with contextlib.suppress(BrokenPipeError):
                if stdin is not None and process.stdin is not None:
                    for chunk in stdin:
                        process.stdin.write(chunk)
        finally:
            if process.stdin is not None:
                with contextlib.suppress(BrokenPipeError):
                    process.stdin.close()
            code = process.wait()
        if code != 0:
            errors.seek(0)
            message = errors.read().decode(errors="replace").strip().splitlines()
            raise OSError(f"ffmpeg failed ({code}): {' / '.join(message[-3:])}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(path: Path) -> dict[str, Any] | None:
    """The facehall record in the tags of a marked video (needs ffprobe), or None."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("reading video tags needs the ffprobe program on PATH")
    output = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    tags = {k.lower(): v for k, v in json.loads(output).get("format", {}).get("tags", {}).items()}
    value = tags.get("facehall")
    return None if value is None else dict(json.loads(value))


# --- the video loop --------------------------------------------------------------------


@dataclass
class VideoStats:
    frames: int = 0
    frames_with_faces: int = 0
    faces: int = 0
    restored: int = 0

    def summary(self) -> str:
        return (
            f"{self.frames} frames, {self.faces} faces in {self.frames_with_faces} frames, "
            f"{self.restored} restored"
        )


def enhance_video(
    enhancer: Enhancer,
    source: Path,
    target: Path,
    *,
    parameters: dict[str, Any],
    visible_mark: bool = False,
    report: Callable[[str], None] = print,
) -> VideoStats:
    """Restore the faces of every frame of ``source`` and write ``target`` with its marks."""
    if target.suffix.lower() not in OUTPUT_SUFFIXES:
        raise ValueError(f"{target}: unsupported video format; use one of {OUTPUT_SUFFIXES}")
    config = enhancer.config.video
    capture, info = open_video(source)
    stats = VideoStats()
    tracks = tracker(config)
    size = (info.width * enhancer.scale, info.height * enhancer.scale)

    def frames() -> Iterator[bytes]:
        for frame in read_frames(capture):
            faces = enhancer.detector.detect(frame)
            if tracks is not None:
                faces = tracks.update(faces)
            result = enhancer.enhance(frame, detections=faces)
            out = marking.watermark(result.image) if visible_mark else result.image
            stats.frames += 1
            stats.faces += len(result.faces)
            stats.frames_with_faces += bool(result.faces)
            stats.restored += sum(f.restored for f in result.faces)
            if stats.frames % 250 == 0:
                report(f"{source}: {stats.frames} of about {info.frames} frames")
            yield out.tobytes()

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        encoded = Path(scratch) / ("video" + target.suffix)
        try:
            run(encode_command(encoded, size, info.fps, config), frames())
        finally:
            capture.release()
        if stats.frames == 0:
            raise ValueError(f"{source}: no frames could be read")
        rec = marking.video_record(
            model=enhancer.info,
            parameters=parameters,
            input_sha256=file_sha256(source),
            input_size=(info.width, info.height),
            output_size=size,
            video={"fps": info.fps, "frames": stats.frames, "audio": config.audio},
            faces={
                "detected": stats.faces,
                "restored": stats.restored,
                "frames_with_faces": stats.frames_with_faces,
            },
            watermark=visible_mark,
        )
        partial = Path(scratch) / ("marked" + target.suffix)
        audio = source if config.audio else None
        run(mux_command(encoded, partial, audio_from=audio, tags=marking.video_tags(rec)))
        partial.replace(target)
    return stats
