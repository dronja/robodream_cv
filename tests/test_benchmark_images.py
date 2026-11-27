"""The image benchmark script (PR 5.5): its workloads and summaries, without Docker."""

import io
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def script(load_script: Any) -> ModuleType:
    module: ModuleType = load_script("benchmark_images")
    return module


def scenes(folder: Path, singles: int = 16) -> None:
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True)
    for i in range(singles):
        pixels = rng.integers(0, 256, (60, 50, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / f"single_clean_{i:04d}.png")
    Image.fromarray(np.zeros((96, 96, 3), np.uint8)).save(folder / "mosaic_00_clean.png")


def test_a_frame_holds_every_scene_inside_its_cell(script: ModuleType) -> None:
    faces = [np.full((60, 50, 3), 200 + i, np.uint8) for i in range(8)]
    frame = script.make_frame(faces, np.random.default_rng(0))
    assert frame.shape == (1080, 1920, 3) and frame.dtype == np.uint8
    for i in range(8):  # each scene is pasted whole, inside its own cell of the 2 x 4 grid
        top, left = (i // 4) * 540, (i % 4) * 480
        cell = frame[top : top + 540, left : left + 480]
        assert (cell == 200 + i).all(axis=2).sum() == 60 * 50
    again = script.make_frame(faces, np.random.default_rng(0))
    assert np.array_equal(frame, again)


def test_the_workloads_come_from_the_scenes(script: ModuleType, tmp_path: Path) -> None:
    scenes(tmp_path / "scenes_x4")
    args = script.parse_args(
        ["--image", "x", "--label", "t", "--scenes", str(tmp_path), "--frames", "2"]
    )
    work = script.workloads(args, 4)
    assert [(w.name, w.query, len(w.uploads), w.repeat) for w in work] == [
        ("single", "", 16, 1),
        ("mosaic", "", 1, 3),
        ("frame", "", 2, 1),
        ("frame", "format=jpeg", 2, 1),
    ]
    frame = Image.open(io.BytesIO(work[2].uploads[0]))
    assert frame.format == "JPEG" and frame.size == (1920, 1080)
    with pytest.raises(SystemExit, match="no scenes"):
        script.workloads(args, 8)


def test_summaries(script: ModuleType) -> None:
    summary = script.summarize([5.0, *[float(i) for i in range(1, 21)]])
    assert summary["first"] == 5.0 and summary["n"] == 21
    assert summary["median"] == 10.5 and summary["p95"] == 19.0 and summary["max"] == 20.0
    assert script.summarize([2.0])["median"] == 2.0
    assert script.restored_faces("8 faces, 7 restored, 1 too large for the model") == 7
    assert script.restored_faces("0 faces, 0 restored") == 0
