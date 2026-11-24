"""Tests for facehall.data.wider on synthetic WIDER FACE archives."""

import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import pytest
from PIL import Image

from facehall.align import CELEBA_TEMPLATE
from facehall.data.manifest import UNKNOWN_IDENTITY, read_manifest
from facehall.data.wider import (
    IMAGE_PREFIX,
    IMAGES_ARCHIVE,
    INFO_FILE,
    OFFICIAL_FILE,
    RETINAFACE_ARCHIVE,
    RETINAFACE_FILE,
    SPLIT_ARCHIVE,
    check_archives,
    parse_official,
    parse_retinaface,
    prepare_wider,
    split_of,
)

SIZE = (120, 100)  # (width, height) of the synthetic photos


def landmarks(side: float, dx: float, dy: float) -> npt.NDArray[np.float64]:
    """Landmarks that map exactly onto the template at crop side ``side`` after a shift."""
    points: npt.NDArray[np.float64] = np.asarray(CELEBA_TEMPLATE) * side - 0.5 + np.array([dx, dy])
    return points


# image -> faces: (box, landmarks or None, attributes, official box override)
FACES: dict[str, list[tuple[tuple[int, int, int, int], object, tuple[int, ...]]]] = {
    "0--Parade/a.jpg": [
        ((10, 20, 32, 32), landmarks(32, 10, 20), (1, 0, 0, 0, 0, 0)),  # x4, native
        ((60, 5, 16, 16), landmarks(16, 60, 5), (2, 0, 1, 0, 1, 0)),  # x8, native
        ((80, 60, 10, 10), None, (0, 0, 0, 0, 0, 0)),  # no landmarks
    ],
    "1--Handshaking/b.jpg": [
        ((5, 5, 32, 32), landmarks(32, 5, 5), (0, 0, 0, 1, 0, 0)),  # invalid
        ((40, 40, 32, 32), landmarks(32, 40, 40), (0, 0, 0, 0, 0, 0)),  # box differs below
        ((0, 0, 64, 64), landmarks(64, 0, 0), (0, 0, 0, 0, 0, 0)),  # too large for both
    ],
    "1--Handshaking/c.jpg": [
        ((-10, 0, 32, 32), landmarks(32, -10, 0), (0, 1, 0, 0, 2, 1)),  # x4, padded
    ],
}
OFFICIAL_BOX_CHANGES = {("1--Handshaking/b.jpg", 1): (41, 40, 32, 32)}


def photo(seed: int) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(SIZE[1], SIZE[0], 3), dtype=np.uint8)


def write_raw(raw: Path) -> dict[str, npt.NDArray[np.uint8]]:
    """Write the three archives; return the pixels of every photo."""
    raw.mkdir(parents=True, exist_ok=True)
    pixels = {name: photo(i) for i, name in enumerate(FACES)}
    official, label = [], []
    for name, faces in FACES.items():
        official += [name, str(len(faces))]
        label.append(f"# {name}")
        for index, (box, points, attributes) in enumerate(faces):
            official_box = OFFICIAL_BOX_CHANGES.get((name, index), box)
            official.append(" ".join(map(str, (*official_box, *attributes))) + " ")
            if points is None:
                values = ["-1.0"] * 15
            else:
                values = [f"{v:.6f}" for p in np.asarray(points) for v in (*p, 0.0)]
            label.append(" ".join([*map(str, box), *values, "0.5"]))
    official += ["2--Demonstration/empty.jpg", "0", "0 0 0 0 0 0 0 0 0 0 "]
    label.append("# 2--Demonstration/empty.jpg")
    with zipfile.ZipFile(raw / IMAGES_ARCHIVE, "w") as archive:
        for name, array in pixels.items():
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")  # lossless, despite the suffix
            archive.writestr(IMAGE_PREFIX + name, buffer.getvalue())
        archive.writestr(IMAGE_PREFIX + "2--Demonstration/empty.jpg", b"")
    with zipfile.ZipFile(raw / SPLIT_ARCHIVE, "w") as archive:
        archive.writestr(OFFICIAL_FILE, "\n".join(official) + "\n")
    with zipfile.ZipFile(raw / RETINAFACE_ARCHIVE, "w") as archive:
        archive.writestr(RETINAFACE_FILE, "\n".join(label) + "\n")
    return pixels


def test_parse_official_handles_images_without_faces() -> None:
    text = "a/x.jpg\n0\n0 0 0 0 0 0 0 0 0 0 \nb/y.jpg\n1\n1 2 3 4 2 0 1 0 1 0 \n"
    parsed = parse_official(text)
    assert parsed["a/x.jpg"] == []
    assert parsed["b/y.jpg"][0].box == (1, 2, 3, 4)
    assert parsed["b/y.jpg"][0].attributes == (2, 0, 1, 0, 1, 0)


def test_parse_retinaface_reads_landmarks() -> None:
    text = "# a/x.jpg\n1 2 3 4 " + " ".join(["5.5 6.5 0.0"] * 5) + " 0.9\n# b/y.jpg\n"
    text += "1 1 2 2 " + " ".join(["-1.0"] * 15) + " -1.0\n"
    parsed = parse_retinaface(text)
    face = parsed["a/x.jpg"][0]
    assert face.landmarks is not None
    np.testing.assert_array_equal(face.landmarks, np.tile([5.5, 6.5], (5, 1)))
    assert parsed["b/y.jpg"][0].landmarks is None


def test_split_is_stable_and_follows_the_fraction() -> None:
    names = [f"e/{i}.jpg" for i in range(200)]
    splits = [split_of(n, 0.5) for n in names]
    assert splits == [split_of(n, 0.5) for n in names]
    assert 60 < splits.count("test") < 140
    assert {split_of(n, 0.0) for n in names} == {"val"}
    assert {split_of(n, 1.0) for n in names} == {"test"}


def test_prepare_writes_native_crops(tmp_path: Path) -> None:
    pixels = write_raw(tmp_path / "raw")
    out = tmp_path / "out"
    info = prepare_wider(tmp_path / "raw", out, expected=None)

    table = read_manifest(out)
    rows = {(r["source_image"], r["face_index"], r["scale"]): r for r in table.to_pylist()}
    assert set(rows) == {
        ("0--Parade/a.jpg", 0, 4),
        ("0--Parade/a.jpg", 1, 8),
        ("1--Handshaking/c.jpg", 0, 4),
    }
    for row in rows.values():
        assert row["split"] == split_of(row["source_image"], 0.5)
        assert row["identity_id"] == UNKNOWN_IDENTITY
        assert row["width"] == row["height"] == 128 // row["scale"]
        assert row["path"].startswith(f"x{row['scale']}/{row['split']}/")
        assert row["align_scale"] == pytest.approx(1.0, abs=1e-5)
        assert row["align_residual"] == pytest.approx(0.0, abs=1e-5)

    # Scale 1 and an integer shift: the crop is the source region itself.
    x4 = rows[("0--Parade/a.jpg", 0, 4)]
    crop = np.asarray(Image.open(out / x4["path"]))
    np.testing.assert_array_equal(crop, pixels["0--Parade/a.jpg"][20:52, 10:42])
    assert x4["padded"] == 0.0
    assert (x4["attr_blur"], x4["attr_occlusion"]) == (1, 0)

    x8 = rows[("0--Parade/a.jpg", 1, 8)]
    crop = np.asarray(Image.open(out / x8["path"]))
    np.testing.assert_array_equal(crop, pixels["0--Parade/a.jpg"][5:21, 60:76])
    assert (x8["attr_blur"], x8["attr_illumination"], x8["attr_occlusion"]) == (2, 1, 1)

    padded = rows[("1--Handshaking/c.jpg", 0, 4)]
    assert padded["padded"] == pytest.approx(10 / 32)
    assert (padded["attr_expression"], padded["attr_pose"]) == (1, 1)
    assert padded["event"] == "1--Handshaking"

    assert info["annotations"] == {
        "images": 4,
        "faces": 7,
        "faces_with_landmarks": 6,
        "matched": 5,
        "no_official_match": 1,
    }
    assert info["selection"] == {
        "x4": {"kept": 2, "invalid": 1, "size": 2},
        "x8": {"kept": 1, "size": 4},
    }
    assert json.loads((out / INFO_FILE).read_text())["manifest"]["rows"] == 3
    assert pc.sum(table.column("scale")).as_py() == 16


def test_prepare_is_deterministic(tmp_path: Path) -> None:
    write_raw(tmp_path / "raw")
    first = prepare_wider(tmp_path / "raw", tmp_path / "a", expected=None)
    second = prepare_wider(tmp_path / "raw", tmp_path / "b", expected=None)
    assert first["manifest"]["content_sha256"] == second["manifest"]["content_sha256"]  # type: ignore[index]


def test_prepare_checks_archives_and_output(tmp_path: Path) -> None:
    write_raw(tmp_path / "raw")
    digests = check_archives(tmp_path / "raw", expected=None)
    wrong = {**digests, SPLIT_ARCHIVE: "0" * 64}
    with pytest.raises(ValueError, match="differs"):
        check_archives(tmp_path / "raw", expected=wrong)
    with pytest.raises(FileNotFoundError, match="missing raw archive"):
        check_archives(tmp_path / "missing", expected=None)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "file").write_text("x")
    with pytest.raises(FileExistsError):
        prepare_wider(tmp_path / "raw", tmp_path / "out", expected=None)


def test_script_runs(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    write_raw(tmp_path / "raw")
    script = load_script("prepare_wider")
    argv = ["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out"), "--no-checksum"]
    assert script.main(argv) == 0
    assert (tmp_path / "out" / INFO_FILE).is_file()
