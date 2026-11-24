"""Real-world low-resolution faces from WIDER FACE, aligned with the RetinaFace landmarks.

Raw inputs (docs/DATA.md §7; terms in docs/THIRD_PARTY_LICENSES.md §2), read from the archives
without extracting them:

- ``WIDER_train.zip``: the WIDER FACE training images (CC BY-NC-ND), downloaded from the Hugging
  Face dataset ``CUHK-CSE/wider_face`` that the official page links to;
- ``wider_face_split.zip``: the official boxes and per-face attributes;
- ``retinaface_gt.zip``: the RetinaFace annotations (InsightFace), the same boxes plus five
  manually annotated landmarks. Landmarks are released for the training images only, so the set
  is built from those; nothing in this project trains on WIDER FACE.

A face is kept for scale ``s`` when its native size already matches the model input: the
similarity that maps its landmarks to the template at the LR size ``hr_size / s`` must scale
the source by a factor within ``native_range`` (output pixels per source pixel). Each kept face
is warped once to its LR crop and stored losslessly as PNG. There is no HR reference.

Faces are split by source image, so all faces of one photo share a split: ``val`` is for tuning
(for example the degradation ranges), ``test`` is frozen. WIDER FACE has no identities, so the
manifest stores ``UNKNOWN_IDENTITY``.
"""

import hashlib
import io
import json
import logging
import time
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from PIL import Image

import facehall
from facehall.align import CELEBA_TEMPLATE, alignment_matrices, valid_mask, warp
from facehall.data.manifest import (
    MANIFEST_FILE,
    REQUIRED_FIELDS,
    UNKNOWN_IDENTITY,
    content_sha256,
    write_manifest,
)
from facehall.provenance import file_sha256, git_commit, package_versions

log = logging.getLogger(__name__)

IMAGES_ARCHIVE = "WIDER_train.zip"
SPLIT_ARCHIVE = "wider_face_split.zip"
RETINAFACE_ARCHIVE = "retinaface_gt.zip"

#: sha256 of the raw archives as downloaded on 2026-11-24 (docs/DATA.md §7.1).
ARCHIVES: dict[str, str] = {
    IMAGES_ARCHIVE: "e23b76129c825cafae8be944f65310b2e1ba1c76885afe732f179c41e5ed6d59",
    SPLIT_ARCHIVE: "c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c",
    RETINAFACE_ARCHIVE: "180bacc15fe01ae02c072daf25a9601a3631f4dd3063ac1ffb77133d10b21979",
}

IMAGE_PREFIX = "WIDER_train/images/"
OFFICIAL_FILE = "wider_face_split/wider_face_train_bbx_gt.txt"
RETINAFACE_FILE = "train/label.txt"

#: Per-face attributes of the official annotation, in file order after the box.
#: Values: blur 0 clear / 1 normal / 2 heavy; expression 0 typical / 1 exaggerated;
#: illumination 0 normal / 1 extreme; invalid 0 / 1; occlusion 0 none / 1 partial / 2 heavy;
#: pose 0 typical / 1 atypical (``wider_face_split/readme.txt``).
ATTRIBUTE_NAMES = ("blur", "expression", "illumination", "invalid", "occlusion", "pose")
STORED_ATTRIBUTES = tuple(name for name in ATTRIBUTE_NAMES if name != "invalid")

SOURCE = "wider-face-train+retinaface"
LICENSE = (
    "WIDER FACE: CC BY-NC-ND 4.0; RetinaFace landmarks: non-commercial research only; "
    "no redistribution of derived crops"
)
SPLIT_NAMES = ("val", "test")  # val: tuning; test: frozen
DATASET_DIR = "x{scale}"
INFO_FILE = "prepare_info.json"
INTERPOLATION = "bicubic"
BORDER = "reflect"  # as for CelebA (D16); the padded fraction is stored

# Defaults, recorded in prepare_info.json (docs/DECISIONS.md, 2026-11-24, real-world test set).
SCALES = (4, 8)
HR_SIZE = 128
NATIVE_RANGE = (0.8, 1.25)
TEST_FRACTION = 0.5

LANDMARK_VALUES = 10  # x, y of: left eye, right eye, nose, left and right mouth corner


@dataclass(frozen=True)
class OfficialFace:
    box: tuple[int, int, int, int]  # x, y, w, h
    attributes: tuple[int, ...]  # ATTRIBUTE_NAMES order


@dataclass(frozen=True)
class AnnotatedFace:
    box: tuple[float, float, float, float]
    landmarks: npt.NDArray[np.float64] | None  # (5, 2), pixel-centre coordinates


@dataclass(frozen=True)
class Face:
    """A face with landmarks and a matching official box."""

    image: str  # ``<event>/<file>.jpg``, as in the annotations
    index: int  # position in the image's annotation list
    box: tuple[int, int, int, int]
    landmarks: npt.NDArray[np.float64]
    attributes: dict[str, int]


@dataclass(frozen=True)
class Selected:
    face: Face
    scale: int
    split: str
    matrix: npt.NDArray[np.float64]  # source -> LR crop
    align_scale: float
    align_angle: float
    residual: float  # RMS landmark distance to the template, as a fraction of the crop side


def parse_official(text: str) -> dict[str, list[OfficialFace]]:
    """``wider_face_*_bbx_gt.txt``: image name, face count, then one line per face."""
    lines = text.splitlines()
    images: dict[str, list[OfficialFace]] = {}
    i = 0
    while i < len(lines):
        name, count = lines[i].strip(), int(lines[i + 1])
        faces = []
        for line in lines[i + 2 : i + 2 + count]:
            values = [int(v) for v in line.split()]
            if len(values) != 4 + len(ATTRIBUTE_NAMES):
                raise ValueError(f"{name}: unexpected face line {line!r}")
            x, y, w, h = values[:4]
            faces.append(OfficialFace(box=(x, y, w, h), attributes=tuple(values[4:])))
        images[name] = faces
        i += 2 + max(count, 1)  # an image without faces still has one placeholder line
    return images


def parse_retinaface(text: str) -> dict[str, list[AnnotatedFace]]:
    """RetinaFace ``label.txt``: ``# <image>``, then one line per face.

    A face line holds the box (x, y, w, h), then five landmarks as (x, y, flag), then one score.
    A landmark x of -1 means the face has no landmarks. The flag and the score are not
    documented and are not used.
    """
    images: dict[str, list[AnnotatedFace]] = {}
    current: list[AnnotatedFace] | None = None
    for line in text.splitlines():
        if line.startswith("#"):
            current = images.setdefault(line[1:].strip(), [])
            continue
        if not line.strip():
            continue
        if current is None:
            raise ValueError("face line before the first image name")
        values = [float(v) for v in line.split()]
        box = (values[0], values[1], values[2], values[3])
        landmarks = None
        if len(values) >= 4 + 3 * 5 and values[4] >= 0:
            landmarks = np.array(values[4:19], dtype=np.float64).reshape(5, 3)[:, :2]
        current.append(AnnotatedFace(box=box, landmarks=landmarks))
    return images


def match_faces(
    official: Mapping[str, Sequence[OfficialFace]],
    retinaface: Mapping[str, Sequence[AnnotatedFace]],
) -> tuple[list[Face], Counter[str]]:
    """Faces with landmarks whose box equals the official box at the same position.

    Returns the faces in (image, index) order and counts of the faces left out.
    """
    faces: list[Face] = []
    skipped: Counter[str] = Counter()
    for image in sorted(retinaface):
        reference = official.get(image, [])
        for index, face in enumerate(retinaface[image]):
            if face.landmarks is None:
                skipped["no_landmarks"] += 1
                continue
            box = tuple(int(v) for v in face.box)
            if index >= len(reference) or reference[index].box != box:
                skipped["no_official_match"] += 1
                continue
            attributes = dict(zip(ATTRIBUTE_NAMES, reference[index].attributes, strict=True))
            faces.append(
                Face(
                    image=image,
                    index=index,
                    box=reference[index].box,
                    landmarks=face.landmarks,
                    attributes=attributes,
                )
            )
    return faces, skipped


def split_of(image: str, test_fraction: float) -> str:
    """``test`` or ``val`` from the sha256 of the image name: stable and order independent."""
    value = int(hashlib.sha256(image.encode()).hexdigest()[:8], 16) / 2**32
    return "test" if value < test_fraction else "val"


def select(
    faces: Sequence[Face],
    scale: int,
    *,
    hr_size: int = HR_SIZE,
    native_range: tuple[float, float] = NATIVE_RANGE,
    test_fraction: float = TEST_FRACTION,
) -> tuple[list[Selected], Counter[str]]:
    """Faces whose native size suits ``scale``, without the ``invalid`` ones."""
    if hr_size % scale:
        raise ValueError(f"HR size {hr_size} is not divisible by the scale {scale}")
    low, high = native_range
    if not 0 < low <= high:
        raise ValueError(f"native range must satisfy 0 < low <= high, got {native_range}")
    if not faces:
        return [], Counter()
    size = hr_size // scale
    landmarks = np.stack([f.landmarks for f in faces])
    matrices = alignment_matrices(landmarks, CELEBA_TEMPLATE, size)
    linear = matrices[:, :, :2]
    factors = np.sqrt(np.abs(np.linalg.det(linear)))
    angles = np.degrees(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
    mapped = np.einsum("bij,bkj->bki", linear, landmarks) + matrices[:, None, :, 2]
    target = np.asarray(CELEBA_TEMPLATE) * size - 0.5
    residuals = np.sqrt(((mapped - target) ** 2).sum(axis=-1).mean(axis=-1)) / size

    selected: list[Selected] = []
    skipped: Counter[str] = Counter()
    for i, face in enumerate(faces):
        if not low <= factors[i] <= high:
            skipped["size"] += 1
        elif face.attributes["invalid"]:
            skipped["invalid"] += 1
        else:
            selected.append(
                Selected(
                    face=face,
                    scale=scale,
                    split=split_of(face.image, test_fraction),
                    matrix=matrices[i],
                    align_scale=float(factors[i]),
                    align_angle=float(angles[i]),
                    residual=float(residuals[i]),
                )
            )
    return selected, skipped


def manifest_schema() -> pa.Schema:
    return pa.schema(
        [
            *REQUIRED_FIELDS,
            pa.field("scale", pa.int32(), nullable=False),
            pa.field("source_image", pa.string(), nullable=False),
            pa.field("event", pa.string(), nullable=False),
            pa.field("face_index", pa.int32(), nullable=False),
            pa.field("bbox", pa.list_(pa.int32(), 4), nullable=False),
            pa.field("landmarks", pa.list_(pa.float32(), LANDMARK_VALUES), nullable=False),
            pa.field("align_scale", pa.float32(), nullable=False),
            pa.field("align_angle", pa.float32(), nullable=False),
            pa.field("align_residual", pa.float32(), nullable=False),
            pa.field("padded", pa.float32(), nullable=False),
            *(pa.field(f"attr_{name}", pa.int8(), nullable=False) for name in STORED_ATTRIBUTES),
        ]
    )


def check_archives(raw: Path, expected: Mapping[str, str] | None = ARCHIVES) -> dict[str, str]:
    """sha256 of every raw archive; raise ValueError if one differs from ``expected``."""
    digests = {}
    for name in (IMAGES_ARCHIVE, SPLIT_ARCHIVE, RETINAFACE_ARCHIVE):
        path = raw / name
        if not path.is_file():
            raise FileNotFoundError(f"missing raw archive {path}")
        digests[name] = file_sha256(path)
        if expected is not None and digests[name] != expected[name]:
            raise ValueError(f"{name}: sha256 {digests[name]} differs from {expected[name]}")
    return digests


def prepare_wider(
    raw: Path,
    out: Path,
    *,
    scales: Sequence[int] = SCALES,
    hr_size: int = HR_SIZE,
    native_range: tuple[float, float] = NATIVE_RANGE,
    test_fraction: float = TEST_FRACTION,
    expected: Mapping[str, str] | None = ARCHIVES,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write the LR crops, the manifest and ``prepare_info.json`` to ``out``.

    Crops go to ``out/x{scale}/{split}/{index:06d}.png``; ``index`` counts faces within the
    scale and split in (image, face) order. Pass ``expected=None`` to skip the archive checksums
    (synthetic test data). Returns the content of ``prepare_info.json``.
    """
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; use overwrite to write into it")
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError(f"test fraction must be in [0, 1], got {test_fraction}")
    digests = check_archives(raw, expected)

    with zipfile.ZipFile(raw / SPLIT_ARCHIVE) as archive:
        official = parse_official(archive.read(OFFICIAL_FILE).decode())
    with zipfile.ZipFile(raw / RETINAFACE_ARCHIVE) as archive:
        retinaface = parse_retinaface(archive.read(RETINAFACE_FILE).decode())
    faces, skipped = match_faces(official, retinaface)
    annotations = {
        "images": len(retinaface),
        "faces": sum(len(v) for v in retinaface.values()),
        "faces_with_landmarks": len(faces) + skipped["no_official_match"],
        "matched": len(faces),
        "no_official_match": skipped["no_official_match"],
    }
    log.info("annotations: %s", annotations)

    chosen: list[Selected] = []
    selection: dict[str, dict[str, int]] = {}
    for scale in scales:
        picked, left_out = select(
            faces,
            scale,
            hr_size=hr_size,
            native_range=native_range,
            test_fraction=test_fraction,
        )
        chosen += picked
        selection[f"x{scale}"] = {"kept": len(picked), **dict(sorted(left_out.items()))}
    chosen.sort(key=lambda s: (s.scale, SPLIT_NAMES.index(s.split), s.face.image, s.face.index))

    paths = []
    counters: Counter[tuple[int, str]] = Counter()
    for item in chosen:
        key = (item.scale, item.split)
        paths.append(f"{DATASET_DIR.format(scale=item.scale)}/{item.split}/{counters[key]:06d}.png")
        counters[key] += 1

    start = time.monotonic()
    rows = _write_crops(raw / IMAGES_ARCHIVE, out, chosen, paths, hr_size)
    log.info("%d crops in %.1f s", len(chosen), time.monotonic() - start)

    table = pa.Table.from_pylist(rows, schema=manifest_schema())
    manifest_path = write_manifest(table, out)
    info: dict[str, object] = {
        "dataset": SOURCE,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw_dir": str(raw),
        "archives": digests,
        "git_commit": git_commit(),
        "versions": {
            "facehall": facehall.__version__,
            **package_versions(["numpy", "opencv-python-headless", "pillow", "pyarrow"]),
        },
        "settings": {
            "scales": list(scales),
            "hr_size": hr_size,
            "native_range": list(native_range),
            "test_fraction": test_fraction,
            "template": "CELEBA_TEMPLATE",
            "interpolation": INTERPOLATION,
            "border": BORDER,
        },
        "annotations": annotations,
        "selection": selection,
        "splits": _summary(chosen, rows),
        "manifest": {
            "file": MANIFEST_FILE,
            "rows": table.num_rows,
            "content_sha256": content_sha256(table),
            "file_sha256": file_sha256(manifest_path),
        },
    }
    (out / INFO_FILE).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def _write_crops(
    images: Path, out: Path, chosen: Sequence[Selected], paths: Sequence[str], hr_size: int
) -> list[dict[str, Any]]:
    """Decode each source image once, warp its faces and write the PNG crops."""
    by_image: dict[str, list[int]] = {}
    for i, item in enumerate(chosen):
        by_image.setdefault(item.face.image, []).append(i)
    rows: list[dict[str, Any] | None] = [None] * len(chosen)
    for directory in {str(Path(p).parent) for p in paths}:
        (out / directory).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(images) as archive:
        for image_name in sorted(by_image):
            data = archive.read(IMAGE_PREFIX + image_name)
            with Image.open(io.BytesIO(data)) as decoded:
                if decoded.mode != "RGB":
                    raise ValueError(f"{image_name}: expected an RGB image, got {decoded.mode}")
                source = np.asarray(decoded)
            height, width = source.shape[:2]
            for i in by_image[image_name]:
                item = chosen[i]
                size = hr_size // item.scale
                crop = warp(source, item.matrix, size, interpolation=INTERPOLATION, border=BORDER)
                padded = 1.0 - float(valid_mask(item.matrix, (width, height), size).mean())
                buffer = io.BytesIO()
                Image.fromarray(np.ascontiguousarray(crop)).save(buffer, format="PNG")
                encoded = buffer.getvalue()
                (out / paths[i]).write_bytes(encoded)
                face = item.face
                rows[i] = {
                    "path": paths[i],
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "split": item.split,
                    "identity_id": UNKNOWN_IDENTITY,
                    "width": size,
                    "height": size,
                    "source": SOURCE,
                    "license": LICENSE,
                    "scale": item.scale,
                    "source_image": face.image,
                    "event": face.image.split("/", 1)[0],
                    "face_index": face.index,
                    "bbox": list(face.box),
                    "landmarks": face.landmarks.reshape(-1).tolist(),
                    "align_scale": item.align_scale,
                    "align_angle": item.align_angle,
                    "align_residual": item.residual,
                    "padded": padded,
                    **{f"attr_{name}": face.attributes[name] for name in STORED_ATTRIBUTES},
                }
    if any(row is None for row in rows):
        raise RuntimeError("some crops were not written")
    return [row for row in rows if row is not None]


def _summary(chosen: Sequence[Selected], rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Per scale and split: faces, images, events, blur and occlusion levels, padded crops."""
    summary: dict[str, object] = {}
    for scale in sorted({s.scale for s in chosen}):
        per_split: dict[str, object] = {}
        for split in SPLIT_NAMES:
            picked = [i for i, s in enumerate(chosen) if s.scale == scale and s.split == split]
            faces = [chosen[i].face for i in picked]
            per_split[split] = {
                "faces": len(picked),
                "images": len({f.image for f in faces}),
                "events": len({f.image.split("/", 1)[0] for f in faces}),
                "blur": _levels(faces, "blur", 3),
                "occlusion": _levels(faces, "occlusion", 3),
                "padded": sum(1 for i in picked if rows[i]["padded"] > 0),
            }
        summary[f"x{scale}"] = per_split
    return summary


def _levels(faces: Sequence[Face], name: str, count: int) -> list[int]:
    values = Counter(f.attributes[name] for f in faces)
    return [values[level] for level in range(count)]
