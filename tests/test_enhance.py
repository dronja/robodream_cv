"""``facehall enhance`` (facehall.enhance) and the output marks (facehall.marking).

The model is a tiny untrained SRResNet saved as a checkpoint, and faces are given as landmarks,
so no weights, detector or dataset are needed.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from facehall import __version__, marking
from facehall.align import CELEBA_TEMPLATE, Face
from facehall.cli import main
from facehall.enhance import Job, enhance_files, numbers, parameters, plan, summary
from facehall.models import build_model
from facehall.pipeline import Enhancer, FaceResult, PipelineConfig
from facehall.train import load_experiment
from facehall.train.checkpoint import save_checkpoint
from facehall.train.config import experiment_to_dict

# A face with 8 px between the eyes (the x4 input size), centred in a 64 x 64 image; CelebA
# landmark convention, as `--landmarks` takes it.
FACE = CELEBA_TEMPLATE * 32 - 0.5 + 16.0
LANDMARKS = ",".join(f"{v:.3f}" for v in FACE.ravel())


class NoFaces:
    def detect(self, image: np.ndarray) -> list[Face]:
        return []


def rgb(size: tuple[int, int] = (64, 48), seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def sample_record(**changes: Any) -> dict[str, Any]:
    face = {
        "source": "given",
        "score": None,
        "restored": True,
        "reason": None,
        "region": [1, 2, 3, 4],
        "landmarks": [[0.0, 0.0]] * 5,
    }
    values: dict[str, Any] = {
        "model": {"name": "run-1", "sha256": "ab" * 32, "scale": 4},
        "parameters": {"max_downscale": 2.0},
        "input_sha256": "cd" * 32,
        "input_size": (16, 12),
        "output_size": (64, 48),
        "faces_from": "landmarks",
        "watermark": False,
        "created": "2026-11-26T00:00:00Z",
    }
    values.update(changes)
    return marking.record([face], **values)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = load_experiment(
        "configs/train/srresnet_x4.yaml", ["model.residual_blocks=1", "model.channels=8"]
    )
    model = build_model(config.model, config.scale, config.data.hr_size // config.scale)
    path = tmp_path_factory.mktemp("run") / "best.safetensors"
    save_checkpoint(path, model, {"config": experiment_to_dict(config), "run": "tiny_x4-test"})
    return path


def test_record_has_no_landmarks_or_file_names() -> None:
    rec = sample_record()
    assert rec["ai_enhanced"] is True and rec["notice"] == marking.NOTICE
    assert rec["software"] == {"name": "facehall", "version": __version__}
    assert rec["input"] == {
        "sha256": "cd" * 32,
        "width": 16,
        "height": 12,
        "faces_from": "landmarks",
    }
    assert rec["faces"] == [
        {"source": "given", "score": None, "restored": True, "reason": None, "region": [1, 2, 3, 4]}
    ]
    assert '"landmarks":' not in json.dumps(rec)


def test_xmp_packet() -> None:
    packet = marking.xmp_packet(sample_record(model={"name": 'a<b"&', "sha256": "x"}))
    assert packet.startswith("<?xpacket begin=") and packet.endswith('<?xpacket end="w"?>')
    assert f'Iptc4xmpExt:DigitalSourceType="{marking.DIGITAL_SOURCE_TYPE}"' in packet
    assert f'xmlns:Iptc4xmpExt="{marking.IPTC_EXT_NS}"' in packet
    assert 'facehall:Model="a&lt;b&quot;&amp;"' in packet  # escaped attribute
    assert 'facehall:AIEnhanced="True"' in packet


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".webp"])
def test_saved_marks_read_back(tmp_path: Path, suffix: str) -> None:
    rec = sample_record()
    path = tmp_path / f"out{suffix}"
    marking.save(path, np.asarray(rgb()), rec)
    assert marking.read_record(path) == rec
    with Image.open(path) as picture:
        info, data = dict(picture.info), picture.getexif()
    assert data[marking.EXIF_DESCRIPTION] == marking.NOTICE
    assert data[marking.EXIF_SOFTWARE] == f"facehall {__version__}"
    packet = info.get("xmp")
    assert packet is not None and marking.DIGITAL_SOURCE_TYPE.encode() in (
        packet if isinstance(packet, bytes) else packet.encode()
    )
    if suffix == ".png":
        assert info["AI-enhanced"] == "true" and info["Description"] == marking.NOTICE
        assert np.array_equal(np.asarray(Image.open(path)), np.asarray(rgb()))  # lossless


def test_unmarked_image_and_unknown_format(tmp_path: Path) -> None:
    rgb().save(tmp_path / "plain.png")
    assert marking.read_record(tmp_path / "plain.png") is None
    with pytest.raises(ValueError, match="unsupported"):
        marking.save(tmp_path / "out.gif", np.asarray(rgb()), sample_record())


def test_watermark_only_marks_the_corner() -> None:
    image = np.full((200, 300, 3), 128, dtype=np.uint8)
    marked = marking.watermark(image)
    changed = np.argwhere((marked != image).any(axis=2))
    assert len(changed) > 0
    assert changed[:, 0].min() > 150 and changed[:, 1].min() > 150  # bottom-right
    assert marked.shape == image.shape and (image == 128).all()  # the input is untouched


def test_plan_files_and_folders(tmp_path: Path) -> None:
    (tmp_path / "in" / "sub").mkdir(parents=True)
    for name in ("a.jpg", "sub/b.png", "notes.txt"):
        (tmp_path / "in" / name).write_bytes(b"x")
    single = tmp_path / "in" / "a.jpg"
    assert plan([single], tmp_path / "x.png") == [Job(single, tmp_path / "x.png")]
    assert plan([single], tmp_path / "out") == [Job(single, tmp_path / "out" / "a.png")]
    jobs = plan([tmp_path / "in"], tmp_path / "out", suffix=".jpg")
    assert [(j.source.name, j.target.relative_to(tmp_path / "out")) for j in jobs] == [
        ("a.jpg", Path("a.jpg")),
        ("b.png", Path("sub/b.jpg")),
    ]


def test_plan_refuses_unsafe_outputs(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    for name in ("one/a.jpg", "two/a.png"):
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(ValueError, match="same output"):
        plan([tmp_path / "one", tmp_path / "two"], tmp_path / "out")
    with pytest.raises(ValueError, match="replace an input"):
        plan([tmp_path / "two" / "a.png"], tmp_path / "two" / "a.png")
    (tmp_path / "out.png").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="--overwrite"):
        plan([tmp_path / "one" / "a.jpg"], tmp_path / "out.png")
    assert plan([tmp_path / "one" / "a.jpg"], tmp_path / "out.png", overwrite=True)
    with pytest.raises(FileNotFoundError):
        plan([tmp_path / "missing.jpg"], tmp_path / "out")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no input images"):
        plan([tmp_path / "empty"], tmp_path / "out")
    with pytest.raises(ValueError, match="unsupported"):
        plan([tmp_path / "one"], tmp_path / "out", suffix=".gif")


def test_summary_counts_the_faces_left_alone_by_kind() -> None:
    def face(restored: bool, reason: str | None = None) -> FaceResult:
        return FaceResult(np.zeros((5, 2)), "detected", 0.9, 1.0, restored, reason)

    faces = [
        face(True),
        face(False, "too large for the model: 2.8 times its input size (limit 2), left to …"),
        face(False, "too large for the model: 5.0 times its input size (limit 2), left to …"),
        face(False, "outside the image"),
    ]
    assert summary(faces) == "4 faces, 1 restored, 1 outside the image, 2 too large for the model"
    assert summary([face(True)]) == "1 face, 1 restored"


def test_numbers_and_parameters() -> None:
    assert numbers("1,2.5,3,4", 4, "--box") == [1.0, 2.5, 3.0, 4.0]
    for text in ("1,2,3", "1,2,x,4"):
        with pytest.raises(ValueError, match="--box needs 4"):
            numbers(text, 4, "--box")
    values = parameters(PipelineConfig())
    assert "models" not in values and "weights" not in values["detector"]
    assert values["detector"]["weights_file"] is None


def test_cli_enhances_an_image_with_given_landmarks(
    tmp_path: Path, checkpoint: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "photo.jpg"
    rgb((64, 64)).save(source, quality=90)
    out = tmp_path / "out" / "photo.png"
    args = ["enhance", str(source), "-o", str(out), "--ckpt", str(checkpoint), "--device", "cpu"]
    assert main([*args, "--landmarks", LANDMARKS, "--watermark"]) == 0
    printed = capsys.readouterr().out
    assert "1 face, 1 restored" in printed and marking.NOTICE in printed
    rec = marking.read_record(out)
    assert rec is not None
    assert rec["model"]["name"] == "tiny_x4-test" and rec["model"]["scale"] == 4
    assert rec["input"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert rec["input"]["faces_from"] == "landmarks" and rec["watermark"] is True
    assert (rec["output"]["width"], rec["output"]["height"]) == (256, 256)
    assert [f["restored"] for f in rec["faces"]] == [True]
    assert Image.open(out).size == (256, 256)
    assert not list(out.parent.glob("*.partial*"))

    # an existing output is kept; a scale that does not match the checkpoint is an error
    with pytest.raises(SystemExit) as stop:
        main([*args, "--landmarks", LANDMARKS])
    assert stop.value.code == 2 and "--overwrite" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["enhance", str(source), "-o", str(out), "--ckpt", str(checkpoint), "--scale", "8"])


def test_cli_rejects_landmarks_for_several_images_and_missing_models(
    tmp_path: Path, checkpoint: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("a.png", "b.png"):
        rgb((32, 32)).save(tmp_path / name)
    with pytest.raises(SystemExit):
        main(
            ["enhance", str(tmp_path / "a.png"), str(tmp_path / "b.png"), "-o", str(tmp_path / "o")]
            + ["--ckpt", str(checkpoint), "--device", "cpu", "--landmarks", LANDMARKS]
        )
    assert "one input image" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(
            ["enhance", str(tmp_path / "a.png"), "-o", str(tmp_path / "o2")]
            + ["--set", "models.x4=missing/best.safetensors", "--device", "cpu"]
        )
    assert "no model; pass --ckpt" in capsys.readouterr().err


def test_a_failing_image_does_not_stop_the_others(tmp_path: Path, checkpoint: Path) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    rgb((32, 32)).save(folder / "good.png")
    (folder / "broken.jpg").write_bytes(b"not an image")
    enhancer = Enhancer.from_checkpoint(checkpoint, PipelineConfig())
    enhancer._detector = NoFaces()
    lines: list[str] = []
    failures = enhance_files(enhancer, plan([folder], tmp_path / "out"), report=lines.append)
    assert [path.name for path, _ in failures] == ["broken.jpg"]
    assert (tmp_path / "out" / "good.png").is_file()
    assert not (tmp_path / "out" / "broken.png").exists()
    assert any("good.png" in line and "0 faces, 0 restored" in line for line in lines)


def test_cli_without_detector_weights_reports_each_image(
    tmp_path: Path, checkpoint: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rgb((32, 32)).save(tmp_path / "a.png")
    code = main(
        ["enhance", str(tmp_path / "a.png"), "-o", str(tmp_path / "out.png")]
        + ["--ckpt", str(checkpoint), "--device", "cpu", "--set", "detector.weights=none.onnx"]
    )
    printed = capsys.readouterr().out
    assert code == 1 and "download_weights.py yunet" in printed and "1 of 1" in printed
