"""Tests for facehall.eval.match: search specs and a search on synthetic data."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from facehall.config import load_config
from facehall.data.config import BlurConfig, DataConfig, DegradeConfig, StageConfig
from facehall.data.dataset import AlignedFaces
from facehall.data.realworld import RealFaces
from facehall.eval.match import expand, load_spec, references, render, run_search, winner_yaml

HR = 32
ROWS = [("val", False, (i % 4 - 2, i % 3 - 1)) for i in range(12)] + [
    ("test", False, (i % 3 - 1, i % 4 - 2)) for i in range(12)
]
BASE = """\
kind: pipeline
first:
  blur:
    prob: 0.0
    gaussian: 1.0
    sigma: [1.5, 1.5]
  jpeg:
    prob: 0.0
    quality: [95, 95]
"""


def write_spec(root: Path, axes: str, extra: str = "") -> Path:
    (root / "base.yaml").write_text(BASE)
    (root / "spec.yaml").write_text(f"base: base.yaml\n{extra}axes:\n{axes}")
    return root / "spec.yaml"


def test_expand_applies_overrides(tmp_path: Path) -> None:
    axes = (
        "  blur:\n"
        "    none: {}\n"
        "    strong: {first.blur.prob: 1.0}\n"
        "  jpeg:\n"
        "    q50: {first.jpeg.prob: 1.0, first.jpeg.quality: [50, 95]}\n"
        "    keep: {}\n"
    )
    spec = load_spec(write_spec(tmp_path, axes, "references: [base.yaml]\n"))
    candidates = expand(spec)
    assert [c.name for c in candidates] == [
        "blur=none,jpeg=q50",
        "blur=none,jpeg=keep",
        "blur=strong,jpeg=q50",
        "blur=strong,jpeg=keep",
    ]
    assert candidates[0].config.first.jpeg.quality == [50, 95]
    assert candidates[0].config.first.blur.prob == 0.0
    assert candidates[3].overrides == {"first.blur.prob": 1.0}
    assert candidates[3].config.first.jpeg.prob == 0.0
    assert [r.name for r in references(spec)] == ["base"]


def test_invalid_specs(tmp_path: Path) -> None:
    clash = "  a:\n    x: {first.blur.prob: 1.0}\n  b:\n    y: {first.blur.prob: 0.5}\n"
    with pytest.raises(ValueError, match="same keys"):
        expand(load_spec(write_spec(tmp_path, clash)))
    (tmp_path / "bad.yaml").write_text("base: base.yaml\nextra: 1\n")
    with pytest.raises(ValueError, match="expected keys"):
        load_spec(tmp_path / "bad.yaml")
    (tmp_path / "empty_axis.yaml").write_text("base: base.yaml\naxes:\n  blur: {}\n")
    with pytest.raises(ValueError, match="must map option names"):
        load_spec(tmp_path / "empty_axis.yaml")


def test_search_prefers_the_matching_degradation(
    tmp_path: Path,
    make_dataset: Callable[..., Path],
    make_realworld: Callable[..., Path],
    image_stats: Callable[[], Any],
) -> None:
    celeba = make_dataset(tmp_path / "celeba", ROWS)
    data = DataConfig(root=str(celeba), hr_size=HR)
    # The "real" crops: other faces (the test split), blurred like the strong candidate.
    blurred = DegradeConfig(
        kind="pipeline",
        first=StageConfig(blur=BlurConfig(prob=1.0, gaussian=1.0, sigma=[1.5, 1.5])),
    )
    source = AlignedFaces(data, blurred, "test", 4, augment=False)
    crops = [
        {
            "scale": 4,
            "split": "val",
            "pixels": (source[i]["lr"] * 255).round().byte().permute(1, 2, 0).numpy(),
        }
        for i in range(len(source))
    ]
    real = RealFaces(make_realworld(tmp_path / "real", crops), "val", 4)
    axes = "  blur:\n    none: {}\n    strong: {first.blur.prob: 1.0}\n"
    spec = load_spec(write_spec(tmp_path, axes, "references: [base.yaml]\n"))

    results = run_search(
        spec,
        real,
        data,
        data_split="val",
        extractor=image_stats(),
        samples=12,
        confirm=2,
        kid_subsets=5,
        kid_subset_size=10,
        batch_size=4,
        workers=0,
    )
    assert results["winner"] == "blur=strong"
    rows = results["rows"]
    assert [r["name"] for r in rows] == ["base", "blur=strong", "blur=none"]
    assert rows[0]["reference"] and rows[1]["second"] is not None
    assert results["synthetic"]["samples"] == 12 and results["real"]["samples"] == 12
    assert "Winner: `blur=strong`" in render(results)

    path = tmp_path / "winner.yaml"
    path.write_text(winner_yaml(results, "header line"))
    assert path.read_text().startswith("# header line\n")
    assert load_config(path, DegradeConfig).first.blur.prob == 1.0


def test_script_parses(load_script: Callable[[str], Any]) -> None:
    script = load_script("match_degradations")
    args = script.parse_args(["--spec", "s.yaml", "--real", "r.yaml"])
    assert (args.real_split, args.data_split, args.samples, args.confirm) == ("val", "val", 5000, 5)
