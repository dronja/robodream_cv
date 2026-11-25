"""Tests for facehall.eval.match: search specs and a search on synthetic data."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facehall.config import load_config
from facehall.data.config import BlurConfig, DataConfig, DegradeConfig, StageConfig
from facehall.data.dataset import AlignedFaces
from facehall.data.realworld import RealFaces
from facehall.eval.match import (
    GATE,
    crop_box,
    dotlist,
    expand,
    gate,
    load_spec,
    references,
    render,
    run_search,
    typical_mask,
    winner_yaml,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
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
    (root / "ref.yaml").write_text(BASE.replace("prob: 0.0\n    quality", "prob: 1.0\n    quality"))
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
    spec = load_spec(write_spec(tmp_path, axes, "references: [ref.yaml]\n"))
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
    refs = references(spec)
    assert [r.name for r in refs] == ["base", "ref"]
    assert refs[0].config.first.jpeg.prob == 0.0 and refs[1].config.first.jpeg.prob == 1.0
    assert spec.typical == {} and spec.centre == (0.0, 0.0, 1.0)


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
    (tmp_path / "bad_centre.yaml").write_text("base: base.yaml\ncentre: [0.1, 0.2]\n")
    with pytest.raises(ValueError, match="centre"):
        load_spec(tmp_path / "bad_centre.yaml")


def test_crop_box_and_gate() -> None:
    assert crop_box(32, (0.1875, 0.28, 0.625)) == (slice(9, 29), slice(6, 26))
    assert crop_box(16, (0.1875, 0.28, 0.625)) == (slice(4, 14), slice(3, 13))
    assert crop_box(8, (0.0, 0.0, 1.0)) == (slice(0, 8), slice(0, 8))
    with pytest.raises(ValueError, match="does not fit"):
        crop_box(16, (0.5, 0.5, 0.6))
    base = dict.fromkeys(GATE, 1.0)
    assert all(gate(dict.fromkeys(GATE, 0.5), base).values())
    checks = gate({**dict.fromkeys(GATE, 0.5), "fid_all": 1.0}, base)
    assert [m for m, ok in checks.items() if not ok] == ["fid_all"]


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
            "attr_occlusion": int(i % 4 == 0),
        }
        for i in range(len(source))
    ]
    real = RealFaces(make_realworld(tmp_path / "real", crops), "val", 4)
    assert typical_mask(real, {"attr_occlusion": 0}).sum() == 9
    with pytest.raises(ValueError, match="no column"):
        typical_mask(real, {"attr_missing": 0})
    axes = "  blur:\n    none: {}\n    strong: {first.blur.prob: 1.0}\n"
    extra = "references: [ref.yaml]\ntypical: {attr_occlusion: 0}\ncentre: [0.25, 0.25, 0.5]\n"
    spec = load_spec(write_spec(tmp_path, axes, extra))

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
    assert [r["name"] for r in rows] == ["base", "ref", "blur=strong", "blur=none"]
    assert rows[0]["reference"] and rows[2]["second"] is not None and rows[3]["second"] is not None
    assert set(rows[2]["second"]) == {*GATE, "score"} and "fid_all" not in rows[2]["first"]
    assert results["synthetic"]["samples"] == 12 and results["real"]["typical_samples"] == 9
    assert results["centre"] == {"box": [0.25, 0.25, 0.5], "rows": [2, 6], "columns": [2, 6]}
    assert results["accepted"] is all(results["gate"].values())
    text = render(results)
    assert "Winner: `blur=strong`" in text and "## Round 2 (seed 1)" in text

    path = tmp_path / "winner.yaml"
    path.write_text(winner_yaml(results, "header line"))
    assert path.read_text().startswith("# header line\n")
    assert load_config(path, DegradeConfig).first.blur.prob == 1.0


def test_search_rejects_duplicate_names(
    tmp_path: Path, make_dataset: Callable[..., Path], make_realworld: Callable[..., Path]
) -> None:
    celeba = make_dataset(tmp_path / "celeba", ROWS)
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    real = RealFaces(
        make_realworld(tmp_path / "real", [{"scale": 4, "split": "val", "pixels": pixels}] * 3),
        "val",
        4,
    )
    spec = load_spec(write_spec(tmp_path, "  a:\n    x: {}\n", "references: [base.yaml]\n"))
    with pytest.raises(ValueError, match="not unique"):
        run_search(
            spec,
            real,
            DataConfig(root=str(celeba), hr_size=HR),
            data_split="val",
            extractor=lambda images: images.float().mean(dim=(2, 3)),
            workers=0,
        )


def test_script_parses(load_script: Callable[[str], Any]) -> None:
    script = load_script("match_degradations")
    args = script.parse_args(["--spec", "s.yaml", "--real", "r.yaml"])
    assert (args.real_split, args.data_split, args.samples, args.confirm) == ("val", "val", 5000, 5)


def test_tuned_x8_config_is_the_accepted_winner() -> None:
    """configs/degrade/realistic_x8.yaml = the base with the PR 2.6 winner's options."""
    spec = load_spec(CONFIGS / "degrade/search/realistic.yaml")
    winner = {"blur": "x0.6", "noise": "s35", "jpeg": "q70", "second": "p0.5"}
    overrides: dict[str, Any] = {}
    for axis, option in winner.items():
        overrides |= spec.axes[axis][option]
    expected = load_config(spec.base, DegradeConfig, dotlist(overrides))
    assert load_config(CONFIGS / "degrade/realistic_x8.yaml", DegradeConfig) == expected
    assert len(expand(spec)) == 135
