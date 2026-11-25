"""Tests for the quality gate (facehall.eval.gate, scripts/quality_gate.py)."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from facehall.config import load_config
from facehall.eval.gate import GateConfig, check, render

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
GROUPS = ["male", "not_male"]


def config() -> GateConfig:
    return GateConfig(
        protocols=["light", "medium"],
        psnr_drop=0.3,
        subgroup_id_drop=0.01,
        subgroups=GROUPS,
        references={4: "ref"},
    )


def write(
    root: Path, label: str, *, psnr: float, lpips: float, ids: float, **groups: float
) -> None:
    for protocol in ("light_x4", "medium_x4"):
        subsets: dict[str, Any] = {}
        for subset in ["all", *GROUPS]:
            value = groups.get(subset, ids)
            subsets[subset] = {
                "psnr_y": {"mean": psnr},
                "lpips": {"mean": lpips},
                "identity": {"mean": value},
            }
        path = root / protocol / "test" / label / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"inputs_sha256": "abc", "subsets": subsets}))


def failed(checks: list[Any]) -> set[str]:
    return {c.name for c in checks if not c.passed}


def test_gate(tmp_path: Path) -> None:
    write(tmp_path, "ref", psnr=25.0, lpips=0.30, ids=0.40)
    write(tmp_path, "good", psnr=24.8, lpips=0.25, ids=0.45, male=0.395)
    checks = check(config(), tmp_path, 4, "good")
    assert len(checks) == 2 * (3 + len(GROUPS)) and not failed(checks)
    assert "**pass**" in render(checks, "good", "ref")

    write(tmp_path, "worse", psnr=24.6, lpips=0.31, ids=0.45, not_male=0.385)
    assert failed(check(config(), tmp_path, 4, "worse")) == {
        "lpips",
        "psnr_y",
        "identity: not_male",
    }
    write(tmp_path, "same_id", psnr=25.0, lpips=0.25, ids=0.40)
    assert failed(check(config(), tmp_path, 4, "same_id")) == {"identity"}


def test_gate_errors(tmp_path: Path) -> None:
    write(tmp_path, "ref", psnr=25.0, lpips=0.30, ids=0.40)
    with pytest.raises(FileNotFoundError, match="evaluate missing"):
        check(config(), tmp_path, 4, "missing")
    with pytest.raises(ValueError, match="no reference"):
        check(config(), tmp_path, 8, "ref")
    write(tmp_path, "other", psnr=25.0, lpips=0.30, ids=0.40)
    path = tmp_path / "light_x4" / "test" / "other" / "metrics.json"
    metrics = json.loads(path.read_text())
    path.write_text(json.dumps({**metrics, "inputs_sha256": "different"}))
    with pytest.raises(ValueError, match="other inputs"):
        check(config(), tmp_path, 4, "other")


def test_repository_gate_config() -> None:
    gate = load_config(CONFIGS / "gate.yaml", GateConfig)
    assert gate.protocols == ["celeba128_light", "celeba128_medium"]
    assert (gate.psnr_drop, gate.subgroup_id_drop) == (0.3, 0.01)
    assert set(gate.references) == {4, 8}
    for protocol in gate.protocols:
        for scale in gate.references:
            assert (CONFIGS / "eval" / f"{protocol}_x{scale}.yaml").is_file()


def test_script_exit_code(tmp_path: Path, load_script: Callable[[str], Any]) -> None:
    write(tmp_path, "ref", psnr=25.0, lpips=0.30, ids=0.40)
    write(tmp_path, "good", psnr=25.0, lpips=0.20, ids=0.50)
    gate = tmp_path / "gate.yaml"
    gate.write_text(
        "protocols: [light, medium]\npsnr_drop: 0.3\nsubgroup_id_drop: 0.01\n"
        "subgroups: [male]\nreferences:\n  4: ref\n"
    )
    script = load_script("quality_gate")
    common = ["--scale", "4", "--config", str(gate), "--eval-root", str(tmp_path)]
    assert script.main([*common, "--candidate", "good"]) == 0
    assert script.main([*common, "--candidate", "ref", "--reference", "good"]) == 1
