"""Tests for the quality gate (facehall.eval.gate, scripts/quality_gate.py)."""

import json
import math
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
    root: Path,
    label: str,
    *,
    psnr: float,
    lpips: float,
    ids: float,
    peak: float | None = None,
    **groups: float,
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
        metrics: dict[str, Any] = {"inputs_sha256": "abc", "subsets": subsets}
        if peak is not None:
            metrics["spectrum"] = {"peak_ratio": peak}
        path.write_text(json.dumps(metrics))


def write_real(root: Path, label: str, *, kid: float, std: float, digest: str = "real") -> None:
    path = root / "real_x4" / "test" / label / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    subsets = {"all": {"kid": {"mean": kid, "std": std}, "fid": 100.0}}
    path.write_text(json.dumps({"inputs_sha256": digest, "subsets": subsets}))


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


def test_gate_artifact_check(tmp_path: Path) -> None:
    gate = config()
    gate.realworld = "real"
    write(tmp_path, "ref", psnr=25.0, lpips=0.30, ids=0.40)
    for label in ("close", "lower", "far"):
        write(tmp_path, label, psnr=25.0, lpips=0.25, ids=0.45)
    write_real(tmp_path, "ref", kid=0.095, std=0.002)
    write_real(tmp_path, "close", kid=0.0989, std=0.010)  # limit: 0.095 + 2 * 0.002
    write_real(tmp_path, "lower", kid=0.050, std=0.002)
    write_real(tmp_path, "far", kid=0.0991, std=0.002)

    checks = check(gate, tmp_path, 4, "close")
    assert len(checks) == 2 * (3 + len(GROUPS)) + 1 and not failed(checks)
    kid = checks[-1]
    assert (kid.protocol, kid.name) == ("real_x4", "kid ×10³")
    assert kid.candidate == pytest.approx(98.9) and kid.reference == pytest.approx(95.0)
    assert kid.limit == "<= reference + 2 std (99.00)"
    assert not failed(check(gate, tmp_path, 4, "lower"))  # one-sided: lower KID is no gain
    assert failed(check(gate, tmp_path, 4, "far")) == {"kid ×10³"}
    gate.kid_stds = 0.0
    assert failed(check(gate, tmp_path, 4, "close")) == {"kid ×10³"}

    gate.kid_stds = -1.0
    with pytest.raises(ValueError, match="must not be negative"):
        check(gate, tmp_path, 4, "close")
    gate.kid_stds = 2.0
    write_real(tmp_path, "far", kid=0.090, std=0.002, digest="other")
    with pytest.raises(ValueError, match="other inputs"):
        check(gate, tmp_path, 4, "far")
    write(tmp_path, "unscored", psnr=25.0, lpips=0.25, ids=0.45)
    with pytest.raises(FileNotFoundError, match="evaluate unscored on real_x4"):
        check(gate, tmp_path, 4, "unscored")


def test_gate_spectral_check(tmp_path: Path) -> None:
    gate = config()
    gate.spectral_peak = 1.0
    write(tmp_path, "ref", psnr=25.0, lpips=0.30, ids=0.40, peak=0.2)
    write(tmp_path, "clean", psnr=25.0, lpips=0.25, ids=0.45, peak=0.9)
    write(tmp_path, "pattern", psnr=25.0, lpips=0.25, ids=0.45, peak=12.0)
    write(tmp_path, "old", psnr=25.0, lpips=0.25, ids=0.45)  # evaluated before the spectrum

    checks = check(gate, tmp_path, 4, "clean")
    assert len(checks) == 2 * (3 + len(GROUPS) + 1) and not failed(checks)
    peak = checks[3 + len(GROUPS)]
    assert (peak.name, peak.candidate, peak.reference, peak.limit) == (
        "spectral peak",
        0.9,
        0.2,
        "<= 1",
    )
    assert failed(check(gate, tmp_path, 4, "pattern")) == {"spectral peak"}
    with pytest.raises(ValueError, match="has no spectrum"):
        check(gate, tmp_path, 4, "old")
    without = check(gate, tmp_path, 4, "clean", "old")  # a reference without the spectrum
    assert math.isnan([c for c in without if c.name == "spectral peak"][0].reference)
    assert "| spectral peak | 0.9000 | — | <= 1 | pass |" in render(without, "clean", "old")
    gate.spectral_peak = -1.0
    with pytest.raises(ValueError, match="must not be negative"):
        check(gate, tmp_path, 4, "clean")


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
    assert (gate.realworld, gate.kid_stds, gate.spectral_peak) == ("wider_real", 2.0, 1.0)
    for protocol in [*gate.protocols, gate.realworld]:
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
