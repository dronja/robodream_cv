"""Quality gate for a new model (``CLAUDE.md`` §10, D23).

The gate compares a candidate with the current reference model of the same scale on the frozen
synthetic protocols in ``configs/gate.yaml``, from their ``metrics.json`` files
(``<eval root>/<protocol>_x<scale>/<split>/<label>/``). On every protocol the candidate must:

- have a lower LPIPS and a higher ID similarity than the reference (all images);
- lose at most ``psnr_drop`` dB of PSNR-Y;
- lose at most ``subgroup_id_drop`` of ID similarity in every subgroup of ``subgroups``.

Both models must have been scored on the same inputs (``inputs_sha256``). The real-world set is
not gated: its FID and KID reward invented faces (``docs/DECISIONS.md``, PR 2.5).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING


@dataclass
class GateConfig:
    protocols: list[str] = MISSING  # protocol names without the "_x<scale>" suffix
    split: str = "test"
    psnr_drop: float = MISSING  # dB
    subgroup_id_drop: float = MISSING
    subgroups: list[str] = field(default_factory=list)  # metrics.json subset names
    references: dict[int, str] = field(default_factory=dict)  # scale -> reference label


@dataclass(frozen=True)
class Check:
    protocol: str
    name: str  # e.g. "lpips", "identity: male"
    candidate: float
    reference: float
    limit: str  # the rule, e.g. "< reference"
    passed: bool


def load_metrics(root: Path, protocol: str, split: str, label: str) -> dict[str, Any]:
    path = root / protocol / split / label / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: evaluate {label} on {protocol} first")
    metrics: dict[str, Any] = json.loads(path.read_text())
    return metrics


def mean(metrics: dict[str, Any], subset: str, metric: str) -> float:
    return float(metrics["subsets"][subset][metric]["mean"])


def check(
    config: GateConfig, root: Path, scale: int, candidate: str, reference: str | None = None
) -> list[Check]:
    """All gate checks of ``candidate`` against ``reference`` (default: the configured one)."""
    reference = reference or config.references.get(scale)
    if not reference:
        raise ValueError(f"no reference model for x{scale} in the gate config")
    checks = []
    for base in config.protocols:
        protocol = f"{base}_x{scale}"
        new = load_metrics(root, protocol, config.split, candidate)
        old = load_metrics(root, protocol, config.split, reference)
        if new.get("inputs_sha256") != old.get("inputs_sha256"):
            raise ValueError(f"{protocol}: {candidate} and {reference} were scored on other inputs")
        checks += protocol_checks(config, protocol, new, old)
    return checks


def protocol_checks(
    config: GateConfig, protocol: str, new: dict[str, Any], old: dict[str, Any]
) -> list[Check]:
    def pair(subset: str, metric: str) -> tuple[float, float]:
        return mean(new, subset, metric), mean(old, subset, metric)

    lpips, identity, psnr = pair("all", "lpips"), pair("all", "identity"), pair("all", "psnr_y")
    checks = [
        Check(protocol, "lpips", *lpips, "< reference", lpips[0] < lpips[1]),
        Check(protocol, "identity", *identity, "> reference", identity[0] > identity[1]),
        Check(
            protocol,
            "psnr_y",
            *psnr,
            f">= reference - {config.psnr_drop}",
            psnr[0] >= psnr[1] - config.psnr_drop,
        ),
    ]
    for group in config.subgroups:
        ids = pair(group, "identity")
        limit = f">= reference - {config.subgroup_id_drop}"
        passed = ids[0] >= ids[1] - config.subgroup_id_drop
        checks.append(Check(protocol, f"identity: {group}", *ids, limit, passed))
    return checks


def render(checks: list[Check], candidate: str, reference: str) -> str:
    passed = all(c.passed for c in checks)
    lines = [
        f"Quality gate: `{candidate}` against `{reference}`: **{'pass' if passed else 'fail'}**",
        "",
        "| Protocol | Check | Candidate | Reference | Rule | Result |",
        "|---|---|---:|---:|---|---|",
    ]
    for c in checks:
        result = "pass" if c.passed else "**fail**"
        lines.append(
            f"| {c.protocol} | {c.name} | {c.candidate:.4f} | {c.reference:.4f} | {c.limit} "
            f"| {result} |"
        )
    return "\n".join(lines) + "\n"
