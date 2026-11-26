"""Quality gate for a new model (``CLAUDE.md`` §10, D23).

The gate compares a candidate with the current reference model of the same scale on the frozen
synthetic protocols in ``configs/gate.yaml``, from their ``metrics.json`` files
(``<eval root>/<protocol>_x<scale>/<split>/<label>/``). On every protocol the candidate must:

- have a lower LPIPS and a higher ID similarity than the reference (all images);
- lose at most ``psnr_drop`` dB of PSNR-Y;
- lose at most ``subgroup_id_drop`` of ID similarity in every subgroup of ``subgroups``;
- have a spectral peak ratio of at most ``spectral_peak`` (D28): the largest SR / HR power ratio
  at high frequencies (``facehall.eval.spectrum``), a check for periodic pixel patterns. The
  limit is absolute: models without a pattern stay below 1, whatever the reference.

Artifact check (D26): on the real-world protocol ``realworld`` (no reference images), the
candidate's KID against real HR faces may exceed the reference's by at most ``kid_stds``
standard deviations of the reference's KID. The check is one-sided: a lower KID is not a gain,
because KID also rewards invented faces (``docs/DECISIONS.md``, PR 2.5). It catches outputs
that move away from real photos, such as the stripe pattern of PR 3.3, which the synthetic
metrics barely see. It missed the finer pattern of PR 3.6, which the spectral check catches.

Both models must have been scored on the same inputs (``inputs_sha256``).
"""

import json
import math
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
    realworld: str = ""  # real-world protocol without "_x<scale>"; empty: no artifact check
    kid_stds: float = 2.0  # KID margin in standard deviations of the reference's KID
    spectral_peak: float = 0.0  # largest allowed spectral peak ratio (D28); 0: no check


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
    if config.kid_stds < 0 or config.spectral_peak < 0:
        raise ValueError("kid_stds and spectral_peak must not be negative")

    def pair(base: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        protocol = f"{base}_x{scale}"
        new = load_metrics(root, protocol, config.split, candidate)
        old = load_metrics(root, protocol, config.split, reference)
        if new.get("inputs_sha256") != old.get("inputs_sha256"):
            raise ValueError(f"{protocol}: {candidate} and {reference} were scored on other inputs")
        return protocol, new, old

    checks = []
    for base in config.protocols:
        checks += protocol_checks(config, *pair(base))
    if config.realworld:
        checks.append(artifact_check(config, *pair(config.realworld)))
    return checks


def artifact_check(
    config: GateConfig, protocol: str, new: dict[str, Any], old: dict[str, Any]
) -> Check:
    """KID ×10³ (all faces) at most the reference's plus ``kid_stds`` of its standard deviations."""
    kid, ref = new["subsets"]["all"]["kid"], old["subsets"]["all"]["kid"]
    limit = 1e3 * (ref["mean"] + config.kid_stds * ref["std"])
    return Check(
        protocol,
        "kid ×10³",
        1e3 * kid["mean"],
        1e3 * ref["mean"],
        f"<= reference + {config.kid_stds:g} std ({limit:.2f})",
        1e3 * kid["mean"] <= limit,
    )


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
    if config.spectral_peak > 0:
        if "spectrum" not in new:
            raise ValueError(
                f"{protocol}: the candidate's metrics.json has no spectrum; evaluate it again"
            )
        peak = float(new["spectrum"]["peak_ratio"])
        reference = float(old["spectrum"]["peak_ratio"]) if "spectrum" in old else math.nan
        limit = f"<= {config.spectral_peak:g}"
        passed = peak <= config.spectral_peak
        checks.append(Check(protocol, "spectral peak", peak, reference, limit, passed))
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
        reference = "—" if math.isnan(c.reference) else f"{c.reference:.4f}"
        lines.append(
            f"| {c.protocol} | {c.name} | {c.candidate:.4f} | {reference} | {c.limit} | {result} |"
        )
    return "\n".join(lines) + "\n"
