"""Tune a degradation by matching synthetic LR crops to real ones (docs/PROJECT.md §9, PR 2.6).

The real LR crops come from a real-world set (``kind: realworld`` protocol, usually its ``val``
split, D20). The synthetic LR crops come from a fixed random subset of CelebA HR crops, degraded
by each candidate configuration. A candidate is better when its synthetic crops are closer to
the real ones in InceptionV3 feature space: lower KID and FID. No model is involved.

A search spec (``configs/degrade/search/*.yaml``) names a base degradation config, reference
configs scored as they are, and axes. Every axis maps option names to overrides (dotted keys of
``DegradeConfig``); the candidates are the Cartesian product of the options, applied to the
base. An empty option keeps the base value. The base itself is always scored, as ``base``.

**Views.** Real and synthetic faces differ in content as well as in degradation: the real crops
show occlusions, unusual poses and cluttered backgrounds, CelebA shows posed portraits. A
degradation can lower the distance by imitating content (noise standing in for clutter), so
every candidate is scored on three views:

- ``all``: every real face, the whole crop;
- ``typical``: only the real faces whose manifest columns have the values in the spec's
  ``typical`` (for WIDER FACE: no occlusion, typical pose);
- ``centre``: a square inside the crop (the spec's ``centre``: x, y and side as fractions of the
  crop), which holds mostly the face itself on the alignment template.

**Selection.** Candidates are ranked by the mean of the three KIDs on one synthetic sample
(``seed``). The best ``confirm`` candidates, the base and the references are scored again on a
fresh synthetic sample with fresh KID subsets (``seed + 1``), now with FID on the ``all`` and
``centre`` views. The winner is the best finalist on that second sample, which limits the
winner's curse of picking the luckiest of many noisy estimates. The winner is *accepted* only if
it is better than the base on all five measures of the second sample (``GATE``); otherwise the
base stays.
"""

import itertools
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from facehall.config import load_config
from facehall.data.config import DataConfig, DegradeConfig
from facehall.data.dataset import AlignedFaces
from facehall.data.degrade import validate_degrade
from facehall.data.realworld import RealFaces
from facehall.eval.noref import FeatureExtractor, Features, fid, kid, to_uint8
from facehall.provenance import git_commit, package_versions
from facehall.train.trainer import VERSIONED, make_loader, manifest_info

log = logging.getLogger(__name__)

BASE = "base"  # name of the base config among the references
VIEWS = ("all", "typical", "centre")
GATE = ("kid_all", "kid_typical", "kid_centre", "fid_all", "fid_centre")
SPEC_KEYS = ("base", "references", "typical", "centre", "axes")

Box = tuple[float, float, float]  # x, y, side as fractions of the crop


@dataclass(frozen=True)
class Candidate:
    name: str
    overrides: dict[str, Any]  # dotted DegradeConfig key -> value
    config: DegradeConfig


@dataclass(frozen=True)
class SearchSpec:
    name: str
    base: Path
    references: list[Path]
    axes: dict[str, dict[str, dict[str, Any]]]  # axis -> option -> overrides
    typical: dict[str, int] = field(default_factory=dict)  # real-set column -> value
    centre: Box = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ViewFeatures:
    full: Features  # (N, D) features of the whole crops
    centre: Features  # (N, D) features of the centre squares


def load_spec(path: Path | str) -> SearchSpec:
    """Read a search spec; file paths in it are relative to the spec file."""
    path = Path(path)
    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise TypeError(f"{path} must hold a mapping")
    data = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(data, dict)
    unknown = set(data) - set(SPEC_KEYS)
    if unknown or "base" not in data:
        raise ValueError(f"{path}: expected keys {SPEC_KEYS} with base; got {sorted(data)}")
    axes = data.get("axes") or {}
    for axis, options in axes.items():
        if not isinstance(options, dict) or not options:
            raise ValueError(f"{path}: axis {axis!r} must map option names to overrides")
        for option, overrides in options.items():
            if not isinstance(overrides, dict):
                raise ValueError(f"{path}: option {axis}={option} must be a mapping")
    centre = data.get("centre") or [0.0, 0.0, 1.0]
    if not isinstance(centre, list) or len(centre) != 3:
        raise ValueError(f"{path}: centre must be [x, y, side]")
    typical = data.get("typical") or {}
    return SearchSpec(
        name=path.stem,
        base=path.parent / str(data["base"]),
        references=[path.parent / str(p) for p in data.get("references") or []],
        axes={str(a): {str(o): dict(v) for o, v in opts.items()} for a, opts in axes.items()},
        typical={str(column): int(value) for column, value in typical.items()},
        centre=(float(centre[0]), float(centre[1]), float(centre[2])),
    )


def expand(spec: SearchSpec) -> list[Candidate]:
    """All option combinations, applied to the base config, in axis and option order."""
    axes = list(spec.axes.items())
    choices = [list(options.items()) for _, options in axes]
    candidates = []
    for combo in itertools.product(*choices):
        name = ",".join(
            f"{axis}={option}" for (axis, _), (option, _) in zip(axes, combo, strict=True)
        )
        overrides: dict[str, Any] = {}
        for _, values in combo:
            clash = set(overrides) & set(values)
            if clash:
                raise ValueError(f"axes override the same keys: {sorted(clash)}")
            overrides |= values
        config = load_config(spec.base, DegradeConfig, dotlist(overrides))
        validate_degrade(config)
        candidates.append(Candidate(name=name or "unchanged", overrides=overrides, config=config))
    return candidates


def references(spec: SearchSpec) -> list[Candidate]:
    """The base (as ``base``) and the reference configs, scored as they are."""
    result = []
    for name, path in [(BASE, spec.base), *((p.stem, p) for p in spec.references)]:
        config = load_config(path, DegradeConfig)
        validate_degrade(config)
        result.append(Candidate(name=name, overrides={}, config=config))
    return result


def dotlist(overrides: dict[str, Any]) -> list[str]:
    return [f"{key}={json.dumps(value)}" for key, value in overrides.items()]


def crop_box(size: int, centre: Box) -> tuple[slice, slice]:
    """Rows and columns of the ``centre`` square in a crop of ``size`` px."""
    x, y, side = centre
    x0, y0, width = round(x * size), round(y * size), round(side * size)
    if width < 2 or min(x0, y0) < 0 or max(x0, y0) + width > size:
        raise ValueError(f"centre {centre} does not fit a {size} px crop")
    return slice(y0, y0 + width), slice(x0, x0 + width)


def view_features(
    batches: Iterable[dict[str, Any]], extractor: FeatureExtractor, box: tuple[slice, slice]
) -> ViewFeatures:
    """Features of the whole ``lr`` crops and of their centre squares."""
    full, centre = [], []
    rows, columns = box
    for batch in batches:
        images = to_uint8(batch["lr"])
        full.append(extractor(images).cpu().numpy())
        centre.append(extractor(images[:, :, rows, columns].contiguous()).cpu().numpy())
    return ViewFeatures(
        np.concatenate(full).astype(np.float32), np.concatenate(centre).astype(np.float32)
    )


def typical_mask(real: RealFaces, typical: dict[str, int]) -> npt.NDArray[np.bool_]:
    """Real faces whose manifest columns have the given values."""
    mask = np.ones(len(real), dtype=bool)
    for column, value in typical.items():
        if not real.has_column(column):
            raise ValueError(f"the real set has no column {column!r}")
        mask &= real.column(column) == value
    if mask.sum() < 2:
        raise ValueError(f"fewer than two real faces have {typical}")
    return mask


def distances(
    real: ViewFeatures,
    typical: npt.NDArray[np.bool_],
    synthetic: ViewFeatures,
    *,
    kid_subsets: int,
    kid_subset_size: int,
    seed: int,
    device: torch.device,
    with_fid: bool,
) -> dict[str, float]:
    """KID on the three views and their mean (``score``); FID on two views if ``with_fid``."""

    def kid_mean(x: Features, y: Features) -> float:
        return kid(
            x, y, subsets=kid_subsets, subset_size=kid_subset_size, seed=seed, device=device
        )[0]

    result = {
        "kid_all": kid_mean(real.full, synthetic.full),
        "kid_typical": kid_mean(real.full[typical], synthetic.full),
        "kid_centre": kid_mean(real.centre, synthetic.centre),
    }
    result["score"] = sum(result[f"kid_{view}"] for view in VIEWS) / len(VIEWS)
    if with_fid:
        result["fid_all"] = fid(real.full, synthetic.full)
        result["fid_centre"] = fid(real.centre, synthetic.centre)
    return result


def gate(winner: dict[str, float], base: dict[str, float]) -> dict[str, bool]:
    """Per measure of ``GATE``: is the winner lower (better) than the base?"""
    return {measure: winner[measure] < base[measure] for measure in GATE}


def sample_indices(size: int, count: int, seed: int) -> list[int]:
    """A sorted random subset of ``count`` indices (all of them if ``size`` is smaller)."""
    count = min(count, size)
    return sorted(np.random.default_rng(seed).choice(size, count, replace=False).tolist())


def run_search(
    spec: SearchSpec,
    real: RealFaces,
    data: DataConfig,
    *,
    data_split: str,
    extractor: FeatureExtractor,
    samples: int = 5000,
    seed: int = 0,
    confirm: int = 5,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    batch_size: int = 256,
    workers: int = 8,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Score the base, the references and every candidate (see the module docstring)."""
    device = device if device is not None else torch.device("cpu")
    scale = real.scale
    if data.hr_size // scale != real.lr_size:
        raise ValueError(f"HR {data.hr_size} / x{scale} does not give the real LR size")
    start = time.perf_counter()
    box = crop_box(real.lr_size, spec.centre)
    refs = references(spec)
    candidates = expand(spec)
    names = [c.name for c in [*refs, *candidates]]
    if len(set(names)) != len(names):
        raise ValueError(f"configuration names are not unique: {names}")
    mask = typical_mask(real, spec.typical)
    real_feats = view_features(make_loader(real, batch_size, workers, device), extractor, box)
    size = len(AlignedFaces(data, DegradeConfig(), data_split, scale, augment=False))
    log.info(
        "%d candidates, %d references, %d real faces (%d typical)",
        len(candidates),
        len(refs),
        len(real),
        mask.sum(),
    )

    def score(candidate: Candidate, round_seed: int, with_fid: bool) -> dict[str, float]:
        dataset = AlignedFaces(data, candidate.config, data_split, scale, seed=round_seed)
        subset = Subset(dataset, sample_indices(size, samples, round_seed))
        feats = view_features(make_loader(subset, batch_size, workers, device), extractor, box)
        result = distances(
            real_feats,
            mask,
            feats,
            kid_subsets=kid_subsets,
            kid_subset_size=kid_subset_size,
            seed=round_seed,
            device=device,
            with_fid=with_fid,
        )
        log.info("%s: %s", candidate.name, {k: round(v, 5) for k, v in result.items()})
        return result

    first = {c.name: score(c, seed, False) for c in [*refs, *candidates]}
    ranked = sorted(candidates, key=lambda c: first[c.name]["score"])
    finalists = ranked[:confirm]
    second = {c.name: score(c, seed + 1, True) for c in [*refs, *finalists]}
    winner = min(finalists, key=lambda c: second[c.name]["score"]) if finalists else None
    checks = gate(second[winner.name], second[BASE]) if winner else {}

    rows = [
        {
            "name": c.name,
            "reference": c in refs,
            "overrides": c.overrides,
            "first": first[c.name],
            "second": second.get(c.name),
        }
        for c in [*refs, *ranked]
    ]
    rows_px, columns_px = box
    return {
        "spec": spec.name,
        "base": str(spec.base),
        "scale": scale,
        "real": real.describe() | {"typical": spec.typical, "typical_samples": int(mask.sum())},
        "centre": {
            "box": list(spec.centre),
            "rows": [rows_px.start, rows_px.stop],
            "columns": [columns_px.start, columns_px.stop],
        },
        "synthetic": {
            "root": data.root,
            "split": data_split,
            "samples": min(samples, size),
            "manifest": manifest_info(Path(data.root)),
        },
        "seed": seed,
        "kid": {"subsets": kid_subsets, "subset_size": kid_subset_size},
        "winner": winner.name if winner else None,
        "winner_overrides": winner.overrides if winner else None,
        "winner_config": (
            OmegaConf.to_container(OmegaConf.structured(winner.config)) if winner else None
        ),
        "gate": checks,
        "accepted": bool(checks) and all(checks.values()),
        "rows": rows,
        "seconds": time.perf_counter() - start,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "versions": package_versions((*VERSIONED, "torchmetrics", "torch-fidelity")),
    }


def render(results: dict[str, Any], top: int = 15) -> str:
    """A Markdown summary: both rounds, the winner and the gate."""

    def name(row: dict[str, Any]) -> str:
        return f"*{row['name']}*" if row["reference"] else f"`{row['name']}`"

    def kids(scores: dict[str, float]) -> str:
        keys = (*(f"kid_{view}" for view in VIEWS), "score")
        return " | ".join(f"{scores[key] * 1e3:.2f}" for key in keys)

    rows = results["rows"]
    refs = [r for r in rows if r["reference"]]
    ranked = [r for r in rows if not r["reference"]]
    checks = results["gate"]
    failed = [measure for measure, passed in checks.items() if not passed]
    verdict = "accepted" if results["accepted"] else f"not accepted (not better on {failed})"
    lines = [
        f"# Degradation match: {results['spec']} ×{results['scale']}",
        "",
        f"Real: {results['real']['samples']:,} faces ({results['real']['split']}), "
        f"{results['real']['typical_samples']:,} typical. Synthetic: "
        f"{results['synthetic']['samples']:,} CelebA {results['synthetic']['split']} crops. "
        f"Winner: `{results['winner']}`, {verdict}. KID ×10³.",
        "",
        f"## Round 1 (seed {results['seed']}): references and the best {top}",
        "",
        "| Configuration | KID all | KID typical | KID centre | Mean |",
        "|---|---:|---:|---:|---:|",
        *(f"| {name(r)} | {kids(r['first'])} |" for r in [*refs, *ranked[:top]]),
        "",
        f"## Round 2 (seed {results['seed'] + 1}): references and finalists",
        "",
        "| Configuration | KID all | KID typical | KID centre | Mean | FID all | FID centre |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [*refs, *ranked]:
        scores = row["second"]
        if scores:
            fids = f"{scores['fid_all']:.2f} | {scores['fid_centre']:.2f}"
            lines.append(f"| {name(row)} | {kids(scores)} | {fids} |")
    return "\n".join(lines) + "\n"


def winner_yaml(results: dict[str, Any], header: str) -> str:
    """The winning configuration as a commented YAML file."""
    comment = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    body = OmegaConf.to_yaml(OmegaConf.create(results["winner_config"]))
    return comment + body
