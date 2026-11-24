"""Tune a degradation by matching synthetic LR crops to real ones (docs/PROJECT.md §9, PR 2.6).

The real LR crops come from a real-world set (``kind: realworld`` protocol, usually its ``val``
split, D20). The synthetic LR crops come from a fixed random subset of CelebA HR crops, degraded
by each candidate configuration. A candidate is better when its synthetic crops are closer to
the real ones in InceptionV3 feature space: lower KID (primary) and FID. No model is involved.

A search spec (``configs/degrade/search/*.yaml``) names a base degradation config, reference
configs scored as they are, and axes. Every axis maps option names to overrides (dotted keys of
``DegradeConfig``); the candidates are the Cartesian product of the options, applied to the
base. An empty option keeps the base value.

Selection: the candidates are ranked by KID on one synthetic sample (``seed``). The best
``confirm`` candidates and the references are then scored again on a fresh synthetic sample and
fresh KID subsets (``seed + 1``); the winner is the best on that second sample, which limits the
winner's curse of picking the luckiest of many noisy estimates.
"""

import itertools
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
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


def load_spec(path: Path | str) -> SearchSpec:
    """Read a search spec; file paths in it are relative to the spec file."""
    path = Path(path)
    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise TypeError(f"{path} must hold a mapping")
    data = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(data, dict)
    unknown = set(data) - {"base", "references", "axes"}
    if unknown or "base" not in data:
        raise ValueError(f"{path}: expected keys base, references, axes; got {sorted(data)}")
    axes = data.get("axes") or {}
    for axis, options in axes.items():
        if not isinstance(options, dict) or not options:
            raise ValueError(f"{path}: axis {axis!r} must map option names to overrides")
        for option, overrides in options.items():
            if not isinstance(overrides, dict):
                raise ValueError(f"{path}: option {axis}={option} must be a mapping")
    return SearchSpec(
        name=path.stem,
        base=path.parent / str(data["base"]),
        references=[path.parent / str(p) for p in data.get("references") or []],
        axes={str(a): {str(o): dict(v) for o, v in opts.items()} for a, opts in axes.items()},
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
        candidates.append(Candidate(name=name or "base", overrides=overrides, config=config))
    return candidates


def references(spec: SearchSpec) -> list[Candidate]:
    result = []
    for path in spec.references:
        config = load_config(path, DegradeConfig)
        validate_degrade(config)
        result.append(Candidate(name=path.stem, overrides={}, config=config))
    return result


def dotlist(overrides: dict[str, Any]) -> list[str]:
    return [f"{key}={json.dumps(value)}" for key, value in overrides.items()]


def real_features(
    dataset: RealFaces,
    extractor: FeatureExtractor,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> Features:
    chunks = [
        extractor(to_uint8(b["lr"])).cpu().numpy()
        for b in make_loader(dataset, batch_size, workers, device)
    ]
    return np.concatenate(chunks).astype(np.float32)


def sample_indices(size: int, count: int, seed: int) -> list[int]:
    """A sorted random subset of ``count`` indices (all of them if ``size`` is smaller)."""
    count = min(count, size)
    return sorted(np.random.default_rng(seed).choice(size, count, replace=False).tolist())


def synthetic_features(
    data: DataConfig,
    degrade: DegradeConfig,
    split: str,
    scale: int,
    indices: list[int],
    extractor: FeatureExtractor,
    *,
    seed: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> Features:
    """Features of the LR crops of ``indices``, degraded by ``degrade`` (deterministic)."""
    dataset = AlignedFaces(data, degrade, split, scale, seed=seed, augment=False)
    loader = make_loader(Subset(dataset, indices), batch_size, workers, device)
    chunks = [extractor(to_uint8(b["lr"])).cpu().numpy() for b in loader]
    return np.concatenate(chunks).astype(np.float32)


def distance(
    real: Features,
    synthetic: Features,
    *,
    kid_subsets: int,
    kid_subset_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    mean, std = kid(
        real, synthetic, subsets=kid_subsets, subset_size=kid_subset_size, seed=seed, device=device
    )
    return {"kid": mean, "kid_std": std, "fid": fid(real, synthetic)}


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
    """Score every candidate and reference; returns the results (see the module docstring)."""
    device = device if device is not None else torch.device("cpu")
    scale = real.scale
    if data.hr_size // scale != real.lr_size:
        raise ValueError(f"HR {data.hr_size} / x{scale} does not give the real LR size")
    start = time.perf_counter()
    loader_args: dict[str, Any] = {"batch_size": batch_size, "workers": workers, "device": device}
    real_feats = real_features(real, extractor, **loader_args)
    candidates = expand(spec)
    refs = references(spec)
    size = len(AlignedFaces(data, DegradeConfig(), data_split, scale, augment=False))
    log.info("%d candidates, %d references, %d real faces", len(candidates), len(refs), len(real))

    def score(candidate: Candidate, round_seed: int) -> dict[str, float]:
        indices = sample_indices(size, samples, round_seed)
        feats = synthetic_features(
            data,
            candidate.config,
            data_split,
            scale,
            indices,
            extractor,
            seed=round_seed,
            **loader_args,
        )
        result = distance(
            real_feats,
            feats,
            kid_subsets=kid_subsets,
            kid_subset_size=kid_subset_size,
            seed=round_seed,
            device=device,
        )
        log.info("%s: KID %.5f, FID %.2f", candidate.name, result["kid"], result["fid"])
        return result

    first = {c.name: score(c, seed) for c in [*refs, *candidates]}
    ranked = sorted(candidates, key=lambda c: first[c.name]["kid"])
    finalists = ranked[:confirm]
    second = {c.name: score(c, seed + 1) for c in [*refs, *finalists]}
    winner = min(finalists, key=lambda c: second[c.name]["kid"]) if finalists else None

    rows = []
    for candidate in [*refs, *ranked]:
        rows.append(
            {
                "name": candidate.name,
                "reference": candidate in refs,
                "overrides": candidate.overrides,
                "first": first[candidate.name],
                "second": second.get(candidate.name),
            }
        )
    return {
        "spec": spec.name,
        "base": str(spec.base),
        "scale": scale,
        "real": real.describe(),
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
        "rows": rows,
        "seconds": time.perf_counter() - start,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "versions": package_versions((*VERSIONED, "torchmetrics", "torch-fidelity")),
    }


def render(results: dict[str, Any], top: int = 15) -> str:
    """A Markdown summary: references, the best candidates, the winner."""

    def cells(scores: dict[str, float] | None) -> str:
        if not scores:
            return "— | —"
        return f"{scores['kid'] * 1e3:.2f} ± {scores['kid_std'] * 1e3:.2f} | {scores['fid']:.2f}"

    lines = [
        f"# Degradation match: {results['spec']} ×{results['scale']}",
        "",
        f"Real: {results['real']['samples']:,} faces ({results['real']['split']}). Synthetic: "
        f"{results['synthetic']['samples']:,} CelebA {results['synthetic']['split']} crops. "
        f"Winner: `{results['winner']}`.",
        "",
        "| Configuration | KID ×10³ (seed 0) | FID (seed 0) | KID ×10³ (seed 1) | FID (seed 1) |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = results["rows"]
    shown = [r for r in rows if r["reference"]] + [r for r in rows if not r["reference"]][:top]
    for row in shown:
        name = f"*{row['name']}* (reference)" if row["reference"] else f"`{row['name']}`"
        lines.append(f"| {name} | {cells(row['first'])} | {cells(row['second'])} |")
    return "\n".join(lines) + "\n"


def winner_yaml(results: dict[str, Any], header: str) -> str:
    """The winning configuration as a commented YAML file."""
    comment = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    body = OmegaConf.to_yaml(OmegaConf.create(results["winner_config"]))
    return comment + body
