"""Checkpoints: weights in a ``.safetensors`` file and metadata in a ``.json`` file next to it.

The JSON file holds the full experiment config, so ``load_checkpoint`` rebuilds the model from
the checkpoint alone, and the sha256 of the weights file, which is checked on loading.
"""

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from facehall.models import build_model
from facehall.provenance import file_sha256
from facehall.train.config import experiment_from_dict

FORMAT = "facehall-checkpoint"
VERSION = 1
SUFFIX = ".safetensors"


def metadata_path(weights: Path) -> Path:
    return weights.with_suffix(".json")


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    """Write the weights of ``model`` to ``path`` and ``metadata`` to the JSON file next to it.

    Both files are written under temporary names and then renamed, so an interrupted save
    leaves the previous checkpoint in place.
    """
    if path.suffix != SUFFIX:
        raise ValueError(f"checkpoint path must end in {SUFFIX}: {path}")
    tensors = {
        name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()
    }
    weights_tmp = path.with_name(path.name + ".partial")
    save_file(tensors, weights_tmp, metadata={"format": FORMAT})
    info = {
        **metadata,
        "format": FORMAT,
        "version": VERSION,
        "file": path.name,
        "sha256": file_sha256(weights_tmp),
    }
    meta = metadata_path(path)
    meta_tmp = meta.with_name(meta.name + ".partial")
    meta_tmp.write_text(json.dumps(info, indent=2) + "\n")
    os.replace(weights_tmp, path)
    os.replace(meta_tmp, meta)


def read_metadata(path: Path) -> dict[str, Any]:
    """Metadata of the checkpoint ``path``, after checking its format and checksum."""
    meta = metadata_path(path)
    info: dict[str, Any] = json.loads(meta.read_text())
    if info.get("format") != FORMAT or info.get("version") != VERSION:
        raise ValueError(f"{meta} is not a {FORMAT} version {VERSION}")
    if file_sha256(path) != info.get("sha256"):
        raise ValueError(f"{path} does not match the sha256 recorded in {meta.name}")
    return info


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model of checkpoint ``path`` in eval mode; returns it with its metadata."""
    info = read_metadata(path)
    config = experiment_from_dict(info["config"])
    model = build_model(config.model, config.scale, config.data.hr_size // config.scale)
    model.load_state_dict(load_file(path))
    return model.to(device).eval(), info
