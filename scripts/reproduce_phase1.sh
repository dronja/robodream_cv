#!/usr/bin/env bash
# Phase 1 end to end (docs/PROJECT.md §10):
#   prepare CelebA -> download the recognition model -> train FNet and CNet (x4, x8)
#   -> evaluate them and the bicubic baseline on the frozen test protocols -> docs/RESULTS.md.
# Full training takes about 1.6 h on an RTX 6000 Ada (docs/PROJECT.md §9): confirm before running.
#
#   bash scripts/reproduce_phase1.sh
#
# Environment: FACEHALL_DATA (default /mnt/storage/AI/Face), FACEHALL_WEIGHTS (default
# $FACEHALL_DATA/weights), CELEBA_MIRROR (default $FACEHALL_DATA/celeba_eurecom/data).
set -euo pipefail
cd "$(dirname "$0")/.."

export FACEHALL_DATA="${FACEHALL_DATA:-/mnt/storage/AI/Face}"
export FACEHALL_WEIGHTS="${FACEHALL_WEIGHTS:-$FACEHALL_DATA/weights}"
mirror="${CELEBA_MIRROR:-$FACEHALL_DATA/celeba_eurecom/data}"

if [ ! -f "$FACEHALL_DATA/celeba/manifest.parquet" ]; then
  uv run python scripts/prepare_dataset.py --src "$mirror" --out "$FACEHALL_DATA/celeba"
fi
if [ ! -f "$FACEHALL_WEIGHTS/buffalo_l/w600k_r50.onnx" ]; then
  uv run python scripts/download_weights.py buffalo_l --root "$FACEHALL_WEIGHTS"
fi

for scale in 4 8; do
  protocol="configs/eval/celeba128_x$scale.yaml"
  uv run facehall eval --config "$protocol" --model bicubic
  for model in fnet cnet; do
    run="$(uv run facehall train --config "configs/train/${model}_x$scale.yaml" | tail -n 1)"
    uv run facehall eval --config "$protocol" --ckpt "$run/best.safetensors"
  done
done
uv run python scripts/make_results.py
