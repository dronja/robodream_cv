# Changelog

All notable changes to the `facehall` package and its reference models. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0.0, a minor release may
change the command line, the API and the model contract. Model versions and their rules:
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) §12.

## [Unreleased]

### Added

- **`facehall ui`:** a local web interface in Material Design 3, in English and Ukrainian.
  Photos, videos and folders are chosen on this computer and read from the disk, without
  uploads. Runs wait in a queue, show their progress and can be cancelled. A viewer compares each
  result with its original (a slider, the faces enlarged, the details), and the outputs, marked
  like those of `facehall enhance`, go to a new folder per run. It binds to 127.0.0.1 and needs a
  session key (D41).
- `ui` settings in `configs/pipeline/default.yaml`. The records in the marks of the outputs do
  not include them, so `facehall enhance` writes the same records as before.

## [0.1.0] - 2026-11-27

The first release of the package. The research notebook (now in `legacy/`) became a package with
training, evaluation, an inference pipeline, a command-line tool, a local REST API and Docker
images. The history of each step is in `docs/PROJECT.md` §9 and `docs/DECISIONS.md`.

### Added

- **Data:** preparation of CelebA aligned from a parquet mirror, with a manifest and the official
  identity-disjoint split (`scripts/prepare_dataset.py`), and of a real-world test set of small
  faces from WIDER FACE (`scripts/prepare_wider.py`).
- **Alignment:** a five-point template, similarity transforms, warp and inverse warp, alignment
  jitter (`facehall.align`).
- **Degradations** from configs: blur, resizing, noise, JPEG, a second stage, grayscale; three
  frozen test levels (`configs/degrade/`).
- **Models:** a bicubic baseline, ports of the legacy FNet and CNet, and SRResNet-v2 without
  BatchNorm (`facehall.models`).
- **Training:** `facehall train` with run directories, checkpoints, EMA, resume, and L1,
  perceptual, adversarial and identity losses; the perceptual and identity losses are made
  robust to pixel patterns (`facehall.losses`).
- **Evaluation:** `facehall eval` with PSNR-Y, SSIM-Y, LPIPS, ArcFace ID similarity,
  subgroups, FID and KID on real photos, a spectral check for periodic patterns, frozen input
  digests, a quality gate (`scripts/quality_gate.py`) and a generated `docs/RESULTS.md`.
- **Inference pipeline:** YuNet face detection, alignment, batched restoration, feathered
  paste-back (`facehall.pipeline`).
- **`facehall enhance`** for photos, folders and videos (face tracks, audio kept), with face boxes
  as a fallback.
- **Output marking:** PNG text, XMP with the IPTC digital source type, EXIF and video container
  tags; an optional visible watermark.
- **`facehall export onnx`** with a parity check against PyTorch, and ONNX Runtime inference
  without PyTorch on CPU or CUDA.
- **`facehall serve`:** a local REST API (`POST /enhance`, `/health`, `/version`) with limits, a
  queue and logs without image data.
- **Docker images** for CPU and GPU, without weights, checked in CI
  (`scripts/smoke_image.py`), and a latency and memory benchmark
  (`scripts/benchmark_images.py`).
- **Documentation:** `docs/PROJECT.md`, `docs/DECISIONS.md`, `docs/DATA.md`,
  `docs/THIRD_PARTY_LICENSES.md`, `docs/RESULTS.md`, `docs/MODEL_CARD.md`.
- **Reference models** `facehall-x4` 0.1.0 and `facehall-x8` 0.1.0, recorded by checksum in
  `docs/models.sha256`. The weights are not published (D39).

### Changed

- The README describes the package and its measured results.

### Removed

- The README link to the legacy dataset and the figures that showed a face from it, and the
  image outputs of the legacy notebook (D38, D40).

### Security

- ONNX Runtime telemetry, on by default in its official builds, is turned off when facehall is
  imported and in the images. FastAPI's telemetry is off too.

[Unreleased]: https://github.com/dronja/robodream_cv/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dronja/robodream_cv/tree/v0.1.0
