# facehall — project overview

> **Single source of truth** for facts, current state, environment, data and the work plan.
> Last updated: **2026-11-24**.
>
> - Rules of work and product requirements: [`CLAUDE.md`](../CLAUDE.md) (it imports this file).
> - Decisions with dates and reasons: [`DECISIONS.md`](DECISIONS.md).
> - Update this file **in the same PR** as the change it describes: status, checkboxes, commands,
>   verified facts. Numbers only from real runs; anything unverified is marked *needs verification*.
> - Do not duplicate facts between documents — link instead.

## Contents

1. [Information map](#1-information-map)
2. [Status snapshot](#2-status-snapshot)
3. [Repository contents](#3-repository-contents)
4. [Development environment](#4-development-environment)
5. [Legacy research code](#5-legacy-research-code)
6. [Data](#6-data)
7. [Decisions — quick reference](#7-decisions--quick-reference)
8. [Licenses — status](#8-licenses--status)
9. [Phase 1 technical design](#9-phase-1-technical-design)
10. [Roadmap and progress](#10-roadmap-and-progress)
11. [Open questions, risks, tech debt](#11-open-questions-risks-tech-debt)
12. [Commands](#12-commands)

---

## 1. Information map

| What | Where | Changes |
|---|---|---|
| Rules of work, product requirements (data, evaluation, responsible use) | `CLAUDE.md` | rarely |
| Facts, status, environment, data, plan, progress | `docs/PROJECT.md` (this file) | every PR |
| Decisions with date and reason | `docs/DECISIONS.md` | append-only |
| Dataset composition, verified source facts, layout, checksums, how to reproduce it | `docs/DATA.md` | when data changes |
| Third-party licenses | `docs/THIRD_PARTY_LICENSES.md` | every PR that adds a dependency, dataset or weights |
| Metric tables (generated) | `docs/RESULTS.md` | *planned, PR 1.7* |
| Model card | `docs/MODEL_CARD.md` | *planned, Phases 3 and 6* |

## 2. Status snapshot

As of 2026-11-24:

- **Phase 0 (repository hygiene): done.** Merged into `main` via PR #3 on 2026-11-23 (merge
  commit `0ac43f9`), CI green.
- **Phase 1: in progress.**
  - Merged on 2026-11-24: PR 1.1 (ML dependencies, PR #4, `8a0ccce`), PR 1.2 (dataset
    preparation, PR #5, `6cf65af`), PR 1.3 (alignment, PR #6, `0a9fc82`), PR 1.4 (dataset +
    degradation v0, PR #7, `2ece720`) and PR 1.5 (models, PR #8, `d2a6d1b`). The prepared CelebA
    dataset is at
    `/mnt/storage/AI/Face/celeba` (§6.2).
  - In review: PR 1.6 (trainer and `facehall train`; branch `feat/train`). Smoke runs on the real
    data pass for all four configs (§9).
  - Next: full runs of FNet and CNet × (×4, ×8), after the user confirms them (about 1.6 h in
    total, §9).
  - The technical design is agreed (§9).
- **Scope:** the product is used locally only, with no production deployment (D13).
- Legacy weights are still tracked in `legacy/`; uploading them to GitHub Releases needs a separate
  confirmation (D7).

## 3. Repository contents

```
robodream_cv/
├── CLAUDE.md                  # rules for Claude Code (Ukrainian); imports this file
├── README.md                  # public description (English)
├── LICENSE
├── pyproject.toml, uv.lock    # uv project, hatchling build; ruff, mypy, pytest config
├── configs/                   # data/celeba_128.yaml, degrade/bicubic.yaml,
│                              # model/{bicubic,fnet,cnet}.yaml, train/{fnet,cnet}_x{4,8}.yaml
├── .python-version            # 3.12
├── .pre-commit-config.yaml    # whitespace, yaml, toml, large files; ruff and mypy via `uv run`
├── .github/workflows/ci.yml   # uv sync --locked → ruff check, ruff format --check → mypy → pytest
├── src/facehall/              # __init__.py (version 0.0.0), cli.py (--version, train), py.typed
│   ├── provenance.py          # file sha256, git commit, package versions
│   ├── data/                  # manifest.py, celeba.py, config.py (schemas), degrade.py,
│   │                          # dataset.py (AlignedFaces)
│   ├── config.py              # load_config: dataclass schema + YAML (include, overrides)
│   ├── models/                # bicubic.py, fnet.py, cnet.py, layers.py (Keras defaults)
│   ├── train/                 # config.py (ExperimentConfig), trainer.py, checkpoint.py, ema.py
│   └── align/                 # similarity.py (Umeyama, matrices), template.py, warp.py
├── scripts/                   # prepare_dataset.py (CelebA mirror → files + manifest),
│                              # license_report.py (license table from package metadata)
├── tests/                     # conftest (load_script, synthetic prepared dataset); import, CLI,
│                              # legacy layout, dependency
│                              # smoke, license report, provenance, manifest, CelebA preparation,
│                              # alignment (similarity, template, warp), config, degrade,
│                              # dataset, models, train
├── docs/                      # PROJECT.md, DECISIONS.md, DATA.md, THIRD_PARTY_LICENSES.md
├── images/                    # figures used by README: CNET, FNET, PULSE, SRgan (.png)
└── legacy/                    # read only: FH.ipynb, FNet_24.keras, CNet_24.keras, SRgan_20.pth
```

- Package `facehall` 0.0.0, Python ≥ 3.12. Runtime dependencies: numpy, omegaconf,
  opencv-python-headless, pillow, pyarrow, safetensors, torch, torchmetrics[image], torchvision
  (versions and licenses: `docs/THIRD_PARTY_LICENSES.md` §4.1). Dev group: ruff, mypy, pytest,
  pre-commit. Entry point: `facehall = facehall.cli:main`.
- PyTorch build: by default torch comes from PyPI (CUDA 13.2 on Linux). The `cpu` extra switches
  to the CPU-only build from `download.pytorch.org/whl/cpu`; the `cpu` and `cuda` extras are
  declared as conflicting, so `uv.lock` holds both builds.
- ruff: line length 100, rules `E F I UP B SIM`, `legacy/` excluded. mypy: strict on `src`,
  `tests` and `scripts`; pyarrow and torchvision have no type information and are excluded from
  import checks.
- Scripts in `scripts/` are thin command-line wrappers; the logic lives in the package. Tests
  import scripts through the `load_script` fixture (`tests/conftest.py`).
- CI runs on pushes to `main` and on every pull request. It installs the CPU build
  (`uv sync --locked --extra cpu`) and sets `UV_NO_SYNC=1`, so `uv run` keeps that build.
- The target layout (configs/, data/, align/, models/, …) is described in `CLAUDE.md` §6.

## 4. Development environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada, 48 GB (driver 595.99, CUDA 13.2) |
| Portable target | RTX 4070 Laptop, 8 GB — every training config needs a profile that fits 8 GB (D6) |
| Python | 3.12.14 via uv for the project (`requires-python >=3.12`); the system Python 3.14 is not used |
| PyTorch | torch 2.15.0+cu132 from PyPI; CUDA and bf16 on the RTX 6000 Ada verified 2026-11-23. `.venv` with the CUDA build ≈5.9 GB, CPU-only environment ≈1.5 GB |
| uv | 0.13.5 in `~/.local/bin` (add it to `PATH` in non-login shells) |
| Locked dev tools | ruff 0.17.2, mypy 2.4.0, pytest 9.1.1 |
| GitHub CLI | gh 2.46.0, authenticated (git protocol: ssh) |
| Pushing | The gh OAuth token has no `workflow` scope, so HTTPS pushes that touch `.github/workflows/` are rejected. SSH works (`git@github.com:dronja/robodream_cv.git`). Fix: `gh auth refresh -h github.com -s workflow`, or switch `origin` to SSH |
| Docker | not installed (needed in Phase 5) |
| Data root | `/mnt/storage/AI/Face` (~18 TB free): `celeba_eurecom/` (raw mirror), `celeba/` (prepared, §6.2). The repo `data/` directory is gitignored |
| Remote | `https://github.com/dronja/robodream_cv` (public), default branch `main` |

## 5. Legacy research code

`legacy/` is read only. Known problems of this code, and what must not be carried over:
`CLAUDE.md` §4.

| File | What it is | Notes |
|---|---|---|
| `FH.ipynb` (456 KB) | All code: TF/Keras FNet and CNet, PyTorch SRGAN generator, training, visualisation | Cells were run out of order; SRGAN training (discriminator, loop) is missing |
| `FNet_24.keras` (9.0 MB) | "Simplified SRResNet": 24×24 → 96×96 (×4), 5 residual blocks with BN, PixelShuffle via `Lambda(depth_to_space)`, sigmoid, MSE | Best legacy result. Keras 3.9.2, contains optimizer state, loading needs `safe_mode=False` |
| `CNet_24.keras` (10.3 MB) | "Simplified C-Face": Conv + `UpSampling2D`, no residual blocks, MSE | Slightly worse than FNet |
| `SRgan_20.pth` (3.0 MB) | PyTorch SRGAN generator `state_dict` | Same layer sequence as FNet, but with one shared PReLU alpha (`nn.PReLU()`) instead of one per channel and a linear output (no sigmoid, no clamp). The notebook loads `best_generator_20.pth` (name mismatch) |

- **Legacy dataset:** 209,813 faces, 89×109 px, "FFHQ 2" + CelebA + Internet images; download link
  (GDrive) in README; the exact composition is undocumented. 89×109 is exactly half of CelebA
  aligned (178×218).
- **Legacy timing:** one epoch on the RTX 4070 Laptop took ≈4 min for FNet and ≈2 min for CNet
  (≈168k images, batch 16).
- **Comparability:** legacy numbers are not comparable with the new protocol and are not reused.
  The FNet and CNet architectures are re-trained on the new protocol as baselines (Phase 1).

## 6. Data

### 6.1 Sources

| Source | Role | Status |
|---|---|---|
| CelebA aligned, HF mirror `eurecom-ds/celeba` | Training and evaluation, Phases 1–3 (D8) | Downloaded, verified and prepared (PR 1.2, §6.2) |
| Legacy 89×109 dataset | — | Excluded (D9) |
| FFHQ | Candidate HR source for 512 px (later) | Not downloaded; non-commercial |
| Real-world low-quality test set | Phase 2 | Source not chosen yet: needs known origin and license or consent |

### 6.2 CelebA — summary

Details, per-split counts, checksums and the evidence behind every statement: `docs/DATA.md`.

- **Raw mirror** (read only): `/mnt/storage/AI/Face/celeba_eurecom/data/`, 20 parquet shards,
  8.92 GB.
- **Prepared dataset:** `/mnt/storage/AI/Face/celeba/`, built by `scripts/prepare_dataset.py`
  (§12). It holds `img_align/{train,val,test}/NNNNNN.{png,jpg}`, `manifest.parquet` and
  `prepare_info.json`. The manifest stays next to the data (D14).
- **Size:** 202,599 images and 10,177 identities. The official partition is identity-disjoint
  (D8).
- **Images:** all RGB 178×218. Train images are PNG files of decoded JPEG pixels; validation and
  test images are the original JPEG files. The bytes are kept as stored, without re-encoding.
- **Annotations:**
  - usable: identity, aligned 5-point landmarks, the 40 binary attributes (in the standard
    CelebA order);
  - not usable: `bbox`, which holds in-the-wild coordinates;
  - not stored: the original file names. Rows are traced by `source_file` + `source_row`.

### 6.3 Alignment and crop geometry

The geometry the alignment code (`facehall.align`, PR 1.3) builds on. Measured on 2026-11-24 over
the prepared manifest.

**Conventions**
- Landmarks and crops use pixel-centre coordinates (OpenCV). The CelebA landmark convention is
  not documented: the pixel-centre reading is an assumption, and its effect is at most 0.5 source
  px.
- The template uses normalized crop coordinates: (0, 0) is the top-left corner of the crop and
  (1, 1) its bottom-right corner. One template therefore serves every crop size (32, 128, 512).

**Template** (`facehall.align.CELEBA_TEMPLATE`)
- Derived from the train-split mean landmarks, so the test split is not used for design.
- Construction: the FFHQ alignment rule reduced to 5 points, a square of side
  max(4 × eye distance, 3.6 × eye-to-mouth distance).
- Then averaged with its mirror image, so that a horizontal flip commutes with alignment. This
  moves no point by more than 0.34 px at 128.
- Normalized (x, y): eyes (0.375, 0.47323) and (0.625, 0.47323), nose (0.5, 0.62947), mouth
  corners (0.388, 0.74094) and (0.612, 0.74094).
- On the mean train landmarks, the rule gives a square of side 153.1 px with x ∈ [12.0, 165.1] and
  y ∈ [38.7, 191.8], inside 178×218.

**Per-image alignment:** a least-squares similarity (Umeyama) maps the 5 landmarks to the
template.

| | train | val | test |
|---|---|---|---|
| Crop side in the source, median (min – max) | 155.8 px (143.4 – 256.9) | 155.9 px (144.4 – 234.7) | 156.0 px (39.3 – 247.8) |
| Crops whose HR 128 would be an upscale (side < 128) | 0 | 0 | 1 |
| Crops that need padding | 4,543 (2.79%) | 479 (2.41%) | 540 (2.71%) |
| Padded area of those crops: mean / p95 / max | 2.8% / 11.8% / 41.6% | 2.6% / 11.6% / 29.6% | 2.9% / 13.6% / 36.8% |
| Crops with more than 10% padding | 331 | 32 | 48 |

- **Upscale check:** so HR 128 is a downscale (×0.82 at the median) for every crop except the one
  test image with broken landmarks (`docs/DATA.md` §4.5). This confirms D4.
- **Large crops** come from profile and small faces: the 5 points fit the frontal template worse,
  and the face appears smaller in the crop.
- **Rotation** from the alignment is small, because the source is already aligned: median |angle|
  0.47°, p95 1.6°.
- **Cost:** decode 0.6 ms and align + warp 0.08 ms per image (single core).

| Scale | LR | HR | Interocular distance at the template |
|---|---|---|---|
| ×4 | 32×32 | 128×128 | 8 px in LR, 32 px in HR |
| ×8 | 16×16 | 128×128 | 4 px in LR, 32 px in HR |

### 6.4 Known data limitations

- All HR targets carry JPEG compression artifacts. Validation and test images are small JPEG files
  (≈6.8 KB each), and the train PNGs hold decoded JPEG pixels (`docs/DATA.md` §4.2). This is
  acceptable for 128 px but not for 512 px.
- Images with `Blurry = 1` (about 5% of every split) are poor HR targets. The manifest keeps them
  in the `attr_blurry` column. Whether to filter them is decided in PR 1.4 (Q4).
- There is no skin-tone annotation: `Pale_Skin` is a binary attribute, not a skin-tone scale.
  The plan is in D10.
- Some CelebA aligned images already contain stretched, streaky borders from the dataset's own
  alignment. Seen on 2026-11-24 in a train crop whose `padded` fraction was 0: the streaks come
  from the source image, not from our crop. How often this happens is not measured yet; `padded`
  (D16) covers only our crop reaching outside 178×218.
- Celebrity photos are not representative of real-world low-quality inputs; the Phase 2
  real-world test set addresses this.
- CelebA terms: non-commercial research only, no redistribution. They were verified on the
  official page and are quoted in `docs/THIRD_PARTY_LICENSES.md` §2. The mirror states no license
  of its own. Consequence: no CelebA images, annotations or manifests go into the public
  repository (D14, Q2).

## 7. Decisions — quick reference

The authoritative log, with reasons, is [`DECISIONS.md`](DECISIONS.md).

| ID | Decision |
|---|---|
| D1 | Offline restoration of photos and video frames |
| D2 | Non-commercial / research use |
| D3 | PyTorch only; the Keras code is legacy reference |
| D4 | ×4 (32→128) and ×8 (16→128) on aligned 128×128 crops from CelebA aligned; 512 px later, with another HR source |
| D5 | CLI + REST API (FastAPI) in Docker (CPU and GPU images) |
| D6 | Local training; every config fits 8 GB VRAM, plus an optional profile for 48 GB |
| D7 | No git history rewrite; weights go to GitHub Releases only after a separate confirmation |
| D8 | Phase 1 data: CelebA aligned from the HF mirror, official identity-disjoint partition |
| D9 | The legacy dataset (including "FFHQ 2" and Internet images) is not used in Phase 1 |
| D10 | Skin-tone subgroup: look for an additional annotation source (license verified and approved first); until then, document the limitation |
| D11 | Approved downloads: PyTorch, ArcFace weights for the ID-similarity metric, LPIPS weights |
| D12 | Documentation layout: this file is the single source of truth for facts, status and the plan |
| D13 | Local use only: no production deployment, public service or published Docker image |
| D14 | Manifests stay next to the data, outside git; the repo holds the generator code, counts and checksums (`docs/DATA.md`) |
| D15 | `Blurry = 1` images are dropped from train only; evaluation also reports the non-blurry subset |
| D16 | Crops reaching outside the image use reflect padding and are kept; each sample carries its `padded` fraction; evaluation also reports the unpadded subset |
| — | Legacy ports keep Keras default initialisation and CNet's per-pixel PReLU (`DECISIONS.md`, 2026-11-24) |
| D17 | Experiment tracking: a plain run directory per run (config, provenance, metrics JSONL, checkpoints); no MLflow for now |

## 8. Licenses — status

The rules and the known-license table are in `CLAUDE.md` §8. The record for everything in use is
`docs/THIRD_PARTY_LICENSES.md`. Status of the items that still need work:

| Item | Status |
|---|---|
| CelebA and the HF mirror | Done: the CelebA terms were verified on the official page; the mirror states no license |
| Python runtime dependencies | Done in PR 1.1 (`docs/THIRD_PARTY_LICENSES.md` §4) |
| ArcFace weights (model chosen in PR 1.7) | InsightFace pretrained models: non-commercial research (compatible with D2) |
| LPIPS linear weights (bundled in torchmetrics) and torchvision backbone weights | *needs verification* (PR 1.7) |
| NVIDIA CUDA libraries (proprietary) and FFmpeg in OpenCV (LGPL-2.1) | Local use only, nothing is distributed (D13). Review the redistribution terms if that changes |
| Legacy dataset on GDrive (linked from README) | Contains CelebA images and Internet images of unknown origin (Q3) |

## 9. Phase 1 technical design

The agreed approach. Details are fixed in the corresponding PRs.

- **General.** PyTorch 2.x, YAML configs with OmegaConf, no magic constants in the code. Every run
  stores its config, git hash, seed and library versions in its own run directory (D17).
- **Dependencies (PR 1.1, done).** torch 2.15 / torchvision 0.29: the PyPI build (CUDA 13.2)
  locally and the CPU build in CI (§3). torch 2.15 CUDA wheels exist only for CUDA 13.2 (PyPI and
  the `cu132` index), so machines need a driver that supports CUDA 13.2 — check this on the 8 GB
  laptop. Also added: omegaconf, safetensors, pyarrow, opencv-python-headless (affine warps),
  torchmetrics[image]. MLflow moved to PR 1.6: it pulls in ~70 more packages (flask, sqlalchemy,
  docker, databricks-sdk, scikit-learn, …), so the tracking backend is chosen together with the
  trainer.
- **Data preparation (PR 1.2, done).** `scripts/prepare_dataset.py` wraps
  `facehall.data.celeba`.
  - Output: image files with the stored bytes (train PNG, val/test JPEG), `manifest.parquet` and
    `prepare_info.json`.
  - Before writing anything, it checks the shard set, the official counts and identity
    disjointness. Every image is decoded once.
  - Runtime about 2 min on this machine. Details: `docs/DATA.md`.
- **Alignment (PR 1.3, done).** `facehall.align` holds:
  - the template and its derivation (§6.3);
  - `umeyama` and matrix helpers;
  - `alignment_matrix(landmarks, template, size)`;
  - `warp`: bicubic by default; border reflect, constant or replicate;
  - `unwarp`, the inverse warp for paste-back in Phase 4;
  - `valid_mask`, which marks padded pixels;
  - `Jitter`: a random similarity about the crop centre, with shift, rotation and log-uniform
    scale. Its limits are set in the PR 1.4 config. Every sample draws four numbers, so the random
    stream does not depend on the settings.
- **Dataset + degradation v0 (PR 1.4, done).** `facehall.data.dataset.AlignedFaces(data, degrade,
  split, scale, seed=…)`, configured by `configs/data/celeba_128.yaml` and
  `configs/degrade/bicubic.yaml` (`facehall.config.load_config`).
  - Data root: `${FACEHALL_DATA:-/mnt/storage/AI/Face}/celeba`.
  - Dropped samples:
    - HR crops that would be upscales (`CLAUDE.md` §7): one test image;
    - `Blurry = 1` images in train (D15).
  - Resulting sizes: train 154,408, val 19,867, test 19,961 (`docs/DATA.md` §3).
  - Each sample:
    - the source is warped once with the alignment matrix, composed in train with jitter and a
      horizontal flip;
    - padding is reflect (D16);
    - LR is the degradation of the HR crop;
    - output: `hr` (3×128×128) and `lr` (3×32×32 or 3×16×16), both float32 in [0, 1] on 8-bit
      levels, plus `index`, `padded` and `blurry`.
  - Degradation v0: torch bicubic with antialias, then 8-bit quantization. It matches PIL
    `BICUBIC` within 1 level; without antialias the difference reaches 142 levels. Each run
    records the exact function (`describe()`).
  - Randomness: a per-sample generator seeded with `(seed, epoch, index)` in train (call
    `set_epoch`) and `(seed, index)` in val/test, so samples do not depend on workers or order.
  - Validation and test crops are computed on the fly, deterministically, not cached (see
    `DECISIONS.md`).
  - Throughput: 909 samples/s single-process. With 8 / 16 DataLoader workers it is 4,200 / 9,900
    samples/s, and 7,700 / 13,100 samples/s when `cv2.setNumThreads(1)` is called in the main
    process first. That is 0.2–0.6 min per train epoch, so data loading is not the bottleneck.
- **Models (PR 1.5, done).** `facehall.models.build_model(config, scale, lr_size)` with
  `configs/model/{bicubic,fnet,cnet}.yaml`. Input and output are (B, 3, h, w) in [0, 1].
  - **Baseline:** `bicubic` (torch bicubic upsampling, clamped).
  - **Ports:** FNet and CNet, layer for layer as in `FH.ipynb`, including BN, the sigmoid output
    and the Keras defaults (Glorot init, zero bias, BN eps 1e-3 / momentum 0.99, PReLU alpha 0).
    `scale` is generalised as one ×2 stage per factor of 2.
  - **Parameter counts:** equal to the notebook's `model.summary()` outputs at 24 px ×4:
    - FNet: 734,723 trainable + 1,408 BN statistics;
    - CNet: 854,979 + 384.
  - **CNet's `PReLU()`** has no `shared_axes`, so it learns one alpha per channel *and* pixel. The
    port keeps this, so CNet accepts only its LR size, and its size grows with it: 1,457,091
    parameters at 32 px ×4, 1,510,531 at 16 px ×8. FNet has 734,723 parameters at ×4 and 882,499
    at ×8.
  - **Legacy Keras weights** cannot be loaded as they are: the layout is NCHW, and PyTorch's
    pixel-shuffle channel order differs from `tf.nn.depth_to_space`. They are not needed (§5).
  - **Training step on the RTX 6000 Ada** (Adam, MSE, 128 px HR):

    | Model | Batch | bf16 | fp32 | Peak memory (bf16 / fp32) |
    |---|---|---|---|---|
    | FNet ×4 | 16 / 64 | 3,800 / 3,700 img/s | 2,000 / 1,800 img/s | 0.3 / 1.2 GiB; 2.4 / 3.7 GiB |
    | FNet ×8 | 16 / 64 | 4,100 / 4,200 img/s | 2,200 / 2,000 img/s | 0.2 / 0.7 GiB; 0.4 / 1.4 GiB |
    | CNet ×4 | 16 / 64 | 4,000 / 3,000 img/s | 2,800 / 2,400 img/s | 0.5 / 1.9 GiB; 2.7 / 5.4 GiB |
    | CNet ×8 | 16 / 64 | 3,900 / 3,000 img/s | 2,700 / 2,400 img/s | 0.4 / 1.5 GiB; 0.5 / 1.9 GiB |

    So every configuration fits 8 GB (D6), and one train epoch (154,408 samples) takes about
    40–50 s in bf16 on this GPU.
  - **Legacy training recipe** (for the PR 1.6 configs):
    - inputs in [0, 1], MSE loss;
    - Adam with Keras defaults (lr 1e-3, β 0.9 / 0.999, ε 1e-7), batch 16;
    - FNet: 10 epochs with early stopping (patience 3 on val loss);
    - CNet: 50 epochs.
- **Trainer + CLI (PR 1.6, done).** `facehall train --config configs/train/<name>.yaml [--smoke]
  [--set key=value …]`, implemented by `facehall.train.train(config)`.
  - **Configs:** `configs/train/{fnet,cnet}_x{4,8}.yaml` carry the legacy recipe (see "Legacy
    training recipe" above). They pull in the data, degradation and model configs through
    `include`, and `--set` overrides any value.
  - **No separate 48 GB profile:** all four configs fit 8 GB at the recipe batch size of 16 (peak
    ≤ 2.8 GiB even in fp32, table above). The batch size is part of the recipe, so a larger one
    is an experiment (`--set train.batch_size=64`), not a profile (D6).
  - **Run directory** `runs/<name>-<YYYYmmdd-HHMMSS>[-smoke]/` (D17):
    - `config.yaml`, the resolved config;
    - `run.json`, provenance: command, git commit, library versions, device, dataset settings and
      counts, and the manifest checksum;
    - `metrics.jsonl`, one record per logged step and per epoch;
    - `train.log`, `summary.json`;
    - checkpoints `last` and `best` (lowest val loss), each a `.safetensors` file with a `.json`
      file next to it. The JSON holds the full config and the sha256 of the weights;
      `load_checkpoint` rebuilds the model from these two files alone.
  - **Training:**
    - Adam;
    - bf16 autocast for the forward pass, with the loss and validation in fp32;
    - gradient accumulation (`train.grad_accum`);
    - early stopping on val loss (`train.early_stopping`);
    - optional EMA of the weights (`train.ema_decay`). With EMA, validation and `best` use the
      EMA weights, and `last_ema` is saved too.
  - **Validation:** after every epoch, on the full val split. It reports the mean per-image
    loss and an RGB PSNR, which is only a monitoring value; the evaluation protocol is PR 1.7.
  - **Reproducibility:**
    - the seed fixes model initialisation, sample order (a permutation per `(seed, epoch)`) and
      augmentation;
    - on CPU, two runs with one seed give identical weights (tested);
    - on GPU, cuDNN autotuning allows small differences.
  - **DataLoader notes,** verified 2026-11-24:
    - call `cv2.setNumThreads(1)` in the main process before the workers start; the trainer does
      this;
    - **never call it inside a forked worker**: once the parent has used OpenCV's thread pool,
      that call deadlocks;
    - workers are not persistent, so each epoch's workers see `set_epoch`.
  - **Smoke runs on the real data,** 2026-11-24 (RTX 6000 Ada, bf16, 8 workers):
    - **Default `--smoke`** (512 train / 256 val samples, 2 epochs): about 4 s per config, all
      four configs.
    - **Timing runs** (40,000 train samples, full val, 2 epochs): 3,600–3,800 train img/s and
      3.2–3.5 s per validation pass for every config, so a full epoch takes about 45–47 s.
    - **Estimated full runs:** FNet ×4 / ×8 at most 10 epochs, about 8 min each; CNet ×4 / ×8
      50 epochs, about 39 min each. About 1.6 h in total when run one after another.
    - **Early val loss:** in the 32-step default smoke runs, val loss is 1.7–6.8× the train loss. In
      the 2,500-step timing runs it is already below the train loss, which is computed on
      augmented samples. This is consistent with BN running statistics that follow slowly
      (Keras momentum 0.99), but the cause was not checked separately.
  - **Not done:** resuming from `last`, because the optimizer state is not saved (tech debt,
    §11).
- **Evaluation (PR 1.7).** PSNR and SSIM on the Y channel, LPIPS, and ID similarity (cosine of
  ArcFace embeddings, SR vs HR). `facehall eval`. `docs/RESULTS.md` is generated by a script. NME
  moves to Phase 3 because it needs a landmark model.
- **Reproduction.** `scripts/reproduce_phase1.sh` runs prepare → train (FNet, CNet × ×4, ×8) →
  eval → `RESULTS.md`.

## 10. Roadmap and progress

Each step is compared with the previous best on the same frozen test set. A smoke run (a few
hundred images, 1–2 epochs, under 5 min) comes before every full run. Full training (longer than
~30 min) starts only after user confirmation.

### Phase 0 — repository hygiene (*done 2026-11-23*)

- [x] Python `.gitignore` (`data/`, `runs/`, `mlruns/`, `outputs/`, weights and checkpoints)
- [x] Move `FH.ipynb`, `*.keras`, `*.pth` to `legacy/` (`git mv`)
- [x] `pyproject.toml` (uv), ruff, mypy, pytest, pre-commit, CI
- [x] README fixes (C-Face reference, input sizes); `docs/DECISIONS.md`
- [x] `docs/PROJECT.md` as the single source of truth; pre-commit uses the locked ruff
- [x] PR merged with green CI (PR #3)
- [ ] *(not blocking)* Legacy weights uploaded to GitHub Releases with checksums, then removed from
      HEAD (D7, separate confirmation)

**Done when** `uv sync && uv run pytest` passes on a clean machine and CI is green.

### Phase 1 — package and reproducible baseline

- [x] **PR 1.1** `build:` ML dependencies (CUDA locally, CPU in CI); `THIRD_PARTY_LICENSES.md`;
      `scripts/license_report.py`
- [x] **PR 1.2** `feat(data):` `prepare_dataset.py`: files, manifest, official split, frozen test;
      `DATA.md`. Tests use a synthetic parquet fixture
- [x] **PR 1.3** `feat(align):` template, similarity transform, warp and inverse warp, jitter.
      Tests: warp and inverse-warp round trip on synthetic points
- [x] **PR 1.4** `feat(data):` dataset + degradation v0 (bicubic). Tests: determinism, shapes,
      value ranges
- [x] **PR 1.5** `feat(models):` bicubic, FNet, CNet. Tests: output shapes, parameter counts
      against legacy
- [x] **PR 1.6** `feat(train):` trainer, checkpoints, experiment tracking, `facehall train`. Tests: CPU smoke
      run on fixtures
- [ ] **PR 1.7** `feat(eval):` PSNR, SSIM, LPIPS, ID-sim, `facehall eval`, generated
      `RESULTS.md`. Tests: metrics on known pairs
- [x] Smoke run on real data; estimate the duration of a full run (§9: about 1.6 h for all four)
- [ ] Full runs FNet, CNet × (×4, ×8) — after confirmation
- [ ] `RESULTS.md` (bicubic vs FNet vs CNet) and `scripts/reproduce_phase1.sh`

**Done when** one command reproduces training and the results table.

### Phase 2 — realistic degradations

- [ ] Degradation pipeline from config: blur (isotropic and anisotropic Gaussian, motion,
      defocus), random interpolation with and without antialias, noise (Gaussian, Poisson, colour
      and grey), JPEG; optionally H.264/H.265, second-order degradation (Real-ESRGAN style),
      grayscale, colour shift
- [ ] Frozen synthetic test with 3 levels (light, medium, heavy) and stored parameters
- [ ] Real-world test set; the source is chosen with the user
- [ ] No-reference metrics (NIQE, FID); library licenses verified
- [ ] Re-train the baseline; compare clean vs realistic training on the real-world set

**Done when** the effect of the degradations on real photos is measured and documented.

### Phase 3 — model quality

- [ ] SRResNet-v2: no BN, residual scaling, more blocks, L1 loss
- [ ] ESRGAN recipe: L1 pretraining → GAN fine-tuning, VGG perceptual loss + adversarial loss with
      a small weight, EMA of weights
- [ ] Identity loss (ArcFace; D2, D11)
- [ ] Prior-guided model (FSRNet-like: coarse SR → prior → fine SR). Landmark heatmaps come from
      the CelebA 5 points; parsing maps would need CelebAMask-HQ (non-commercial, separate
      decision); facial component loss
- [ ] NME metric (landmark model; license verified)
- [ ] Subgroup metrics (`Male`, `Young`; skin tone per D10); quality-gate thresholds fixed after
      Phase 1

**Done when** the new model passes the quality gates in `CLAUDE.md` §10.

### Phase 4 — inference pipeline

- [ ] Face detection + 5 points. Candidates: YuNet (OpenCV Zoo), MediaPipe,
      RetinaFace/InsightFace — weight licenses verified. Detection runs on a bicubic upscale;
      manual bbox as a fallback
- [ ] Align → restore → inverse warp → feathered-ellipse mask → paste-back; optional colour
      correction and background upscale
- [ ] `facehall enhance` for photos, folders and video (per frame; check flicker between frames)
- [ ] Output marking: PNG text / XMP (`AI-enhanced`, model, version, parameters, input sha256);
      optional visible watermark; consider C2PA

**Done when** an arbitrary photo with several faces is processed end to end.

### Phase 5 — product

- [ ] ONNX export + parity test (max abs diff ≤ 1e-3); ONNX Runtime in the pipeline
- [ ] FastAPI `/enhance`, `/health`, `/version`: size limits, timeouts, input validation; uploaded
      images are not stored, and logs contain no images or faces
- [ ] Docker CPU and GPU images for local use, not published (D13; install Docker first); latency
      and memory benchmark
- [ ] Gradio demo (optional)

**Done when** `docker run` + `curl` returns an enhanced image and the latency is documented.

### Phase 6 — release

- [ ] `MODEL_CARD.md`, `DATA.md`, `THIRD_PARTY_LICENSES.md`, README with real results
- [ ] Versioned weights with checksums. Publishing them (GitHub Releases, Hugging Face Hub) needs
      a separate confirmation (D7, D13). Semver for the package and the models
- [ ] `CHANGELOG.md`

## 11. Open questions, risks, tech debt

### Open questions

Ask the user before the PR that needs the answer. Closed: Q1 (manifest in git) by D14, Q4
(`Blurry` images) by D15, Q5 (padding) by D16.

| # | Question | Needed by | Proposal |
|---|---|---|---|
| Q2 | Comparison grids with CelebA faces in the public `docs/RESULTS.md` | PR 1.7 | Write the grids to `outputs/` (gitignored); public docs show numbers, or images with a permissive license |
| Q3 | README links the legacy dataset on GDrive (CelebA images + Internet images of unknown origin); `images/*.png` may also show dataset faces | before Phase 6 | Remove the link or restrict access to it; check the figures |

### Risks

- ×8 from 16 px means ≈4 px between the eyes: the output is mostly invented, and ID similarity
  will be low. This must be stated clearly (`CLAUDE.md` §11).
- JPEG artifacts in the HR targets (§6.4) limit perceptual quality.
- CI time: about 40 s with the CPU build of torch (PR #4); watch it as the tests grow.
- DataLoader workers use `fork`, the Linux default. Python 3.12 warns that forking a
  multi-threaded process may deadlock (seen in the tests). The one deadlock found so far is the
  OpenCV one (§9), and the trainer avoids it. Python 3.14 changes the default to `forkserver`;
  revisit this when moving to it.
- The mirror has no checksum from the original CelebA release. Integrity rests on the row and
  identity counts. The sha256 of every shard is recorded in `docs/DATA.md` §2, so later changes to
  the local copy are detectable.

### Tech debt

- *Needs verification:* whether rows inside each split follow the official file order
  (`docs/DATA.md` §4.4). This is needed before joining annotations keyed by file name (D10).
- The legacy weights are still in HEAD (D7).
- Training cannot resume from `last`: the optimizer state is not saved. Add this before runs
  long enough for interruptions to matter (Phase 3).

## 12. Commands

Working now:

```bash
export PATH="$HOME/.local/bin:$PATH"   # if uv is not on PATH
uv sync                                # dependencies; PyTorch from PyPI (CUDA 13.2 on Linux)
uv sync --extra cpu                    # CPU-only PyTorch; then `uv run --extra cpu ...`
uv run pytest                          # tests (CPU, no dataset)
uv run pre-commit run --all-files      # lint, format, type check
uv run facehall --version
uv export --no-dev --no-hashes --no-emit-project --extra cuda --format requirements.txt \
    | uv run python scripts/license_report.py   # license table for THIRD_PARTY_LICENSES.md
uv run python scripts/prepare_dataset.py --src /mnt/storage/AI/Face/celeba_eurecom/data \
    --out /mnt/storage/AI/Face/celeba          # CelebA files + manifest (docs/DATA.md §6)
uv run facehall train --config configs/train/fnet_x4.yaml --smoke   # pipeline check, seconds
uv run facehall train --config configs/train/fnet_x4.yaml           # full run (confirm first)
uv run facehall train --config configs/train/fnet_x4.yaml --set train.batch_size=32 --set seed=1
FACEHALL_DATA=/other/root uv run facehall train --config ...        # data root on another machine
```

Planned (target interface):

```bash
uv run facehall eval --ckpt runs/<id>/best.safetensors --split test
uv run facehall enhance photo.jpg -o out.png --scale 4
uv run facehall export onnx --ckpt runs/<id>/best.safetensors --out models/x4.onnx
uv run facehall serve --port 8000
```
