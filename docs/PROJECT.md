# facehall — project overview

> **Single source of truth** for facts, current state, environment, data and the work plan.
> Last updated: **2026-11-23**.
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
| Dataset composition, how to reproduce it | `docs/DATA.md` | *planned, PR 1.2* |
| Third-party licenses | `docs/THIRD_PARTY_LICENSES.md` | *planned, PR 1.1* |
| Metric tables (generated) | `docs/RESULTS.md` | *planned, PR 1.7* |
| Model card | `docs/MODEL_CARD.md` | *planned, Phases 3 and 6* |

## 2. Status snapshot

As of 2026-11-23:

- **Phase 0 (repository hygiene): in review.** Branch `chore/phase-0-repo-hygiene` → PR into
  `main`. Local checks pass: ruff, ruff format, mypy (strict), pytest (8 tests). Phase 0 is done
  when CI is green on the PR and the PR is merged.
- `origin/main` is at `6c75963` — the legacy state before any refactoring.
- **Phase 1: not started.** Data is downloaded and verified (§6.2); the technical design is
  agreed (§9).
- Legacy weights are still tracked in `legacy/`; uploading them to GitHub Releases needs a separate
  confirmation (D7).

## 3. Repository contents

```
robodream_cv/
├── CLAUDE.md                  # rules for Claude Code (Ukrainian); imports this file
├── README.md                  # public description (English)
├── LICENSE
├── pyproject.toml, uv.lock    # uv project, hatchling build; ruff, mypy, pytest config
├── .python-version            # 3.12
├── .pre-commit-config.yaml    # whitespace, yaml, toml, large files; ruff and mypy via `uv run`
├── .github/workflows/ci.yml   # uv sync --locked → ruff check, ruff format --check → mypy → pytest
├── src/facehall/              # __init__.py (version 0.0.0), cli.py (only --version), py.typed
├── tests/                     # test_import.py, test_cli.py, test_legacy_layout.py
├── docs/                      # PROJECT.md, DECISIONS.md
├── images/                    # figures used by README: CNET, FNET, PULSE, SRgan (.png)
└── legacy/                    # read only: FH.ipynb, FNet_24.keras, CNet_24.keras, SRgan_20.pth
```

- Package `facehall` 0.0.0; runtime dependencies: `numpy>=1.26`, `pillow>=10.0`;
  dev group: ruff, mypy, pytest, pre-commit. Entry point: `facehall = facehall.cli:main`.
- ruff: line length 100, rules `E F I UP B SIM`, `legacy/` excluded. mypy: strict on `src` and
  `tests`.
- CI runs on pushes to `main` and on every pull request.
- The target layout (configs/, data/, align/, models/, …) is described in `CLAUDE.md` §6.

## 4. Development environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada, 48 GB (driver 595.99, CUDA 13.2) |
| Portable target | RTX 4070 Laptop, 8 GB — every training config needs a profile that fits 8 GB (D6) |
| Python | 3.12.14 via uv for the project; the system Python 3.14 is not used |
| uv | 0.13.5 in `~/.local/bin` (add it to `PATH` in non-login shells) |
| Locked dev tools | ruff 0.17.2, mypy 2.4.0, pytest 9.1.1 |
| GitHub CLI | gh 2.46.0, authenticated (git protocol: ssh) |
| Docker | not installed (needed in Phase 5) |
| Data root | `/mnt/storage/AI/Face` (~18 TB free); the repo `data/` directory is gitignored |
| Remote | `https://github.com/dronja/robodream_cv` (public), default branch `main` |

## 5. Legacy research code

`legacy/` is read only. Known problems of this code, and what must not be carried over:
`CLAUDE.md` §4.

| File | What it is | Notes |
|---|---|---|
| `FH.ipynb` (456 KB) | All code: TF/Keras FNet and CNet, PyTorch SRGAN generator, training, visualisation | Cells were run out of order; SRGAN training (discriminator, loop) is missing |
| `FNet_24.keras` (9.0 MB) | "Simplified SRResNet": 24×24 → 96×96 (×4), 5 residual blocks with BN, PixelShuffle via `Lambda(depth_to_space)`, sigmoid, MSE | Best legacy result. Keras 3.9.2, contains optimizer state, loading needs `safe_mode=False` |
| `CNet_24.keras` (10.3 MB) | "Simplified C-Face": Conv + `UpSampling2D`, no residual blocks, MSE | Slightly worse than FNet |
| `SRgan_20.pth` (3.0 MB) | PyTorch SRGAN generator `state_dict` | Architecture equals FNet; the notebook loads `best_generator_20.pth` (name mismatch) |

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
| CelebA aligned, HF mirror `eurecom-ds/celeba` | Training and evaluation, Phases 1–3 (D8) | Downloaded and verified |
| Legacy 89×109 dataset | — | Excluded (D9) |
| FFHQ | Candidate HR source for 512 px (later) | Not downloaded; non-commercial |
| Real-world low-quality test set | Phase 2 | Source not chosen yet: needs known origin and license or consent |

### 6.2 CelebA mirror — verified facts

Verified on 2026-11-23 by reading the parquet files directly.

Location: `/mnt/storage/AI/Face/celeba_eurecom/data/` — 20 parquet shards (train 18, validation 1,
test 1), 8.4 GB. According to the dataset card, the mirror was produced from
`torchvision.datasets.CelebA`.

| Split | Images | Identities |
|---|---|---|
| train | 162,770 | 8,192 |
| validation | 19,867 | 985 |
| test | 19,962 | 1,000 |
| **total** | **202,599** | **10,177** |

The identity overlap between any two splits is **0**: the official partition is identity-disjoint.

Schema: `image` (struct `bytes`, `path`), `attributes` int8[40], `identity` int64, `bbox` int32[4],
`landmarks` int32[10].

| Field | Usable | Evidence |
|---|---|---|
| `image` | yes | `img_align_celeba`: a sample of 200 images is all JPEG 178×218, mean size 7.0 KB. `path` is empty, so original file names are not stored; the row order probably follows the official file order (*needs verification*) |
| `landmarks` | yes | Aligned-image coordinates `[lx, ly, rx, ry, nose_x, nose_y, lmouth_x, lmouth_y, rmouth_x, rmouth_y]`. Test split means: eyes (69.3, 111.2) and (107.7, 111.2), nose (88.0, 135.1), mouth (71.2, 152.1) and (105.7, 152.1). Interocular distance: median 39 px, p5 33, p95 43 |
| `bbox` | **no** | Coordinates of the in-the-wild images, not the aligned ones: max x+w = 4628; 88.5% of test boxes fall outside 178×218 |
| `attributes` | partly | 40 binary CelebA attributes, values {0, 1}. Test split: `Male` 38.6%, `Young` 75.7%, `Pale_Skin` 4.2% (indices 20, 39, 26 in the CelebA attribute order). `Blurry` (index 10) is also available |

### 6.3 Crop geometry: 128 px HR is a downscale

The FFHQ-style alignment (a square built from the eye-to-eye and eye-to-mouth vectors), applied to
the mean test landmarks, gives a crop of side ≈154 px with x ∈ [11.7, 165.3] and y ∈ [38.5, 192.1].
The crop lies fully inside 178×218. Resizing it to 128 px is a ×0.83 downscale, so an HR 128×128
target does not violate the rule "HR must not be an upscale" (D4).

| Scale | LR | HR | Interocular distance |
|---|---|---|---|
| ×4 | 32×32 | 128×128 | ≈8 px in LR, ≈32 px in HR |
| ×8 | 16×16 | 128×128 | ≈4 px in LR, ≈32 px in HR |

Faces with a large interocular distance (p95 = 43 px gives a crop side of ≈172 px) or faces offset
from the mean can touch the image border. Such crops need padding, and the padded area should be
recorded. The exact template is fixed in PR 1.3.

### 6.4 Known data limitations

- HR targets are JPEG (≈7 KB per image), so compression artifacts are part of the "ground truth".
  This is acceptable for 128 px but not for 512 px.
- Images with `Blurry = 1` are poor HR targets; whether to exclude them is open (Q4).
- There is no skin-tone annotation: `Pale_Skin` is a binary attribute, not a skin-tone scale.
  The plan is in D10.
- Celebrity photos are not representative of real-world low-quality inputs; the Phase 2
  real-world test set addresses this.
- CelebA terms: non-commercial research only, no redistribution. The mirror's own license
  *needs verification*. Consequence: no CelebA images or annotations go into the public repository
  (Q1, Q2).

### 6.5 Planned layout and manifest (PR 1.2)

Proposal; final layout recorded in `docs/DATA.md`:

```
/mnt/storage/AI/Face/
├── celeba_eurecom/                       # raw download, kept as is, read only
└── celeba/                               # produced by scripts/prepare_dataset.py
    ├── img_align/{split}/{index:06d}.jpg # original JPEG bytes, not re-encoded
    └── manifest.parquet
```

Manifest columns: `path, sha256, source, license, identity_id, width, height, split, landmarks`,
plus selected attributes (`male`, `young`, `blurry`). The split is the official partition; the test
split is frozen and never used for tuning. The code locates the data through a config value
(`data.root`), not through hard-coded paths.

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

## 8. Licenses — status

The rules and the known-license table are in `CLAUDE.md` §8. The final record is
`docs/THIRD_PARTY_LICENSES.md` (created in PR 1.1). These items must be verified before the PR that
uses them is merged:

| Item | Status |
|---|---|
| HF mirror `eurecom-ds/celeba` | The dataset card has no license field — *needs verification*; the original CelebA terms apply |
| ArcFace weights (model chosen in PR 1.7) | InsightFace pretrained models: non-commercial research (compatible with D2) |
| LPIPS weights and torchvision backbone weights | *needs verification* (PR 1.7) |
| Python dependencies added in PR 1.1 | *needs verification* (PR 1.1) |
| Legacy dataset on GDrive (linked from README) | Contains CelebA images and Internet images of unknown origin (Q3) |

## 9. Phase 1 technical design

The agreed approach. Details are fixed in the corresponding PRs.

- **General.** PyTorch 2.x, YAML configs with OmegaConf, no magic constants in the code. Every run
  stores its config, git hash, seed and library versions. MLflow runs locally (`mlruns/`, gitignored).
- **Dependencies (PR 1.1).** torch and torchvision use the CUDA build locally and the CPU build in
  CI (uv explicit indexes + extras with `conflicts`), so CI does not download CUDA wheels. The CUDA
  variant must match the driver (CUDA 13.2). Other dependencies: omegaconf, safetensors, pyarrow,
  opencv-python-headless (affine warps), torchmetrics[image], mlflow.
- **Data preparation (PR 1.2).** `scripts/prepare_dataset.py`: parquet → files with the original
  JPEG bytes, plus the manifest (§6.5). Row and identity counts are checked against §6.2. Writes
  `docs/DATA.md`.
- **Alignment (PR 1.3).** `facehall.align`: a canonical 5-point template for 128×128 (from the
  §6.3 geometry), a similarity transform fitted by least squares (Umeyama), a warp and an inverse
  warp (reused for paste-back in Phase 4), and configurable alignment jitter (shift, rotation,
  scale).
- **Dataset + degradation v0 (PR 1.4).** Train crops are warped on the fly from the 178×218 source,
  with jitter. Validation and test crops are deterministic and cached. LR is a bicubic downscale
  with antialias; the exact function is fixed in `configs/degrade/bicubic.yaml` and recorded.
  Validation and test use a fixed seed.
- **Models (PR 1.5).** The `bicubic` baseline, plus FNet and CNet ported 1:1 from `FH.ipynb`: the
  same layers as in the notebook, including BN and the sigmoid output, parameterised by `scale`.
- **Trainer + CLI (PR 1.6).** AMP (bf16), gradient accumulation and seeding. Checkpoints `last` and
  `best` are saved in safetensors with JSON metadata. `facehall train --config … [--smoke]`.
  Configs `configs/train/{fnet,cnet}_x{4,8}.yaml` come with an 8 GB default and a large profile.
- **Evaluation (PR 1.7).** PSNR and SSIM on the Y channel, LPIPS, and ID similarity (cosine of
  ArcFace embeddings, SR vs HR). `facehall eval`. `docs/RESULTS.md` is generated by a script. NME
  moves to Phase 3 because it needs a landmark model.
- **Reproduction.** `scripts/reproduce_phase1.sh` runs prepare → train (FNet, CNet × ×4, ×8) →
  eval → `RESULTS.md`.

## 10. Roadmap and progress

Each step is compared with the previous best on the same frozen test set. A smoke run (a few
hundred images, 1–2 epochs, under 5 min) comes before every full run. Full training (longer than
~30 min) starts only after user confirmation.

### Phase 0 — repository hygiene (*in review*)

- [x] Python `.gitignore` (`data/`, `runs/`, `mlruns/`, `outputs/`, weights and checkpoints)
- [x] Move `FH.ipynb`, `*.keras`, `*.pth` to `legacy/` (`git mv`)
- [x] `pyproject.toml` (uv), ruff, mypy, pytest, pre-commit, CI
- [x] README fixes (C-Face reference, input sizes); `docs/DECISIONS.md`
- [x] `docs/PROJECT.md` as the single source of truth; pre-commit uses the locked ruff
- [ ] PR merged with green CI
- [ ] *(not blocking)* Legacy weights uploaded to GitHub Releases with checksums, then removed from
      HEAD (D7, separate confirmation)

**Done when** `uv sync && uv run pytest` passes on a clean machine and CI is green.

### Phase 1 — package and reproducible baseline

- [ ] **PR 1.1** `build:` ML dependencies (CUDA locally, CPU in CI); `THIRD_PARTY_LICENSES.md`
- [ ] **PR 1.2** `feat(data):` `prepare_dataset.py`: files, manifest, official split, frozen test;
      `DATA.md`. Tests use a synthetic parquet fixture
- [ ] **PR 1.3** `feat(align):` template, similarity transform, warp and inverse warp, jitter.
      Tests: warp and inverse-warp round trip on synthetic points
- [ ] **PR 1.4** `feat(data):` dataset + degradation v0 (bicubic). Tests: determinism, shapes,
      value ranges
- [ ] **PR 1.5** `feat(models):` bicubic, FNet, CNet. Tests: output shapes, parameter counts
      against legacy
- [ ] **PR 1.6** `feat(train):` trainer, checkpoints, MLflow, `facehall train`. Tests: CPU smoke
      run on fixtures
- [ ] **PR 1.7** `feat(eval):` PSNR, SSIM, LPIPS, ID-sim, `facehall eval`, generated
      `RESULTS.md`. Tests: metrics on known pairs
- [ ] Smoke run on real data; estimate the duration of a full run
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
- [ ] Docker CPU and GPU images (install Docker first); latency and memory benchmark
- [ ] Gradio demo (optional)

**Done when** `docker run` + `curl` returns an enhanced image and the latency is documented.

### Phase 6 — release

- [ ] `MODEL_CARD.md`, `DATA.md`, `THIRD_PARTY_LICENSES.md`, README with real results
- [ ] Weights in GitHub Releases or on the Hugging Face Hub, with checksums; semver for the package
      and the models
- [ ] `CHANGELOG.md`

## 11. Open questions, risks, tech debt

### Open questions

Ask the user before the PR that needs the answer.

| # | Question | Needed by | Proposal |
|---|---|---|---|
| Q1 | CelebA terms forbid redistribution, and a manifest with identity ids and landmarks is part of the dataset. Should the manifest be committed? | PR 1.2 | Keep the full manifest next to the data; commit only the generator script, the counts and the manifest sha256 |
| Q2 | Comparison grids with CelebA faces in the public `docs/RESULTS.md` | PR 1.7 | Write the grids to `outputs/` (gitignored); public docs show numbers, or images with a permissive license |
| Q3 | README links the legacy dataset on GDrive (CelebA images + Internet images of unknown origin); `images/*.png` may also show dataset faces | before Phase 6 | Remove the link or restrict access to it; check the figures |
| Q4 | Exclude CelebA images with `Blurry = 1` from train and/or test? | PR 1.2 | Count them first, then decide |

### Risks

- ×8 from 16 px means ≈4 px between the eyes: the output is mostly invented, and ID similarity
  will be low. This must be stated clearly (`CLAUDE.md` §11).
- JPEG artifacts in the HR targets (§6.4) limit perceptual quality.
- CI time and size grow once torch is added; CPU wheels mitigate this.
- The mirror has no checksum from the original CelebA release. Integrity rests on the row and
  identity counts; the sha256 of every shard is recorded in PR 1.2.

### Tech debt

- `docs/DATA.md` and `docs/THIRD_PARTY_LICENSES.md` are referenced by `DECISIONS.md` but do not
  exist yet (PR 1.2 and PR 1.1).
- The legacy weights are still in HEAD (D7).

## 12. Commands

Working now:

```bash
export PATH="$HOME/.local/bin:$PATH"   # if uv is not on PATH
uv sync                                # dependencies
uv run pytest                          # tests (CPU, no dataset)
uv run pre-commit run --all-files      # lint, format, type check
uv run facehall --version
```

Planned (target interface):

```bash
uv run python scripts/prepare_dataset.py --src /mnt/storage/AI/Face/celeba_eurecom/data \
    --out /mnt/storage/AI/Face/celeba
uv run facehall train --config configs/train/fnet_x4.yaml --smoke
uv run facehall train --config configs/train/fnet_x4.yaml
uv run facehall eval --ckpt runs/<id>/best.safetensors --split test
uv run facehall enhance photo.jpg -o out.png --scale 4
uv run facehall export onnx --ckpt runs/<id>/best.safetensors --out models/x4.onnx
uv run facehall serve --port 8000
```
