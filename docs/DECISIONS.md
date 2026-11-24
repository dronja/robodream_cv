# Decisions

Log of project decisions: date, decision, reason. Open questions are marked **OPEN**.

## 2026-10-24

### D1 — Scenario and audience
**Decision:** offline restoration of photos and video frames.
**Reason:** default from the project plan; matches the README use cases and keeps the pipeline simple.

### D2 — Commercial vs. non-commercial use
**Decision:** non-commercial / research use.
**Reason:** the existing data (CelebA, FFHQ) and candidate models (ArcFace weights from InsightFace) are non-commercial only.
**Follow-ups:**
- Images from "Internet sources" in the legacy dataset have unknown origin and consent. They are excluded from the test set and documented in `docs/DATA.md`.
- If the use case changes to commercial, CelebA, FFHQ and NC-licensed models must be replaced (see `docs/THIRD_PARTY_LICENSES.md`).

### D3 — Framework
**Decision:** PyTorch. Keras code is kept in `legacy/` for reference only.
**Reason:** one framework, easier ONNX export, the legacy SRGAN generator is already PyTorch.

### D4 — Scales and resolution
**Decision:** x4 and x8 on aligned 128x128 crops; 512 px later.
**Reason:** 96 px targets are too small for a current product; 512 is deferred until the 128 px pipeline works.
**OPEN:** the legacy dataset is 89x109 px, so a 128 px HR target built from it would be an upscale, which violates the "HR must not be an upscale" rule. An HR source with resolution >= 128 px (and a compatible license) is needed before Phase 1, or the target must be reduced for the legacy data.
**Closed 2026-11-23:** see "D4 (update)" below.

### D5 — Deployment
**Decision:** CLI + REST API (FastAPI) in Docker (CPU and GPU images).
**Reason:** default from the project plan.

### D6 — Training location
**Decision:** local training. Configs target 8 GB VRAM, with an optional larger profile.
**Reason:** the project plan assumes a laptop GPU (RTX 4070 Laptop, 8 GB); the current dev machine has an RTX 6000 Ada (48 GB), so the 8 GB profile stays the portable baseline.

### D7 — Weights in git
**Decision:** do not rewrite git history. Legacy files move to `legacy/` with `git mv`; weights are removed from HEAD and published in GitHub Releases only after separate explicit confirmation.
**Reason:** rewriting history is risky and unnecessary; removing weights before they are published elsewhere could lose the only convenient copy.

### Tooling
**Decision:** package name `facehall`; `uv` for environments (installed to `~/.local/bin`, uv 0.13.5); `requires-python >=3.11`, development on Python 3.12.
**Reason:** Python 3.14 (system default) is risky for the PyTorch ecosystem.

## 2026-11-23

### D4 (update) — HR source for 128 px
**Decision:** HR 128x128 crops are built from CelebA aligned (178x218): x4 = 32->128, x8 = 16->128. The OPEN item from 2026-10-24 is closed.
**Reason:** an FFHQ-style crop from the mean CelebA landmarks is about 154 px and fits inside the image, so 128 px is a x0.83 downscale, not an upscale (`docs/PROJECT.md` §6.3).

### D8 — Phase 1 dataset and split
**Decision:** CelebA aligned from the Hugging Face mirror `eurecom-ds/celeba`, with the official train/validation/test partition. The test split is frozen.
**Reason:** the partition was verified to be identity-disjoint (8192 / 985 / 1000 identities, zero overlap), so a custom identity split is not needed, and the official partition keeps results comparable with the literature.

### D9 — Legacy dataset
**Decision:** the legacy 89x109 dataset (including the "FFHQ 2" and Internet images) is not used in Phase 1.
**Reason:** its composition, origin and consent are unknown, and 89x109 px is too small for a 128 px HR target.

### D10 — Skin-tone subgroup
**Decision:** look for an additional skin-tone annotation source; its license is verified and approved by the user before use. Until such a source exists, the missing skin-tone breakdown is documented as a limitation (model card).
**Reason:** CelebA has no skin-tone annotation; `Pale_Skin` is a binary attribute (4.2% of the test split), not a skin-tone scale.

### D11 — Approved downloads (approved 2026-10-24, recorded now)
**Decision:** downloading PyTorch, ArcFace weights for the ID-similarity metric and LPIPS weights is approved. The exact ArcFace model is chosen in Phase 1 and recorded in `docs/THIRD_PARTY_LICENSES.md` before merge.
**Reason:** required for the Phase 1 metrics; the non-commercial terms are compatible with D2.

### D12 — Documentation layout
**Decision:** `docs/PROJECT.md` is the single source of truth for facts, status, environment, data and the plan. `CLAUDE.md` keeps the rules and requirements and imports `docs/PROJECT.md`; this file keeps the decisions with reasons.
**Reason:** user request: one place to read the project state instead of re-exploring the code in every session.

### Tooling (update) — Python 3.12+
**Decision:** `requires-python >=3.12` (was `>=3.11`).
**Reason:** numpy 2.5 requires Python 3.12, and Python 3.11 was never tested: development and CI run on 3.12.

### PyTorch builds
**Decision:** by default torch comes from PyPI (CUDA 13.2 on Linux). The `cpu` extra selects the CPU-only build from the PyTorch index (CI, the CPU Docker image). The `cpu` and `cuda` extras are declared as conflicting, so one `uv.lock` holds both builds.
**Reason:** CI does not need to download the CUDA libraries (the CUDA `.venv` is about 5.9 GB, the CPU one about 1.5 GB), and local development keeps the GPU build without extra flags.

### MLflow deferred to PR 1.6
**Decision:** MLflow is not added with the other ML dependencies in PR 1.1; the experiment-tracking backend is chosen together with the trainer in PR 1.6.
**Reason:** MLflow pulls in about 70 more packages (flask, sqlalchemy, docker, databricks-sdk, scikit-learn and others), and nothing uses it before the trainer exists.

## 2026-11-24

### D13 — Local use only
**Decision:** the product is used locally only. There is no production deployment, public service or published Docker image.
**Reason:** user statement, 2026-11-24.
**Consequences:**
- The Docker images (D5) are built and run locally and are not pushed to a public registry. The redistribution terms of the NVIDIA CUDA libraries and of FFmpeg (LGPL-2.1, bundled in the OpenCV wheel) apply only to distribution; review them if this decision changes.
- The repository stays public, so the dataset terms still keep CelebA images, annotations and manifests out of it (D14; Q2 and Q3 in `docs/PROJECT.md` §11 are unchanged).
- Publishing trained weights stays subject to D7 (separate confirmation) and to the CelebA terms on derived data.
- The responsible-use requirements (`CLAUDE.md` §11) still apply.

### D14 — Manifest outside git (closes Q1)
**Decision:** the manifest (`manifest.parquet`) and `prepare_info.json` stay next to the prepared data, outside the repository. The repository holds the code that generates them, plus the counts and checksums in `docs/DATA.md`.
**Reason:** the manifest carries CelebA annotations (identities, landmarks, attributes), and the CelebA terms forbid distributing any portion of the dataset. The manifest content checksum lets anyone with the data check that they rebuilt the same manifest. The user accepted the proposal on 2026-11-24 by asking to continue with it.

### Image files keep the mirror's encoding
**Decision:** `scripts/prepare_dataset.py` writes the stored image bytes without re-encoding: train images as PNG, validation and test images as JPEG, as they are stored in the mirror.
**Reason:** in the mirror, train images are PNG files of decoded JPEG pixels (`docs/DATA.md` §4.2), while validation and test images are the original JPEG files. Re-encoding the train images as JPEG would add a second lossy generation. The cost is disk space: the train image files take 8.63 GB, against about 1.1 GB at the mean size of the validation and test JPEG files.

### Alignment template and conventions (PR 1.3)
**Decision:**
- The canonical 5-point template is the FFHQ alignment rule (reduced to 5 points), applied to the CelebA train-split mean landmarks and then averaged with its mirror image.
- Per image, a least-squares similarity (Umeyama) over all 5 points maps the landmarks to the template.
- Coordinates are pixel-centre coordinates; the template uses normalized crop coordinates.
- Warp defaults: bicubic interpolation and reflect border; both are configurable.

**Reason:**
- Using the train split keeps the test split out of design decisions.
- The mirror-averaged template makes horizontal flips commute with alignment, and it moves no point by more than 0.34 px at 128.
- Umeyama on 5 points uses the nose and every mouth corner rather than only eye and mouth midpoints, and needs no special cases.
- Normalized coordinates give the same face region at 32, 128 and 512 px.
- Reflect padding follows FFHQ; the padding policy for training and evaluation is open (Q5).

### D15 — `Blurry` images (closes Q4)
**Decision:** images with `Blurry = 1` are dropped from train (`exclude_blurry: [train]` in `configs/data/celeba_128.yaml`). Validation and test keep them; evaluation also reports metrics on the non-blurry subset (every sample carries a `blurry` flag).
**Reason:** a blurry HR target teaches the model to produce blur; keeping val and test complete keeps the frozen test close to the official one. The user accepted the proposal on 2026-11-24.

### D16 — Crops that reach outside the image (closes Q5)
**Decision:** crops are filled with reflect padding in every split. No crop is dropped for padding; every sample carries the fraction of padded HR pixels (`padded`), and evaluation also reports metrics on the unpadded subset.
**Reason:** reflect padding follows FFHQ and affects 2.4–2.8% of crops (`docs/PROJECT.md` §6.3); keeping the crops keeps the test set complete. The user accepted the proposal on 2026-11-24.

### Test set: one image removed by the upscale rule
**Decision:** the dataset drops every sample whose HR crop would be an upscale of the source (`CLAUDE.md` §7). In the current data this removes exactly one image, `img_align/test/001557.jpg`, whose landmarks are broken (`docs/DATA.md` §4.5). The frozen test set is therefore the official test split minus this image: 19,961 images.
**Reason:** the rule is a project requirement, and the crop of this image is a ×3.3 zoom into the eye region, not a face. The user did not object when asked on 2026-11-24.

### Dataset v0 settings (PR 1.4)
**Decision:**
- Degradation v0 is `torch.nn.functional.interpolate(mode="bicubic", antialias=True, align_corners=False)`, clamped to [0, 1] and quantized to 8-bit levels. It matches PIL's `BICUBIC` resize within 1 8-bit level.
- Train augmentation: horizontal flip with p = 0.5; jitter of up to 3% shift, 5° rotation and a ×1.05 scale. These are initial values, to be tuned.
- Validation and test crops are computed on the fly, deterministically, instead of being cached as planned.
**Reason:**
- Bicubic with antialias is the standard "BI" degradation of the SR literature, and the torch function keeps it on the tensor path. Without antialias the result differs from PIL by up to 142 levels (aliasing).
- The flip is valid because the template is symmetric. The jitter models imprecise landmarks on small faces (`CLAUDE.md` §6).
- Decoding and warping take about 0.7 ms per image, and a DataLoader with 8 workers delivers about 7,700 samples/s. Caching would add invalidation logic for no measurable gain; it can be added if profiling in PR 1.6 shows a need.

### Legacy model ports (PR 1.5)
**Decision:**
- FNet and CNet are ported layer for layer from `legacy/FH.ipynb`.
- The ports keep the Keras default initialisation and BatchNorm settings: Glorot-uniform weights, zero bias, BN epsilon 1e-3 and momentum 0.99, PReLU alpha initialised to 0.
- CNet keeps its per-pixel PReLU, so it is built for one LR size.
- The only generalisation is the scale: one ×2 stage per factor of 2.

**Reason:**
- Phase 1 measures the legacy architectures on the new protocol, so the ports must not quietly improve them.
- Initialisation and BN settings change training behaviour, so they are part of the recipe.
- The trainable parameter counts equal the notebook's `model.summary()` at 24 px ×4 (FNet 734,723; CNet 854,979).
- The per-pixel PReLU is a legacy quirk; SRResNet-v2 in Phase 3 drops it together with BN.

### D17 — Experiment tracking: one plain run directory per run
**Decision:** every run writes `runs/<name>-<timestamp>[-smoke]/`, containing:
- the resolved config;
- `run.json` with provenance (command, git commit, library versions, device, dataset settings and counts, manifest checksum);
- `metrics.jsonl`, the text log and `summary.json`;
- the checkpoints.

MLflow is not added. It can be added later as an option that reads the same records.

**Reason:**
- The user chose this option on 2026-11-24.
- It needs no new dependencies (MLflow pulls in about 70 packages).
- `RESULTS.md` (PR 1.7) can be generated directly from these files.
- The product is local only (D13), so a tracking server adds little.

This replaces the MLflow default in `CLAUDE.md` §6.

### Trainer (PR 1.6): differences from the legacy recipe
**Decision:** the configs follow the legacy recipe (MSE, Adam with Keras defaults, batch 16; FNet 10 epochs with patience 3, CNet 50 epochs), with these differences:
- the forward pass uses bf16 autocast (legacy: fp32). The loss and validation are in fp32. `--set train.precision=fp32` restores fp32;
- the train order is shuffled with a permutation per `(seed, epoch)` (legacy: no shuffle, `CLAUDE.md` §4.3);
- the last incomplete train batch is dropped;
- the `best` checkpoint (lowest val loss) replaces Keras `restore_best_weights`;
- there is no separate 48 GB profile.

Configs are composed with a small `include` mechanism plus `--set` overrides, not with Hydra.

**Reason:**
- bf16 roughly doubles the training speed (PR 1.5 table), and both target GPUs support it.
- No shuffling was a known legacy bug, not part of the recipe.
- Dropping the last batch keeps every optimizer step the same size, including with gradient accumulation.
- Every config fits 8 GB at batch 16, so a larger batch would change the recipe rather than adapt it to the hardware.
- `include` keeps the data, degradation and model configs reusable without adding a dependency.

### D18 — ID-similarity model: InsightFace `buffalo_l` recognition model
**Decision:** the ID-similarity metric uses `w600k_r50.onnx` from the InsightFace `buffalo_l` pack (ArcFace, ResNet50 trained on WebFace600K). It runs through ONNX Runtime (MIT) on CPU and is used for non-commercial research only (D2).
- Download: `scripts/download_weights.py buffalo_l`, with a pinned size and sha256.
- Only the recognition model is kept from the pack.
- Terms and checksums: `docs/THIRD_PARTY_LICENSES.md` §3.

**Open point:** the InsightFace README says these models are for non-commercial research only. Its "2025-11-24 Update" also asks users to contact recognition-oss-pack@insightface.ai "for licensing" of `buffalo_l`, without saying whether research use needs that separate license. The user may ask InsightFace; revisit this decision if they answer.

**Reason:**
- ArcFace embeddings are the usual identity metric in face SR papers, so our numbers stay comparable with them.
- The alternatives checked on 2026-11-24 do not state a weight license either:
  - facenet-pytorch (FaceNet trained on VGGFace2);
  - AdaFace (weights on Google Drive only).
- The user chose this option on 2026-11-24.

### D19 — Comparison grids with CelebA faces stay local (closes Q2)
**Decision:** grids and sample crops are written only to `outputs/` (gitignored). `docs/RESULTS.md` contains numbers only.
**Reason:** the CelebA terms forbid publishing or distributing any portion of the dataset (`docs/THIRD_PARTY_LICENSES.md` §2), and SR outputs of CelebA faces are derived data. The user accepted the proposal on 2026-11-24.

### Evaluation protocol (PR 1.7)
**Decision:**
- **Protocol and checkpoint:** a protocol (`configs/eval/celeba128_x{4,8}.yaml`) fixes the split, data, degradation, seed and metrics. A checkpoint contributes only the model, and its scale and HR size must match. Phase 1 protocols use the frozen test split.
- **SR output:** SR is rounded to 8-bit levels before scoring, as a saved image would be.
- **PSNR and SSIM:** on the BT.601 luma Y in [16, 235], with peak 255 and no border crop. SSIM uses torchmetrics (Gaussian window 11, σ 1.5). PSNR is capped at about 148 dB for identical images (MSE floor 1e-10).
- **LPIPS:** the AlexNet variant (the torchmetrics default) on RGB.
- **ID similarity:**
  - the cosine of ArcFace embeddings of SR and HR;
  - both crops are mapped to the ArcFace 112 px template with one fixed similarity derived from our template (bilinear, black border, as InsightFace `norm_crop`), so no face detector is involved;
  - preprocessing is RGB and (x − 127.5) / 127.5, as InsightFace selects for this model;
  - HR embeddings are cached per protocol and split, with a fingerprint of everything they depend on.
- **Subsets:**
  - all;
  - `Blurry` = 0 (D15);
  - no padding (D16);
  - the CelebA attributes `Male` and `Young` (= 1 / = 0).
  - There is no skin-tone subgroup yet (D10).
- **Outputs and report:** results go to `outputs/eval/<protocol>/<split>/<model>/` (`metrics.json`, `per_image.parquet`). `docs/RESULTS.md` is generated from these files only.

**Reason:**
- Y-channel PSNR and SSIM without border handling tricks are the SR-literature convention for comparisons.
- Quantizing SR measures what a user would actually get.
- The fixed recognition alignment is exact for our aligned crops, and it avoids detector failures on SR outputs.
- Per-image results allow any later breakdown without re-running.
- Sanity check on 2026-11-24 (60 val identities, two images each): same-identity HR pairs scored a mean cosine of 0.565, different-identity pairs a mean of 0.015 (max 0.143), and identical crops 1.0. This confirms the alignment and preprocessing.

### Degradation pipeline (PR 2.1)
**Decision:**
- **Pipeline:** degradations are drawn per sample from a config (`kind: pipeline`). The order is:
  1. optional grayscale;
  2. stage 1: blur → resize → noise → JPEG;
  3. an optional second stage in the Real-ESRGAN style (arXiv:2107.10833), so the first stage ends at an intermediate size;
  4. an optional colour shift;
  5. 8-bit rounding.
- **Blur units:** sizes are in LR pixels and are scaled to each stage's working resolution.
- **Grayscale:** turns the HR target grey as well.
- **Colour shift:** shifts the LR only, and is implemented but off in `configs/degrade/realistic.yaml`.
- **Parameters:** the drawn parameters of every sample are returned as JSON.
- **Initial ranges** for `realistic.yaml`:

  | Stage | Blur | Noise | JPEG |
  |---|---|---|---|
  | First | p 0.9; σ 0.1–1.0 LR px | p 0.7 | p 0.8, quality 30–95 |
  | Second (p 0.5) | p 0.5; σ 0.1–0.6 | p 0.5 | p 0.8 |

  - First-stage blur also includes motion 0.5–3 px and defocus 0.3–1.5 px. Its noise is Gaussian σ 1–15 levels or Poisson with 100–3000 photons at white, grey with p 0.4.
  - Resize mix: bicubic 0.4, bilinear 0.3, area 0.2, nearest 0.1; antialias p 0.7.
  - Grayscale p 0.02.
- **Not implemented:** H.264/H.265 (optional) and the sinc filter.

**Reason:**
- **The order follows image formation:** optics and motion blur before sampling, sensor noise after it, then compression. The second stage models images that were resized and recompressed again (`CLAUDE.md` §7).
- **LR-pixel units:** real blur is set by the camera, in pixels of the captured image, whatever the size of the face. They also let one config serve ×4 and ×8, whereas HR-pixel units would make ×8 inputs twice as sharp relative to their pixels.
- **Grey target:** for a grey input the model must not invent colours. GFPGAN keeps colour targets by default, which trains colourisation, and offers `gt_gray` as an option.
- **Colour shift off:** an LR-only colour shift trains the model to replace colours with typical ones, which is invention. It stays available for experiments.
- **Initial ranges:**
  - They take the literature as a starting point. Training configs checked on 2026-11-24:
    - Real-ESRGAN `options/train_realesrgan_x4plus.yml`:
      - `blur_sigma: [0.2, 3]` in HR px at ×4 (about 0.05–0.75 LR px), `blur_sigma2: [0.2, 1.5]`;
      - `noise_range: [1, 30]`, `noise_range2: [1, 25]`;
      - `gray_noise_prob: 0.4`, `jpeg_range: [30, 95]`.
    - GFPGAN `options/train_gfpgan_v1.yml`, 512 px faces:
      - `blur_sigma: [0.1, 10]` before `downsample_range: [0.8, 8]`, so the blur reaches several LR px;
      - `noise_range: [0, 20]`, `jpeg_range: [60, 100]`;
      - `gray_prob: 0.01`, `color_jitter_prob: 0.3`.
  - Our JPEG range and grey-noise probability follow Real-ESRGAN. The Gaussian noise range (1–15 levels) is half of Real-ESRGAN's, as a lighter start for faces of 16–32 px.
  - In a visual check on 8 val crops, an upper σ of 1.2 LR px made most samples very blurry, so it was lowered to 1.0.
  - They are tuned in PR 2.4 against the real-world test set, so they are a default, not a finding.
- **Parameters per sample:** so validation and test results can be broken down by degradation (PR 2.2).

### Synthetic test levels and frozen inputs (PR 2.2)
**Decision:**
- **Levels:** the synthetic test has three degradation levels, light, medium and heavy (`configs/degrade/levels/`).
  - Every sample gets blur, a resize, noise and JPEG.
  - The ranges grow from level to level without overlapping, and stay no stronger than the training mix `realistic`.
  - Heavy adds a second resize and compression stage.
  - The resize mix is the training mix in every level, and there is no grayscale.
  - The ranges are in `docs/PROJECT.md` §9.
- **Protocols:** each level is its own protocol per scale (`celeba128_<level>_x{4,8}`) on the frozen test split with seed 0. The two Phase 1 bicubic protocols get `level: bicubic`.
- **Frozen inputs:**
  - every protocol lists the sha256 of all its LR and HR crops (`inputs_sha256`);
  - an evaluation whose inputs differ fails;
  - refreezing is a deliberate change that requires re-evaluating every model.
- **Per image:** the drawn degradation parameters are stored with the results.

**Reason:**
- **All four degradations at every level:** the levels then measure severity, not which degradations happen to be present. The per-image parameters still allow a breakdown by a single factor.
- **Within the training mix:** the levels test the training distribution from its mild end to its upper end. Robustness to unseen degradations is the job of the real-world test set (PR 2.3).
- **Frozen inputs:**
  - The protocol config and seed define the inputs only together with the library versions. JPEG encoding (OpenCV/libjpeg), resampling (torch) and random streams (numpy) may change between versions.
  - A digest turns such a change into an error instead of silently incomparable numbers.
  - A digest is cheap: about 4 s per protocol, and it is computed during evaluation anyway.
- **Bicubic protocols frozen too:** the PR 2.1 pipeline left `kind: bicubic` unchanged, so re-evaluating the Phase 1 models from this commit must reproduce the Phase 1 numbers exactly. It did: the tables were identical on 2026-11-24.

### Realistic training, synthetic comparison (PR 2.4)
**Decision:**
- **What was trained:** FNet ×4 and ×8 were re-trained on `configs/degrade/realistic.yaml` with the unchanged Phase 1 recipe, so clean and realistic training differ in one setting only. CNet was not re-trained.
- **Order:** PR 2.4 came before PR 2.3, at the user's choice on 2026-11-24, and covers only the synthetic levels. The real-world comparison and the tuning of the realistic ranges follow PR 2.3.
- **No quality gate yet:** none is set for this step. Its numbers are a reference for Phase 2 and Phase 3.

**Reason:**
- **One variable:** keeping the recipe isolates the effect of the degradations.
- **FNet only:** FNet was the best Phase 1 baseline. CNet takes about 4× longer to train.
- **Waiting for real photos:** the synthetic levels come from our own degradation model, so tuning the ranges on them would be circular. Real low-quality photos are needed for that.
- **Results** (`docs/PROJECT.md` §9):
  - realistic training improves PSNR and LPIPS on degraded inputs;
  - it barely changes ID similarity, which at ×4 stays below plain bicubic upsampling;
  - it loses a lot on clean inputs.
  - The next steps are therefore range tuning (after PR 2.3) and identity-aware losses (Phase 3).
