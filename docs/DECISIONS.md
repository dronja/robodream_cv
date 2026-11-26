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

### D20 — Real-world test set: native low-resolution faces from WIDER FACE
**Decision:**
- **Source:** the WIDER FACE training images, aligned with the manually annotated RetinaFace landmarks. Both are non-commercial research only (`docs/THIRD_PARTY_LICENSES.md` §2).
- **Selection:** faces whose native size already matches the model input. The alignment to the 32 px (×4) or 16 px (×8) template must scale the source by 0.8–1.25, so the LR input is not synthetically resized. Faces marked `invalid` are dropped.
- **Storage:** the LR crops are prepared once and stored as PNG with a manifest (`scripts/prepare_wider.py`, `docs/DATA.md` §7).
- **Split:** by source image, 50/50 from a hash of the image name. `val` is for tuning (for example the degradation ranges), `test` is frozen.
- **Stays local:** crops, the manifest and grids showing these faces, as for CelebA (D19).
- **Not used:**
  - LFW and WIDER-Test, by the user's choice;
  - WebPhoto-Test and CelebChild-Test: photos crawled from the Internet, origin and terms unknown;
  - TinyFace: no terms stated; part of it comes from MegaFace, which is no longer distributed;
  - QMUL-SurvFace: surveillance footage, which conflicts with `CLAUDE.md` §11.

**Defaults taken without a separate question:** the native range 0.8–1.25, the 50/50 split and keeping occluded and blurred faces. Their attributes are stored, so results can be broken down by them.

**Reason:**
- **Native resolution:** the faces have the same pixel count as our input and carry real camera blur, noise and compression. They are not a synthetic downscale.
- **Clear terms:** CC BY-NC-ND fits non-commercial research (D2) and local use (D13).
- **Training images:** RetinaFace released landmarks only for the training images. Nothing in this project trains on WIDER FACE.
- **Split by image:** WIDER FACE has no identities. Splitting by image keeps the faces of one photo together. The same people may still appear in several photos of one event. That matters little, because the `val` part is used only to tune a few degradation parameters.
- **Size:** ×4: val 8,160 and test 8,645 faces. ×8: val 2,752 and test 2,172 faces (`docs/DATA.md` §7.3).
- The user chose WIDER FACE, without LFW and WIDER-Test, on 2026-11-24.

### No-reference metrics for the real-world set (PR 2.3)
**Facts checked on 2026-11-24:**
- **NIQE** cannot be computed on our crops. The reference implementation (BasicSR `niqe.py`) splits the image into 96×96 blocks and needs the covariance over blocks; a 128 px output has one block.
- **pyiqa** (NIQE and other no-reference metrics) is licensed PolyForm-Noncommercial-1.0.0 and has 27 direct requirements (transformers, accelerate, bitsandbytes, facexlib, …). It is not added.
- **FID and KID** are available through torchmetrics and torch-fidelity 0.4.0 (Apache-2.0), already in the lock file. They need the InceptionV3 weights `weights-inception-2015-12-05-6726825d.pth` from the torch-fidelity GitHub releases. The torch-fidelity source says the weights were "converted from the official TensorFlow weights", "distributed under Apache License 2.0". FID needs at least 2,048 images for a full-rank covariance (TTUR README); the ×8 test split has 2,172.

**Resolved by D21:** the user approved the download of the Inception weights on 2026-11-24.

### D21 — InceptionV3 weights for FID and KID (closes Q6)
**Decision:** the torch-fidelity InceptionV3 weights are downloaded by `scripts/download_weights.py inception`. Size and sha256 are pinned in `facehall.weights`; terms are in `docs/THIRD_PARTY_LICENSES.md` §3. They are used for non-commercial research only (D2).
**Reason:**
- They are the standard FID network, and torchmetrics and torch-fidelity (Apache-2.0) are already in the lock file.
- The user approved the download on 2026-11-24.

### Real-world evaluation protocol (PR 2.5)
**Decision:**
- **Protocols:** `configs/eval/wider_real_x{4,8}.yaml` (`kind: realworld`) score a model on the frozen WIDER FACE `test` crops (D20). The frozen inputs are the sha256 of all LR crops.
- **Metrics:**
  - **LR-PSNR-Y**, per face: the SR is downscaled with torch bicubic and antialias, rounded to 8 bit and compared with the input.
  - **FID**, on all faces, against real HR faces.
  - **KID**, mean and std over 100 random subsets of up to 1,000 faces, for every subset of at least 200 faces.
  - The real HR faces are the 19,867 CelebA validation crops at 128 px.
- **Features:** InceptionV3 pool features (2048-d) with the TensorFlow-compatible bilinear resize to 299 px (`antialias=False`).
  - FID and KID use the torchmetrics formulas in float64 on stored features.
  - The reference features are cached per fingerprint.
- **Subsets:** the WIDER attributes: blur 0 / 1 / 2, occluded or not, atypical pose, unpadded.
- **Grids:** samples stay local (D20). `RESULTS.md` holds numbers only.

**Reason:**
- **CelebA validation as the reference:** these are real HR faces with the same template and alignment. The test split stays untouched for tuning, because the WIDER `val` part will be scored against the same reference.
- **CelebA bias in FID/KID:** the reference is celebrity photos, so FID and KID also reward outputs that look like CelebA. They measure realism of the set, not fidelity to the person. They are read together with LR-PSNR-Y and the grids.
- **LR-PSNR-Y is consistency, not quality:**
  - plain upsampling scores high on it;
  - a model that removes noise or compression scores lower, because the input itself contains them.
- **KID besides FID:** KID is unbiased in the sample size. Subsets of 15–550 faces cannot support a reliable FID, and the ×8 test split has only 2,172 faces.
- **Feature cache:** computing the reference features once makes each model evaluation take about 15 s (×4) or 9 s (×8).

### Degradation tuning against real faces (PR 2.6)
**Decision:**
- **Method:** `facehall.eval.match` with the search space `configs/degrade/search/realistic.yaml`
  (`scripts/match_degradations.py`):
  - real LR crops: the WIDER FACE `val` part (D20), 8,160 faces at ×4 and 2,752 at ×8;
  - synthetic LR crops: 5,000 CelebA validation crops, degraded by each candidate;
  - distance: KID and FID on the InceptionV3 features of PR 2.5.
- **Search space:** 135 candidates around `realistic.yaml`:
  - first-stage blur sizes with their upper ends scaled by 0.3, 0.6, 1.0, 1.5 or 2.0;
  - first-stage noise σ up to 15, 25 or 35 levels, with the matching Poisson peaks;
  - lowest JPEG quality 30, 50 or 70, in both stages;
  - second-stage probability 0, 0.5 or 1.

  The base and the four synthetic test levels are scored as references.
- **Views:** every candidate is scored on three views:
  - all real faces, whole crop;
  - typical faces only: no occlusion, typical pose;
  - the centre square of the crop: x 0.1875–0.8125, y 0.28–0.905, from the eyebrows to the
    chin.
- **Selection rule:**
  1. rank the candidates by the mean KID of the three views;
  2. score the 5 best, the base and the references again on a fresh synthetic sample (seed 1);
  3. the best of those 5 replaces the base only if, on that sample, it is better on all three
     KIDs and on FID of the whole crop and of the centre.
- **×8:** the winner passed the rule. It is `configs/degrade/realistic_x8.yaml`:
  - first-stage blur ×0.6: σ ≤ 0.6, motion ≤ 1.8, defocus ≤ 0.9 LR px;
  - noise σ ≤ 35 levels (Poisson peak 26.5–32,500);
  - JPEG quality 70–95 in both stages;
  - second stage unchanged at 0.5.
- **×4:** the winner (blur ×1.5, noise σ ≤ 25, no second stage) was better on the three KIDs but
  worse on both FIDs. `realistic.yaml` stays the ×4 mix.
- **Data use:** the WIDER `test` crops were not used. Re-training on the ×8 mix is a separate
  step.

**Results,** round 2 (seed 1), KID ×10³ (all / typical / centre) and FID (all / centre):

| | ×4 | ×8 |
|---|---|---|
| base `realistic.yaml` | 24.58 / 19.76 / 15.65; 34.44 / 24.13 | 25.35 / 25.96 / 22.17; 31.98 / 24.85 |
| winner | 20.62 / 18.55 / 14.06; 35.81 / 27.05 | 14.22 / 13.40 / 15.13; 25.38 / 21.89 |
| closest test level | medium: 39.49 / 32.39 / 26.71; 44.39 / 30.73 | light: 27.91 / 24.24 / 27.30; 34.15 / 28.03 |

Run on 2026-11-24 from commit `a9bff19`, about 25 min per scale. The full tables are in the local
`outputs/degrade_match/`.

**Reason:**
- **Real photos as the target:** the ranges were initial values from the literature and a visual
  check (PR 2.1). Tuning them on our own synthetic levels would be circular.
- **Content confound:** real and synthetic faces differ in content as well as in degradation:
  occlusions, poses, crowds and backgrounds against posed portraits. Noise can stand in for
  clutter.
  - The first version of the search ranked by KID on whole crops only.
  - At ×4 most of its gain disappears on the content-controlled views. In round 2, base → winner
    is −4.0 on all faces, but −1.2 on typical faces and −1.6 on the centre.
  - Hence the three views and the FID check.
- **Both KID and FID:** the two metrics weigh the feature distributions differently. A change
  that only one of them prefers is not adopted.
- **Second round:** with 135 candidates, the best first-round score is biased low by chance. On
  the fresh sample, the ×8 finalists moved by 0.2–0.9 (mean KID ×10³), and the winner stayed far
  from the base: 14.25 against 24.49.
- **One mix per scale:** the two scales prefer different settings, and neither winner suits the
  other scale. Mean KID ×10³, round 1:
  - the ×8 winner at ×4: 26.89, rank 92 of 135 (base 20.03);
  - the ×4 winner at ×8: 29.39, rank 88 of 135 (base 24.26).

  Why real 16 px faces look less blurred and less compressed than 32 px ones, relative to our
  pipeline, was not checked.
- **Caveat on noise:** the ×8 winner is at the top of the noise axis (σ ≤ 35).
  - The centre view barely separates it from the σ ≤ 25 finalist (centre KID ×10³ 15.13 against
    15.16). That finalist has the better FIDs (24.70 / 20.39).
  - So part of the extra noise may stand for clutter around small faces rather than camera
    noise.
  - A scratch probe outside the committed grid (σ up to 50, not in the repository) scored worse
    than σ ≤ 35 on all three views.
  - The rule's choice is kept, and the caveat is recorded.
- **Frozen test levels are not changed:** the ×8 mix now lies below the medium and heavy levels in
  blur (medium σ 0.4–0.7, motion 1–2, defocus 0.6–1.0 LR px), and above them in JPEG quality
  (medium 50–75, heavy 30–50).
  - A model trained on it is evaluated outside its training range on those levels.
  - The levels stay frozen, so that results remain comparable.
  - Of the levels, medium is closest to real ×4 faces and light to real ×8 faces (table above).

### FNet ×8 on the tuned mix (PR 2.7)
**Decision:**
- **What was trained:** FNet ×8 was re-trained on `configs/degrade/realistic_x8.yaml` (PR 2.6)
  with the PR 2.4 recipe (`configs/train/fnet_x8_tuned.yaml`). Only the degradation differs from
  `fnet_x8_realistic`.
- **×4 not re-trained:** its mix did not change in PR 2.6.
- **Both ×8 models stay as references.** Which ×8 mix Phase 3 trains on is open (Q7 in
  `docs/PROJECT.md` §11).

**Reason:**
- **One variable:** as in PR 2.4, the comparison isolates the effect of the degradation ranges.
- **Cost:** the run took 7.5 min, under the 30 min that needs a separate confirmation. The user
  asked to continue with this step on 2026-11-25.
- **Results** (`docs/PROJECT.md` §9): a trade-off, not a gain everywhere.
  - The tuned mix raises ID similarity on clean, light and medium inputs.
  - It loses LPIPS on medium and heavy inputs. Those levels lie outside its smaller blur range.
  - On real photos, FID and KID are about equal.

### D22 — Phase 3 trains ×8 on the tuned mix (closes Q7)
**Decision:**
- ×8 models of Phase 3 train on `configs/degrade/realistic_x8.yaml` (PR 2.6).
- ×4 models train on `configs/degrade/realistic.yaml`, which PR 2.6 left unchanged.

**Reason:**
- The tuned mix comes from real photos.
- In PR 2.7 it gave better ID similarity on clean, light and medium inputs. Light is the level
  closest to real ×8 faces (PR 2.6).
- FID and KID on real photos were equal.
- It loses LPIPS on medium and heavy inputs, which are far from real ×8 faces.
- The user chose it on 2026-11-25.

### D23 — Quality gate thresholds (`CLAUDE.md` §10)
**Decision:** a new model passes if, against the current reference model of its scale, on both
the light and the medium synthetic test levels (frozen `test` split):
- LPIPS is lower and ID similarity is higher (all images);
- PSNR-Y is at most 0.3 dB lower;
- in every subgroup (`Male` = 1 / 0, `Young` = 1 / 0), ID similarity is at most 0.01 lower.

**Implementation:** `configs/gate.yaml`, `facehall.eval.gate` and `scripts/quality_gate.py`.

**References:**
- **Initial:** the models trained on the same degradation mix with the previous recipe:
  `fnet_x4_realistic-20261124-153302` for ×4 and `fnet_x8_tuned-20261125-090336` for ×8.
- **Replacement:** a model that passes becomes the new reference, in the same PR.

**Not gated:**
- the real-world set: its FID and KID reward invented faces (PR 2.5). It is reported and checked
  visually;
- the bicubic and heavy levels: reported only. Heavy is far from real faces (PR 2.6).

**Defaults taken without a separate question:**
- the choice of light and medium;
- the reference models;
- applying the subgroup rule to both levels.

**Reason:**
- **Thresholds:** the user approved them on 2026-11-25.
- **Levels:** light and medium are the synthetic levels closest to real faces: medium at ×4, light
  at ×8 (PR 2.6).
- **Same mix:** a reference on the same mix isolates the change of model and recipe.
- **Checked on an earlier pair:** the PR 2.7 model (`fnet_x8_tuned`) fails this gate against
  `fnet_x8_realistic` on LPIPS, although the user chose its mix (D22). The gate compares models,
  not degradation mixes.

### SRResNet-v2 and the L1 recipe (PR 3.1)
**Decision:**
- **Model** `srresnet` (`facehall.models.srresnet`, `configs/model/srresnet.yaml`):
  - SRResNet without BatchNorm, with residual scaling 0.1 (EDSR);
  - 16 residual blocks of 64 channels; 3×3 head and tail; ReLU;
  - one conv + pixel-shuffle ×2 stage per factor of 2;
  - the bicubic upsampling of the input added to the output;
  - output linear in training and clamped to [0, 1] in eval mode;
  - 1,517,571 parameters at ×4 and 1,665,283 at ×8.
- **Recipe** (`configs/train/srresnet_x{4,8}.yaml`):
  - L1 loss; Adam with lr 2e-4 and β (0.9, 0.999);
  - cosine decay per optimizer step down to 1e-6;
  - batch 16, 30 epochs, no early stopping, no EMA;
  - ×4 on `realistic.yaml`, ×8 on `realistic_x8.yaml` (D22).
- **Resume:** `facehall train --resume RUN_DIR`. It continues an interrupted run after its last
  completed epoch, from `last_state.pt`, with the run's own config; `--set` and `--smoke` are
  not accepted.
- **Gate result** (D23):
  - ×4 passes all 14 checks and becomes the ×4 reference in `configs/gate.yaml`;
  - ×8 fails: ID similarity is lower than `fnet_x8_tuned` on light (0.170 against 0.175) and
    medium (0.104 against 0.110), and `Young` = 0 on medium drops by 0.0101. `fnet_x8_tuned`
    stays the ×8 reference.

**Defaults taken without a separate question:** residual scaling 0.1, the bicubic skip, lr 2e-4
with cosine decay, 30 epochs and no EMA.

**Reason:**
- **Architecture:** `CLAUDE.md` §9 plans these changes. EDSR (arXiv:1707.02921) removed
  BatchNorm from SR networks. In FNet it may also explain the early gap between train and val
  loss in the PR 1.6 smoke runs (not checked).
- **L1 and lr 2e-4:** the PSNR-oriented setting of ESRGAN (arXiv:1809.00219). The cosine
  decay replaces ESRGAN's step halving: one setting fewer, and the rate depends only on the
  step, which makes resuming exact.
- **Bicubic skip:** the network predicts only the difference from upsampling, so colours and
  low frequencies start right.
- **30 epochs:** about 25 min per run on the RTX 6000 Ada (about 50 s per epoch), under the
  30 minutes that need a separate confirmation. Validation loss still fell slightly at the end
  (best epochs 30 and 29), so a longer run may gain a little more.
- **No EMA:** it belongs to the ESRGAN-recipe step (PR 3.2). Keeping it out leaves the
  architecture and loss as the changes of this step.
- **Resume:** it was tech debt (§11). It becomes needed with longer Phase 3 runs. On CPU, an
  interrupted and resumed run gives the same weights as an uninterrupted one (tested).
- **No ×8 exception:** at ×8 the new model is clearly better on LPIPS (0.288 against 0.346 on
  light) and on PSNR-Y. Its ID losses are small. The gate is still applied as agreed: a model
  that fails it does not replace the reference.

### ESRGAN recipe on SRResNet-v2 (PR 3.2)
**Decision:**
- **Losses** (`facehall.losses`, `facehall.train.objective`): the generator loss of ESRGAN
  (arXiv:1809.00219, Sec. 3.3, Eq. 3), perceptual + λ · adversarial + η · L1, with λ = 5e-3 and
  η = 1e-2 (Sec. 4.1):
  - perceptual: L1 between VGG19 `conv5_4` features of SR and HR, before the activation
    ("VGG19-54"), inputs normalized with the ImageNet statistics; torchvision weights (approved
    on 2026-11-25);
  - adversarial: the relativistic average GAN of Sec. 3.2, Eqs. 1–2;
  - the discriminator is updated once after every generator step, on the same batch.
- **Discriminator:** the layer layout of `VGGStyleDiscriminator` in the authors' BasicSR code
  (Apache-2.0), re-implemented: 128 px input, 64 to 512 channels, BatchNorm, LeakyReLU 0.2,
  14,499,401 parameters.
- **Recipe** (`configs/train/srresnet_gan_x{4,8}.yaml`):
  - the generator starts from the PR 3.1 `best` checkpoint of its scale (`train.init`), as
    ESRGAN starts from its PSNR-oriented model;
  - Adam 1e-4, β (0.9, 0.999), for both networks; cosine decay to 1e-6, which the
    discriminator follows;
  - batch 16, 8 epochs (about 77,000 steps), EMA of the generator with decay 0.999;
  - `best` is the last epoch (`train.best_metric: none`): the pixel validation loss does not
    rank a GAN, and the evaluation protocol is the measure.
- **Code:** `train.init` checks that the checkpoint holds the same model, scale and HR size;
  the discriminator and its optimizer are part of `last_state.pt`, so `--resume` works for GAN
  runs too (tested on CPU).

**Defaults taken without a separate question:**
- the SRResNet-v2 generator instead of ESRGAN's RRDB network: it isolates the effect of the
  losses, and keeps the comparison with PR 3.1 direct;
- cosine decay instead of halving at 50k, 100k, 200k and 300k of 400k steps;
- 8 epochs: one run is about 30 minutes, under the limit that needs a separate confirmation.
  That is about a fifth of ESRGAN's 400,000 steps;
- the paper's relativistic loss and β2 = 0.999. The BasicSR config `train_ESRGAN_x4.yml` (read
  on 2026-11-25) uses a vanilla GAN loss and β2 = 0.99; the paper is the reference here;
- EMA decay 0.999, from the same BasicSR config (the paper does not give one).

**Result** (2026-11-25, from commit `d4f7713`): the recipe fails the quality gate (D23) at both
scales, and the references stay as they are:
- LPIPS and FID improve a lot: ×4 light LPIPS 0.195 → 0.122, WIDER FID 89.5 → 58.5;
- ID similarity falls by about half on every level, below plain bicubic upsampling: ×4 light
  0.445 → 0.243 (bicubic 0.547); PSNR-Y falls by 2.8–5.2 dB;
- in the grids, the faces are sharp but belong to other people, with stains and blotches; at
  ×8, a dark smudge under the nose appears on almost every face.

**Reason for keeping the code:** the losses, the discriminator, `train.init` and the
discriminator state in `last_state.pt` are the base of the next Phase 3 steps (identity loss,
and a smaller adversarial weight under an identity constraint). The configs stay as the record
of this run; they are not a recommended recipe.

### Identity loss on facenet embeddings (PR 3.3)
**Decision:**
- **Recognition network:** facenet's Inception-ResNet-v1, model `20180402-114759` trained on
  VGGFace2, in the PyTorch port of facenet-pytorch. The user approved the download on
  2026-11-25 with the weight license unknown (`docs/THIRD_PARTY_LICENSES.md` §3). It is not the
  model of the ID-similarity metric (`buffalo_l`, D18): another architecture, other training data
  and another input crop, so the metric stays an independent measure. Candidates considered:
  AdaFace (weights license not stated, the same 112 px ArcFace input), SFace (Apache 2.0, ONNX
  only, so it is not differentiable without a conversion), InsightFace `arcface_torch`
  (non-commercial, and the same family as the metric).
- **Code:** `facehall.losses.facenet` re-implements the network with the facenet-pytorch layer
  names (MIT notice kept), so the published state dict loads as it is; its output equals the
  original's exactly (checked on random inputs). `facehall.losses.identity.IdentityLoss` is the
  mean `1 - cos` of the embeddings of SR and HR; the network is frozen and stays in eval mode.
- **Input crop:** a fixed square of the aligned crop, side 0.875 and vertical centre 0.55 (112 px
  at 128), resized to 160 px. Chosen on CelebA validation: 600 identities with two images each;
  the ROC AUC of same against different identity pairs was 0.977–0.984 over 18 squares of side
  0.56–1.0, highest for this one (same-identity mean cosine 0.667, different 0.020). The crop
  barely matters.
- **Recipe** (`configs/train/srresnet_id_x{4,8}.yaml`): the PR 3.1 `best` checkpoint of each
  scale fine-tuned with L1 (weight 1) plus the identity loss; no perceptual or adversarial term,
  so the effect of the identity term is isolated after PR 3.2. Adam 1e-4 with cosine decay to
  1e-6, batch 16, EMA 0.999, `best` = the last epoch. Validation also reports the mean facenet
  similarity.
- **Weight and length, chosen on the val split** (never on test): 2-epoch runs with identity
  weights 0.1, 0.3 and 1.0, scored with `facehall eval --split val` on the light and medium
  levels. Rule, set before the runs: the highest mean ArcFace ID similarity over light and
  medium among the weights that lose at most 0.3 dB of PSNR-Y against the starting L1 model on
  both levels (the PSNR rule of D23). The recipe keeps the 2 epochs of the selection runs;
  longer runs were not tested.

  Val results, PSNR-Y change / LPIPS / ID similarity:

  | Scale | Weight | light | medium |
  |---|---|---|---|
  | ×4 | start (L1) | 27.92 dB / 0.2004 / 0.4475 | 26.40 dB / 0.2465 / 0.2971 |
  | ×4 | 0.1 | −0.14 / 0.2028 / 0.5012 | −0.15 / 0.2510 / 0.3377 |
  | ×4 | **0.3** | −0.27 / 0.1987 / 0.5080 | −0.27 / 0.2476 / 0.3425 |
  | ×4 | 1.0 | −0.42 / 0.1971 / 0.5120 | −0.40 / 0.2497 / 0.3460 |
  | ×8 | start (L1) | 24.44 dB / 0.2950 / 0.1725 | 23.09 dB / 0.3377 / 0.1060 |
  | ×8 | **0.1** | −0.15 / 0.2952 / 0.2124 | −0.11 / 0.3329 / 0.1296 |
  | ×8 | 0.3 | −0.26 / 0.2881 / 0.2118 | −0.19 / 0.3261 / 0.1280 |
  | ×8 | 1.0 | −0.39 / 0.2984 / 0.2139 | −0.28 / 0.3390 / 0.1303 |

  So ×4 uses 0.3 and ×8 uses 0.1. At ×8, 0.1 and 0.3 are nearly equal on ID (mean 0.1710
  against 0.1699), and 0.3 has the better LPIPS; the rule, fixed in advance, decides.

**Result** (2026-11-25, runs from commit `423675c`): the ID metric rises a lot, but the outputs
carry a periodic artifact, so the gain cannot be taken at face value:
- **Test, PSNR-Y / LPIPS / ID similarity on light and medium:** ×4 27.76 / 0.195 / 0.509 and
  26.22 / 0.244 / 0.340 (the L1 start: 28.03 / 0.195 / 0.445 and 26.49 / 0.241 / 0.292); ×8
  24.35 / 0.289 / 0.206 and 23.03 / 0.327 / 0.124 (the reference `fnet_x8_tuned`: 24.29 / 0.346 /
  0.174 and 22.99 / 0.399 / 0.110). ID similarity rises in every subgroup.
- **Gate (D23):** ×4 fails one of 14 checks, LPIPS on medium (0.2441 against 0.2407); ×8 passes
  all 14.
- **Real photos:** FID / KID ×10³ get worse at both scales: ×4 89.5 / 92.3 → 118.3 / 124.6, ×8
  100.8 / 98.8 → 144.9 / 141.4.
- **Grids** (8 synthetic and 8 real samples per scale, zoomed; not measured): a regular grid of
  thin horizontal and vertical stripes, a period of about 3–4 px, over the face, strongest
  around the eyes and nose. Clear at ×8 on every sample, faint at ×4 at full size and clear when
  zoomed. This is the known failure of optimizing a recognition network's embedding: a pattern
  that raises its similarity. Recognition models share such patterns to some degree, so part of
  the ArcFace gain may come from the pattern, not from the face (a hypothesis, not tested).

**Reference:** by the letter of D23, the ×8 model would replace `fnet_x8_tuned`. It is not made
the reference until the user decides (`docs/PROJECT.md` §11, Q8): the gate does not see the
artifact, and a reference whose ID score is inflated would make the gate harder for faithful
models. `configs/gate.yaml` is unchanged.

### D24 — The PR 3.3 ×8 identity model does not become the reference (closes Q8)
**Decision** (the user, 2026-11-25): `srresnet_id_x8-20261125-153350` passes the gate (D23) but is
not made the ×8 reference; `fnet_x8_tuned-20261125-090336` stays.

**Reason:** its outputs carry a periodic grid of stripes that the gate does not see, and part of
its ID gain may come from that pattern (PR 3.3). A reference with an inflated ID score would make
the gate harder for faithful models. A gate check that sees such artifacts is not decided yet;
PR 3.4 uses one for selecting its weights on val (below).

### Identity loss robust to pixel patterns (PR 3.4)
**Decision:**
- **Defences** (`facehall.losses.identity`, acting on SR and HR alike):
  - a random similarity of the recognition square per image: shift up to 0.05 of its side
    (5.6 px at 128, more than one period of the PR 3.3 grid), rotation up to 5°, scale within
    ×1.1, drawn from the loss's own seeded generator; the same transform for the SR and HR of one
    image. A fixed pixel pattern then lands at a different phase in every step;
  - a Gaussian low-pass, sigma 1 HR pixel, before resampling.
- **Checked on CelebA validation pairs** (600 identities, two images each; the PR 3.3 test): the
  defences keep the network's ability to tell people apart. ROC AUC of same against different
  identity pairs: none 0.9842, jitter 0.9850, blur sigma 0.5 / 1.0 / 1.5 0.9842 / 0.9827 /
  0.9780, jitter and blur 1.0 0.9837. Sigma 1.0 is the largest tested blur that loses at most
  0.005 of AUC.
- **Code:** the square is now sampled with `affine_grid` / `grid_sample`. Without jitter, the
  network input equals the PR 3.3 resize inside the square (tested); its outermost pixels now
  sample the image around the square instead of repeating its edge, so the PR 3.3 runs are not
  bit-exact from this code. The jitter generator state is saved in `last_state.pt`, and a
  resumed run equals an uninterrupted one (tested on CPU).
- **Recipe:** `configs/train/srresnet_id_robust_x{4,8}.yaml`, the PR 3.3 recipe with the
  defences: L1 + identity, 2 epochs, Adam 1e-4 with cosine decay, EMA 0.999.
- **Weight selection on the val split, rule set before the runs:** 2-epoch runs with identity
  weights 0.1, 0.3 and 1.0. A weight qualifies if, on val:
  - on light and medium, it meets the D23 rules against the gate reference of its scale
    (`srresnet_x4` at ×4, `fnet_x8_tuned` at ×8): LPIPS not higher, PSNR-Y at most 0.3 dB lower;
  - on the WIDER `val` faces, its KID is at most the starting L1 model's KID plus two standard
    deviations of that estimate. This is the artifact check: the PR 3.3 grid raised the test KID
    ×10³ from 92 to 125 at ×4 and from 99 to 141 at ×8.

  Among the qualifying weights, the highest mean ArcFace ID similarity on light and medium wins.
  If none qualifies, the recipe is reported as failing and no final run is made.

**Selection** (2026-11-25, runs from commit `21aa160`). Val, PSNR-Y and LPIPS change against
the gate reference / ID similarity; WIDER val KID ×10³ (± std):

| Scale | Weight | light | medium | WIDER val KID | Qualifies |
|---|---|---|---|---|---|
| ×4 | start (L1, also the reference) | 27.92 dB / 0.2004 / 0.4475 | 26.40 dB / 0.2465 / 0.2971 | 95.1 ± 2.2 (limit 99.5) | — |
| ×4 | 0.1 | −0.08 / +0.0015 / 0.4858 | −0.08 / +0.0057 / 0.3236 | 87.6 | no: LPIPS |
| ×4 | 0.3 | −0.15 / −0.0003 / 0.4880 | −0.16 / +0.0057 / 0.3238 | 84.5 | no: LPIPS on medium |
| ×4 | 1.0 | −0.27 / −0.0108 / 0.4928 | −0.29 / −0.0073 / 0.3305 | 102.3 | no: KID |
| ×4 | PR 3.3, 0.3 (no defences) | −0.27 / −0.0017 / 0.5080 | −0.27 / +0.0011 / 0.3425 | 121.4 | no: LPIPS, KID |
| ×8 | reference `fnet_x8_tuned` | 24.22 dB / 0.3525 / 0.1779 | 22.94 dB / 0.4058 / 0.1120 | — | — |
| ×8 | start (L1) | +0.22 / −0.0576 / 0.1725 | +0.15 / −0.0682 / 0.1060 | 93.8 ± 1.5 (limit 96.9) | — |
| ×8 | **0.1** | +0.10 / −0.0515 / 0.2029 | +0.07 / −0.0649 / 0.1247 | 87.8 | **yes** |
| ×8 | 0.3 | +0.00 / −0.0616 / 0.2023 | −0.01 / −0.0773 / 0.1253 | 117.3 | no: KID |
| ×8 | 1.0 | −0.14 / −0.0702 / 0.2053 | −0.12 / −0.0844 / 0.1273 | 151.0 | no: KID |
| ×8 | PR 3.3, 0.1 (no defences) | +0.07 / −0.0573 / 0.2124 | +0.04 / −0.0730 / 0.1296 | 125.9 | no: KID |

- **The KID check works as intended:** it rejects both PR 3.3 models, and in the zoomed val
  grids (3 synthetic and 4 real samples per model, not measured) the candidates it rejects show
  faint stripes around the eyes, while the ones it keeps look clean.
- **With the defences**, small weights leave the real-photo KID *lower* than the L1 start's.
  Larger weights bring the artifact back, sooner at ×8.
- **×4: no weight qualifies**, so ×4 gets no full run. The identity term costs LPIPS on medium
  (+0.006 at 0.1 and 0.3) unless the weight is so high that the artifact returns.
- **×8: weight 0.1**, full run below. Its val ID gain over the reference keeps most of PR 3.3's
  (light 0.2029 against 0.2124), now without the grid.

**Result** (2026-11-25, ×8 run `srresnet_id_robust_x8-20261125-174436` from commit `fb2c38c`,
2 epochs, 400 s at 811 img/s):
- **Gate (D23) against `fnet_x8_tuned`: passes all 14 checks.** Light: ID 0.1745 → 0.1968, LPIPS
  0.346 → 0.294, PSNR-Y 24.29 → 24.39 dB; medium: ID 0.1098 → 0.1191, LPIPS 0.399 → 0.333;
  every subgroup gains ID similarity.
- **Against PR 3.3** (no defences): ID 0.206 → 0.197 on light, 0.124 → 0.119 on medium; WIDER
  test FID / KID ×10³ 144.9 / 141.4 → 105.0 / 98.4, the same as the L1 start (100.8 / 98.8).
- **Grids** (8 synthetic and 8 real test samples, not measured): no stripe grid; the faces look
  like the L1 model's, with slightly more defined eyes and mouth.
- **Reference:** by D23 the model would replace `fnet_x8_tuned`. The user said to keep the
  reference when deciding on the PR 3.3 model (D24), so this replacement waited for an explicit
  decision: D25.

### D25 — The PR 3.4 ×8 model becomes the ×8 reference (closes Q9)
**Decision** (the user, 2026-11-25): `srresnet_id_robust_x8-20261125-174436` replaces
`fnet_x8_tuned-20261125-090336` as the ×8 reference in `configs/gate.yaml`. ×4 keeps
`srresnet_x4-20261125-093232`.

**Reason:** it passes all 14 gate checks (D23), and unlike the PR 3.3 model (D24) it shows no
stripe grid and keeps the real-photo KID of the L1 model.

### D26 — Artifact check in the quality gate (closes Q10)
**Decision** (the user approved the proposal on 2026-11-26): the gate (D23) gets one more check.
On the frozen WIDER `test` faces of the candidate's scale (`wider_real_x{4,8}`), the
candidate's KID against real HR faces (all faces) may exceed the reference's by at most 2
standard deviations of the reference's KID.

**Implementation:** `realworld` and `kid_stds` in `configs/gate.yaml`, `facehall.eval.gate`
(`artifact_check`). A candidate now needs a real-world evaluation before the gate runs.

**Defaults taken without a separate question:**
- **One-sided:** a lower KID is not a gain. KID against CelebA rewards invented faces as well
  as faithful ones (PR 2.5), so the check only catches outputs that move away from real photos.
  This amends D23's "the real-world set is not gated".
- **KID, not FID:** KID comes with a spread; the PR 3.4 selection rule used it too.
- **The spread:** the standard deviation that torchmetrics reports over KID subsets of the
  reference. It measures the estimator's sampling spread on one set of outputs, not the spread
  between training runs.
- **All faces**, not the WIDER subsets.

**Reason:**
- **The PR 3.3 artifact:** the stripe grid passed the synthetic checks (LPIPS and PSNR-Y barely
  react to it) but raised the real-photo KID clearly. The same rule on WIDER `val` rejected
  every model with visible stripes in the PR 3.4 selection.
- **Checked on the earlier pairs** (2026-11-26, from the existing evaluations; KID ×10³ of the
  candidate against the limit):

  | Candidate | Reference | KID check | Gate verdict before → after |
  |---|---|---|---|
  | `srresnet_x4` (PR 3.1) | `fnet_x4_realistic` | 92.3 ≤ 100.1, pass | pass → pass |
  | `srresnet_gan_x4` (PR 3.2) | `srresnet_x4` | 55.1 ≤ 96.5, pass | fail → fail |
  | `srresnet_id_x4` (PR 3.3) | `srresnet_x4` | 124.6 > 96.5, **fail** | fail → fail |
  | `srresnet_x8` (PR 3.1) | `fnet_x8_tuned` | 98.8 ≤ 106.8, pass | fail → fail |
  | `srresnet_gan_x8` (PR 3.2) | `fnet_x8_tuned` | 78.8 ≤ 106.8, pass | fail → fail |
  | `srresnet_id_x8` (PR 3.3) | `fnet_x8_tuned` | 141.4 > 106.8, **fail** | pass → **fail** |
  | `srresnet_id_robust_x8` (PR 3.4) | `fnet_x8_tuned` | 98.4 ≤ 106.8, pass | pass → pass |

  The only changed verdict is the PR 3.3 ×8 model, which the user had rejected by hand (D24).
- **Current limits:** ×4 96.46 (`srresnet_x4`, 92.29 ± 2.09), ×8 102.05
  (`srresnet_id_robust_x8`, 98.37 ± 1.84).

**Limits of the check:** KID sees any shift of the output statistics, not artifacts as such. A
model whose outputs are faithful but look less like CelebA could fail it. Visual checks of the
grids stay part of every gate decision.

### Robust identity loss with a perceptual term at ×4 (PR 3.6)
**Context:** at ×4 the robust identity loss (PR 3.4) raises ID similarity on val without the
grid artifact, but at the weights that keep the real-photo KID (0.1, 0.3) it loses about 0.006
LPIPS on medium, so no weight passed the selection.

**Decision:**
- **Recipe:** `configs/train/srresnet_id_perc_x4.yaml`, the PR 3.4 ×4 recipe (L1 1.0, robust
  identity, 2 epochs from `srresnet_x4`, Adam 1e-4 with cosine decay, EMA 0.999) plus the VGG19
  perceptual loss of PR 3.2.
- **Identity weight 0.3:** the PR 3.4 ×4 weight with the highest val ID among those that kept
  the KID; it failed only LPIPS on medium.
- **Perceptual layer conv5_4** (ESRGAN, already used in PR 3.2). A default taken without a
  separate question; conv3_4 is the next candidate if this fails.
- **Perceptual weights 0.01, 0.03 and 0.1,** set from the gradients on the start model
  (2026-11-26; 32 training batches of 16, norms of the gradient with respect to SR, at weight
  1): L1 1.13e-3, identity 0.212, perceptual conv5_4 1.82 (conv3_4 0.66). So the perceptual
  gradient is about ⅓×, 1× and 3× the gradient of the identity term at weight 0.3.
- **Weight selection on the val split, rule set before the runs** (the PR 3.4 rule with the
  D23 checks in full): a weight qualifies if, on val:
  - on light and medium, against the ×4 reference `srresnet_x4`: LPIPS lower, ID similarity
    higher, PSNR-Y at most 0.3 dB lower;
  - on the WIDER `val` faces, KID at most the reference's plus two standard deviations (95.1 ±
    2.2, limit 99.5; D26).

  Among the qualifying weights, the highest mean ID similarity on light and medium wins. A full
  run of the winner then goes through the gate on test. If none qualifies, the recipe is
  reported as failing and no full run is made.

SELECTION_PLACEHOLDER
