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

**Selection** (2026-11-26, runs from commit `55114c4`, three in parallel). Val, PSNR-Y and LPIPS
change against `srresnet_x4` / ID similarity; WIDER val KID ×10³:

| Perceptual weight | light | medium | WIDER val KID | Qualifies |
|---|---|---|---|---|
| reference `srresnet_x4` (L1 start) | 27.92 dB / 0.2004 / 0.4475 | 26.40 dB / 0.2465 / 0.2971 | 95.1 ± 2.2 (limit 99.5) | — |
| none (PR 3.4, identity 0.3) | −0.15 / −0.0003 / 0.4880 | −0.16 / +0.0057 / 0.3238 | 84.5 | no: LPIPS on medium |
| **0.01** | −0.29 / −0.0202 / 0.4832 | −0.26 / −0.0199 / 0.3200 | 79.4 | **yes** |
| 0.03 | −0.37 / −0.0288 / 0.4771 | −0.32 / −0.0326 / 0.3162 | 81.3 | no: PSNR-Y |
| 0.1 | −0.50 / −0.0381 / 0.4658 | −0.42 / −0.0468 / 0.3088 | 84.0 | no: PSNR-Y |

- **LPIPS:** even the smallest perceptual weight wins back far more LPIPS than the identity
  term costs (−0.020 against the reference on both levels).
- **PSNR-Y is the binding rule:** at 0.01 the margin on light is 0.01 dB.
- **KID** falls with every perceptual weight.

**A new periodic pattern.** The zoomed val grids (5 synthetic and 5 real samples per model, not
measured) show a fine diagonal hatching on the lips and the skin with every perceptual weight,
stronger at 0.1. To measure it, the power spectrum was compared with the HR one, with an ad hoc
script that is not in the repository. Method:
- luminance (BT.601) of every 10th val image of the protocol (1,987 images), mean removed, Hann
  window;
- power spectra summed over the images, SR (8-bit) and HR separately;
- the ratio SR / HR per frequency bin; the **peak ratio** is its maximum over the bins at radius
  ≥ 0.25 cycles/px.

| Model | Protocol (val) | Power ratio at 0.375–0.5 cycles/px | Peak ratio |
|---|---|---|---|
| `srresnet_x4` (L1) | medium ×4 | 0.016 | 0.17 |
| robust identity 0.3 (PR 3.4) | medium ×4 | 0.018 | 0.33 |
| `srresnet_id_x4` (PR 3.3, the grid) | medium ×4 | 0.023 | 9.30 |
| + perceptual 0.01 / 0.03 / 0.1 | medium ×4 | 0.123 / 0.138 / 0.144 | 26.96 / 24.81 / 23.42 |
| `fnet_x8_tuned` | light ×8 | 0.008 | 0.26 |
| `srresnet_x8` (L1) | light ×8 | 0.011 | 0.08 |
| `srresnet_id_x8` (PR 3.3, the grid) | light ×8 | 0.018 | 2.27 |
| `srresnet_id_robust_x8` (PR 3.4, reference) | light ×8 | 0.014 | 0.11 |

- **The perceptual pattern is periodic:** sharp peaks at periods of 2 and 4 px, along the axes
  and the diagonals. It fits the max-pooling structure of VGG19. The PR 3.3 grid gives four
  peaks at a period of about 3.7 px.
- **It is near the Nyquist frequency,** where real faces have almost no power, so its peak ratio
  is higher than the PR 3.3 grid's although it is much less visible.
- **The models without a pattern** stay at or below 0.33, including both current references.
- **KID does not see it:** it even falls. D26 catches the PR 3.3 grid but not this pattern.

**Result** (2026-11-26, run `srresnet_id_perc_x4-20261126-094522` from commit `5cc7e53`,
2 epochs, 498 s at 649 img/s alone):
- **Gate (D23, D26) against `srresnet_x4`: passes all 15 checks.** Light: LPIPS 0.195 → 0.176,
  ID 0.445 → 0.481, PSNR-Y −0.29 dB; medium: LPIPS 0.241 → 0.222, ID 0.292 → 0.316, PSNR-Y
  −0.26 dB; every subgroup gains ID; WIDER test KID ×10³ 92.3 → 77.3.
- **Spectrum:** peak ratio 26.87 on val medium, the same as the sweep run.
- **Grids** (6 synthetic and 6 real test samples at about 3×, not measured): sharper eyes and
  lips than the L1 model, with a fine hatching on lips and skin in some faces; much less
  visible than the PR 3.3 stripes.
- **Reference:** by D23 and D26 the model would replace `srresnet_x4`. It carries a periodic
  pattern, the reason for D24, so the replacement waits for the user (`docs/PROJECT.md` §11,
  Q11); `configs/gate.yaml` is unchanged.

### D27 — The PR 3.6 ×4 model does not become the reference (closes Q11)
**Decision** (the user, 2026-11-26): `srresnet_id_perc_x4-20261126-094522` passes the gate (D23,
D26) but does not replace `srresnet_x4-20261125-093232` as the ×4 reference.

**Reason:** its outputs carry a fine periodic pattern from the perceptual term (spectral peak
ratio 25.7 on val medium), as the PR 3.3 model did (D24).

### D28 — Spectral check in the quality gate (closes Q12)
**Decision** (the user approved the proposal on 2026-11-26): the gate (D23, D26) gets a
periodic-pattern check. On each gated protocol (light and medium test), the candidate's
**spectral peak ratio** must be at most **1.0**.

**The measure** (`facehall.eval.spectrum`):
- luma of every SR and HR image (BT.601 weights), mean removed, times a 2-D Hann window;
- power spectra summed over all images of the protocol;
- the ratio SR / HR per frequency bin; the peak ratio is its maximum over the bins at radius
  ≥ 0.25 cycles/px.

It is now recorded by every synthetic evaluation (`spectrum` in `metrics.json`, also the
ratios of five radial bands). Evaluations from before this change lack it; the gate then asks for
a new evaluation of the candidate.

**Implementation:** `spectral_peak` in `configs/gate.yaml`; `facehall.eval.gate`.

**Defaults taken without a separate question:**
- **An absolute limit, not one relative to the reference:** a model without a pattern stays
  below 1 whatever the reference.
- **The meaning of 1:** in no frequency bin may SR have more power, summed over the protocol,
  than the real HR faces. A faithful model can at best match HR power; more is invented energy.
- **Radius 0.25 cycles/px:** periods of 4 px and shorter; the patterns seen so far have periods
  of 2–4 px.

**Limit set on val** (2026-11-26, every val image, from the working tree that became this PR's
first commit; peak ratio on light / medium):

| Model | ×4 | ×8 |
|---|---|---|
| bicubic upsampling | 0.27 / 0.27 | 0.01 / 0.01 |
| FNet (`fnet_x4_realistic`, `fnet_x8_tuned`) | 0.16 / 0.10 | 0.24 / 0.23 |
| SRResNet-v2 L1 (`srresnet_x{4,8}`) | 0.22 / 0.18 | 0.08 / 0.06 |
| robust identity, val sweep 0.3 | 0.49 / 0.31 | 0.21 / 0.29 |
| robust identity ×8, 0.1 (`srresnet_id_robust_x8`) | — | 0.12 / 0.11 |
| robust identity, val sweep 1.0 (rejected by KID in PR 3.4) | 2.93 / 6.62 | 1.34 / 1.53 |
| identity, PR 3.3 (`srresnet_id_x{4,8}`; stripe grid) | 5.92 / 9.21 | 2.30 / 2.79 |
| ESRGAN recipe (`srresnet_gan_x{4,8}`) | 4.21 / 2.35 | 21.28 / 18.89 |
| perceptual 0.03 (val sweep) / 0.01 (`srresnet_id_perc_x4`) | 33.10 / 23.75; 37.01 / 25.66 | — |

- **The two groups:** models without a visible pattern stay at or below 0.49; models with a
  known pattern (PR 3.3, PR 3.6, ESRGAN) are at 2.3 or above. So the proposed limit of 1 holds.
- **Rejected by KID before:** the PR 3.4 val models with weight 1.0, which KID rejected, are
  above the limit too.
- **What it misses:** the PR 3.4 ×8 val model with weight 0.3 (KID 117.3, faint stripes in the
  zoomed grid) stays at 0.21–0.29. The KID check (D26) is still needed: the two checks see
  different things.
- **ESRGAN** puts more power than HR into every high-frequency band (ratios 0.9–1.6), not only
  into peaks: invented texture, as its grids showed (PR 3.2).

**Limits of the check:** it sees regular patterns and excess high-frequency power, not every
artifact (blotches, distorted features). Visual checks stay part of every gate decision.

**Result** (2026-11-26, from commit `462ff81`): the light and medium test evaluations of five
models were run again to record the spectrum. Every other metric reproduced exactly (largest
difference 0, same input digests). Spectral peak ratio on test light / medium and the gate
verdict before → after:

| Candidate | Reference | Spectral peak | Gate verdict |
|---|---|---|---|
| `srresnet_x4` (PR 3.1, ×4 reference) | `fnet_x4_realistic` | 0.22 / 0.17 | pass → pass |
| `srresnet_id_x4` (PR 3.3) | `srresnet_x4` | 5.98 / 9.32, **fail** | fail → fail |
| `srresnet_id_perc_x4` (PR 3.6) | `srresnet_x4` | 38.10 / 26.33, **fail** | pass → **fail** |
| `srresnet_id_x8` (PR 3.3) | `fnet_x8_tuned` | 2.31 / 2.87, **fail** | fail → fail |
| `srresnet_id_robust_x8` (PR 3.4, ×8 reference) | `fnet_x8_tuned` | 0.12 / 0.12 | pass → pass |

The only changed verdict is the PR 3.6 model, which the user had rejected by hand (D27). Both
references pass. Other models keep their old `metrics.json` without the spectrum until they are
evaluated again (the gate shows "—" for such a reference).

### Robust perceptual loss at ×4 (PR 3.8)
**Context:** the VGG19 perceptual term won back the LPIPS that the identity term costs at ×4,
but left a fine periodic hatching (PR 3.6, D27), which the spectral check (D28) now rejects.

**Decision:**
- **Defences** on the perceptual input, as for the identity term (PR 3.4), now shared code
  (`facehall.losses.views.RandomView`):
  - a random similarity of the whole image per image: shift up to 0.05 of its side (6.4 px at
    128, more than the 2 and 4 px periods of PR 3.6), rotation up to 5°, scale within ×1.1;
    the same transform for the SR and HR of one image, drawn from the loss's own seeded
    generator (a stream of its own, saved in `last_state.pt` for `--resume`);
  - a Gaussian low-pass, sigma 1 HR pixel, before resampling.

  Without defences the perceptual loss is computed as before. The identity loss now uses the
  same code; it was checked to give bit-identical values and jitter streams.
- **Recipe:** `configs/train/srresnet_id_perc_robust_x4.yaml`, the PR 3.6 recipe (L1 1.0,
  robust identity 0.3, VGG19 conv5_4, 2 epochs from `srresnet_x4`) with the defences on the
  perceptual input. The values are the identity term's, a default taken without a separate
  question.
- **Perceptual weights 0.03, 0.1 and 0.3,** set by the PR 3.6 rule. On the start model
  (2026-11-26, 32 training batches of 16, gradient norm with respect to SR at weight 1), the
  defended term gives 0.660, against 1.818 without the defences and 0.212 for the identity
  term. The three weights then give about ⅓×, 1× and 3× the gradient of the identity term at
  weight 0.3.
- **Weight selection on the val split, rule set before the runs** (the PR 3.6 rule plus D28):
  a weight qualifies if, on val:
  - on light and medium, against `srresnet_x4`: LPIPS lower, ID similarity higher, PSNR-Y at
    most 0.3 dB lower, and spectral peak ratio at most 1.0;
  - on the WIDER `val` faces, KID at most the reference's plus two standard deviations (limit
    99.5).

  Among the qualifying weights, the highest mean ID similarity on light and medium wins. A full
  run of the winner then goes through the gate on test. If none qualifies, the recipe is
  reported as failing and no full run is made.

**Selection** (2026-11-26, runs from commit `e2d2c6c`, three in parallel). Val, PSNR-Y and LPIPS
change against `srresnet_x4` / ID similarity / spectral peak ratio; WIDER val KID ×10³:

| Perceptual weight | light | medium | WIDER val KID | Qualifies |
|---|---|---|---|---|
| reference `srresnet_x4` | 27.92 dB / 0.2004 / 0.4475 / 0.22 | 26.40 dB / 0.2465 / 0.2971 / 0.18 | 95.1 ± 2.2 (limit 99.5) | — |
| PR 3.6, 0.01 without defences | −0.29 / −0.0202 / 0.4832 / 37.0¹ | −0.26 / −0.0199 / 0.3200 / 25.7¹ | 79.4 | no: spectral peak |
| **0.03** | −0.21 / −0.0163 / 0.4788 / 0.30 | −0.24 / −0.0206 / 0.3169 / 0.34 | 65.0 | **yes** |
| 0.1 | −0.27 / −0.0261 / 0.4704 / 1.22 | −0.30 (−0.305) / −0.0359 / 0.3115 / 1.62 | 66.7 | no: spectral peak, PSNR-Y on medium |
| 0.3 | −0.39 / −0.0392 / 0.4591 / 3.53 | −0.42 / −0.0523 / 0.3036 / 3.94 | 84.7 | no: PSNR-Y, spectral peak |

¹ the full PR 3.6 run (D28 calibration); the sweep run was not measured with this code.

- **The defences work:** at 0.03 the LPIPS gain is about that of PR 3.6 (−0.016 and −0.021
  against −0.020 and −0.020). The spectral peak falls from 37.0 / 25.7 to 0.30 / 0.34, the range
  of the models without a pattern.
- **Side effects:** PSNR-Y loses less than in PR 3.6 (−0.21 against −0.29 dB on light), and the
  real-photo KID is the lowest of the project (65.0).
- **Larger weights bring a pattern back,** as with the identity term. In the zoomed val grids
  (6 samples per model, 5×, not measured), 0.03 shows no hatching on the lips where PR 3.6
  does, and 0.1 shows faint horizontal stripes on the chin.
- ID similarity is about that of PR 3.6 (0.479 against 0.483 on light).

**Result** (2026-11-26, run `srresnet_id_perc_robust_x4-20261126-121325` from commit `e20e095`,
2 epochs, 503 s at 641 img/s):
- **Gate (D23, D26, D28) against `srresnet_x4`: passes all 17 checks.**
  - Light: LPIPS 0.195 → 0.180, ID 0.445 → 0.477, PSNR-Y −0.20 dB, spectral peak 0.34.
  - Medium: LPIPS 0.241 → 0.222, ID 0.292 → 0.313, PSNR-Y −0.24 dB, spectral peak 0.43.
  - Every subgroup gains ID; WIDER test KID ×10³ 92.3 → 61.9.
- **Against PR 3.6:** about the same LPIPS and ID (light 0.179 / 0.477 against 0.176 / 0.481), a
  smaller PSNR-Y loss, no pattern (peak 0.34 against 38.1) and a lower real-photo KID (61.9
  against 77.3).
- **Spectrum on the ungated levels** (val, all images): clean bicubic inputs 0.92 (L1 0.27, PR
  3.6 102.4), heavy 0.65 (L1 0.09, PR 3.6 23.8). On clean inputs a weak residue of the PR 3.6
  peaks remains at the same frequencies, about 110× weaker and below the limit. It is not
  visible in the zoomed test grids (4 clean samples at 5×, not measured).
- **Grids** (6 synthetic, 6 real and 4 clean test samples, not measured): no hatching; eyes and
  lips slightly sharper than the L1 model's. In some faces the eyes have a slight bluish tint,
  as with the identity term alone.
- **Reference:** by D23 the model would replace `srresnet_x4`. The user decided each earlier
  replacement (D24, D25, D27), so this one waited for an explicit decision: D29.

### D29 — The PR 3.8 ×4 model becomes the ×4 reference (closes Q13)
**Decision** (the user, 2026-11-26): `srresnet_id_perc_robust_x4-20261126-121325` replaces
`srresnet_x4-20261125-093232` as the ×4 reference in `configs/gate.yaml`. ×8 keeps
`srresnet_id_robust_x8-20261125-174436` (D25).

**Reason:** it passes all 17 gate checks (D23, D26, D28) without a visible pattern.

### Robust perceptual loss at ×8 (PR 3.9)
**Context:** at ×4 the defended perceptual term won back LPIPS without a periodic pattern (PR
3.8, D29). The ×8 reference (D25) has the robust identity term only.

**Decision:**
- **Recipe:** `configs/train/srresnet_id_perc_robust_x8.yaml`, the ×8 reference recipe
  (`srresnet_id_robust_x8`: L1 1.0, robust identity 0.1, 2 epochs from `srresnet_x8`,
  `realistic_x8.yaml`) plus the defended VGG19 conv5_4 term of PR 3.8, with the same defences
  (shift 0.05, rotation 5°, scale ×1.1, blur sigma 1 HR px). So the candidate differs from the
  reference only by the perceptual term.
- **Perceptual weights 0.005, 0.014 and 0.04.** On the ×8 start model (2026-11-26, 32 training
  batches of 16, gradient norm with respect to SR at weight 1): the defended term 0.676, the
  undefended one 1.989, the identity term 0.279 (0.028 at its weight 0.1). At ×4 only the weight
  at about ⅓× the identity gradient qualified, and 1× and 3× brought a pattern back (PR 3.8).
  So the grid moves one step down: about 1/9×, ⅓× and 1× the identity gradient. A default taken
  without a separate question.
- **Weight selection on the val split, rule set before the runs** (the PR 3.8 rule against the
  ×8 reference): a weight qualifies if, on val:
  - on light and medium, against `srresnet_id_robust_x8-20261125-174436`: LPIPS lower, ID
    similarity higher, PSNR-Y at most 0.3 dB lower, and spectral peak ratio at most 1.0;
  - on the WIDER `val` faces, KID at most the reference's plus two standard deviations.

  The reference is evaluated on val for this (it had no val evaluation). Among the qualifying
  weights, the highest mean ID similarity on light and medium wins; a full run of the winner then
  goes through the gate on test. If none qualifies, the recipe is reported as failing and no full
  run is made.

**Selection** (2026-11-26, runs from commit `8e84f12`, three in parallel; the reference evaluated
on val from the same commit). Val, PSNR-Y change / LPIPS change / ID similarity / spectral peak
ratio against `srresnet_id_robust_x8`; WIDER val KID ×10³:

| Perceptual weight | light | medium | WIDER val KID | Qualifies |
|---|---|---|---|---|
| reference `srresnet_id_robust_x8` | 24.33 dB / 0.2996 / 0.2018 / 0.12 | 23.00 dB / 0.3387 / 0.1232 / 0.11 | 89.1 ± 1.9 (limit 92.9) | — |
| 0.005 | −0.02 / −0.0229 / 0.1960 / 0.13 | −0.02 / −0.0282 / 0.1193 / 0.11 | 74.1 | no: ID similarity |
| 0.014 | −0.05 / −0.0341 / 0.1940 / 0.11 | −0.04 / −0.0400 / 0.1194 / 0.11 | 71.2 | no: ID similarity |
| 0.04 | −0.11 / −0.0473 / 0.1869 / 0.10 | −0.08 / −0.0560 / 0.1155 / 0.10 | 70.3 | no: ID similarity |

- **No weight qualifies.** Every weight fails on ID similarity only, and ID falls as the weight
  grows. LPIPS, PSNR-Y, the spectral peak and KID all pass, with large LPIPS gains (−0.023 to
  −0.056) and a lower real-photo KID.
- **The ID loss is not run-to-run noise:**
  - a second run of the reference recipe (the PR 3.4 sweep run with weight 0.1,
    `srresnet_id_robust_x8_sweep_l0.1-20261125-163518`) differs from the reference by 0.0011
    (light) and 0.0015 (medium);
  - the standard error of each mean is 0.0006;
  - the smallest deficit, at 0.005, is 0.0058 and 0.0039, and it grows with the weight.
- **The same cost appears at ×4:**
  - there, adding the defended perceptual term (0.03) to the identity term (0.3) lowered val ID
    from 0.4880 to 0.4788 on light and from 0.3238 to 0.3169 on medium (PR 3.4 sweep run against
    PR 3.8 sweep run);
  - the ×4 candidate passed only because its reference, `srresnet_x4`, had no identity term;
  - the ×8 reference already has one, so the cost decides here.
- **No pattern:** spectral peak 0.10–0.13, the reference's range.
- **Grids** (5 synthetic and 5 real val samples, zoomed, not measured): slightly sharper hair and
  contours than the reference, no hatching; the faces are as generic as the reference's.

**Result:**
- No full run is made; the ×8 reference stays `srresnet_id_robust_x8-20261125-174436` (D25).
- The config keeps weight 0.005, the closest to the rule, marked as not trained in full.

### Identity weight raised to compensate the perceptual term at ×8 (PR 3.10)
**Context:** at ×8 the defended perceptual term gains a lot of LPIPS and real-photo KID, but
lowers val ID similarity by 0.004–0.012 below the reference's, so no weight met the rule
(PR 3.9). The user asked to compensate with the identity term.

**Decision:**
- **Recipe:** `configs/train/srresnet_id_perc_robust_x8.yaml` with perceptual weight 0.014 and
  a higher identity weight, chosen on val. Of the PR 3.9 weights, 0.014 has about the highest
  ID (mean 0.1567, against 0.1576 at 0.005) and a larger LPIPS gain. The PR 3.9 runs are
  reproduced with `--set train.losses.identity=0.1 --set train.losses.perceptual=<weight>`.
- **Identity weights 0.3 and 1.0.** These follow the PR 3.4 ×8 sweep (val, without the
  perceptual term):

  | Identity weight | ID light | ID medium | WIDER val KID ×10³ |
  |---|---|---|---|
  | 0.1 | 0.2029 | 0.1247 | 87.8 |
  | 0.3 | 0.2023 | 0.1253 | 117.3 |
  | 1.0 | 0.2053 | 0.1273 | 151.0 |

  - So at ×8 the identity weight barely moves the metric ID. Weights between 0.1 and 0.3 are
    not expected to help.
  - The higher weights failed there only on KID (faint stripes). The defended perceptual term
    lowers KID strongly (89.1 → 71.2 at 0.014), so together they may pass.
  - On these numbers, the expected ID gain (+0.002 to +0.003 at 1.0) is smaller than the PR 3.9
    deficit on light (0.0078 at 0.014). A pass is therefore unlikely unless the terms interact.
  - Defaults taken without a separate question; the user asked for the compensation itself.
- **Weight selection on the val split, rule set before the runs:** the PR 3.9 rule unchanged.
  - Against `srresnet_id_robust_x8-20261125-174436` on light and medium: LPIPS lower, ID higher,
    PSNR-Y at most 0.3 dB lower, spectral peak at most 1.0.
  - WIDER `val` KID ×10³ at most 92.9.
  - The highest mean ID wins.

  Because the spectral check missed the faint ×8 stripes of PR 3.4 (D28), zoomed val grids are
  checked too. If no weight qualifies, no full run is made.

**Selection** (2026-11-26, runs from commit `7a5b9bc`, two in parallel). Val, PSNR-Y change /
LPIPS change / ID similarity / spectral peak ratio against `srresnet_id_robust_x8`; WIDER val
KID ×10³:

| Identity weight (perceptual 0.014) | light | medium | WIDER val KID | Qualifies |
|---|---|---|---|---|
| reference `srresnet_id_robust_x8` | 24.33 dB / 0.2996 / 0.2018 / 0.12 | 23.00 dB / 0.3387 / 0.1232 / 0.11 | 89.1 ± 1.9 (limit 92.9) | — |
| 0.1 (PR 3.9) | −0.05 / −0.0341 / 0.1940 / 0.11 | −0.04 / −0.0400 / 0.1194 / 0.11 | 71.2 | no: ID |
| 0.3 | −0.13 / −0.0329 / 0.1965 / 0.26 | −0.10 / −0.0397 / 0.1199 / 0.31 | 78.5 | no: ID |
| 1.0 | −0.24 / −0.0356 / 0.2001 / 0.81 | −0.19 / −0.0402 / 0.1230 / 0.99 | 96.5 | no: ID, KID |

- **No weight qualifies, as expected before the runs.** A tenfold identity weight recovers most
  of the ID deficit (light 0.1940 → 0.2001, medium 0.1194 → 0.1230), but not all of it.
- **What it costs:** PSNR-Y, a pattern (spectral peak 0.81 / 0.99, just under the limit) and
  real-photo KID (96.5, over the limit).
- **The LPIPS gain stays** at every identity weight.
- **Grids** (3 synthetic and 3 real val samples at 5×, not measured): no clear stripe grid. At
  1.0, stronger magenta fringes around the eyes; 0.3 looks like 0.1.
- **Result:** no full run; the ×8 reference stays `srresnet_id_robust_x8-20261125-174436` (D25).
  The config keeps identity 0.3, which fails only on ID, marked as not trained in full.
- **Consequence:** at ×8, with these losses and 2-epoch fine-tuning, the gate's ID rule does not
  allow the perceptual term. Either the identity of the ×8 output has to improve by other means
  (for example the prior-guided model of the plan), or the gate has to be changed, which only
  the user can decide.

### D30 — Move to Phase 4 before the remaining Phase 3 items (closes Q14)
**Decision** (the user, 2026-11-26): the next work is Phase 4, the inference pipeline. The
remaining Phase 3 items (prior-guided model, NME metric, skin-tone subgroup) stay in the
roadmap for later.

**Reason:** Phase 3 meets its "done when" criterion, because both references passed the gate
(D25, D29). The last two ×8 steps (PR 3.9, PR 3.10) did not pass the val rule, and the
prior-guided model is the next way to improve ×8 identity.

### D31 — Face detector: YuNet, on a bicubic upscale by the model scale
**Decision** (the user chose the detector on 2026-11-26; the settings are defaults taken without
a separate question):
- **Detector:** OpenCV Zoo YuNet, `face_detection_yunet_2026may.onnx`. It is MIT-licensed
  (`docs/THIRD_PARTY_LICENSES.md` §3) and runs through `cv2.FaceDetectorYN` of the existing
  OpenCV dependency. The user chose it over InsightFace SCRFD-10GF from `buffalo_l` (stronger on
  small faces, non-commercial) and over both together. The README calls `2026may` the default
  for OpenCV 5.x. On 200 CelebA val images (a one-off check), `2023mar` gives the same
  detections and landmarks.
- **Landmark order:** YuNet's order (right eye of the person first) is `LANDMARK_NAMES` in
  image terms. Checked against the CelebA annotations.
- **Detection on a bicubic upscale by the model scale** (`YuNet(upsample=…)`). The face then has
  about the 32 px between the eyes of the HR template. In the pipeline this is the
  background-upscaled frame.
- **Score threshold 0.6**, NMS 0.3.
- **Templates per scale for YuNet's points** (`YUNET_TEMPLATES`): fitted on CelebA **train**
  scenes, so that aligning detected points gives the training crops up to the per-image error.

**Measurements** (2026-11-26, `scripts/measure_detection.py` from commit `f45cafa`, local
`outputs/detection/results.json`):
- **Scenes:** aligned CelebA images downscaled to a given distance between the eyes (area
  interpolation), padded by their own size with their mean colour.
- **Degraded variant:** Gaussian blur σ 0.5 px, noise σ 3 levels and JPEG quality 70, after
  the downscale.
- **"Detected"** means the highest-scoring face lies within 0.5 interocular distances (NME) of
  the annotation.

Detection rate, clean / degraded, 300 val scenes per cell:

| Eyes (px) | ×1 | ×2 | ×4 | ×8 | ×16 |
|---|---|---|---|---|---|
| 4 (×8 input) | 0.100 / 0.097 | 0.937 / 0.827 | 0.983 / 0.910 | **0.990 / 0.950** | 0.973 / 0.610 |
| 6 | 0.643 / 0.643 | 0.987 / 0.973 | 0.993 / 0.987 | 0.993 / 0.973 | 0.970 / 0.867 |
| 8 (×4 input) | 0.920 / 0.930 | 0.987 / 0.987 | **0.997 / 0.993** | 0.993 / 0.977 | 0.990 / 0.927 |
| 12 | 0.987 / 0.990 | 0.997 / 0.997 | 1.000 / 0.997 | 0.993 / 0.987 | 0.973 / 0.873 |

Landmark NME at the chosen upscale: 4 px ×8 0.124 / 0.177; 8 px ×4 0.097 / 0.112. At ×16 the
degraded faces get worse again.

Alignment error on 800 val scenes. The crop from the detected points is compared with the crop
from the annotated points, as the shift of the crop centre (fraction of the side), rotation and
scale. "Within jitter" is the fraction inside the training jitter on all three axes (shift
0.03, rotation 5°, scale ×1.05):

| Scale, input | Template | Detected | Points (HR px) | Shift p50 / p95 | Rotation p50 / p95 | Scale p50 / p95 | Within jitter |
|---|---|---|---|---|---|---|---|
| ×4 clean | CelebA | 0.998 | 3.46 | 0.014 / 0.032 | 1.36° / 4.14° | 1.036 / 1.112 | 0.617 |
| ×4 clean | YuNet | 0.998 | 3.40 | 0.013 / 0.030 | 1.36° / 4.13° | 1.033 / 1.105 | 0.629 |
| ×4 degraded | CelebA | 0.998 | 3.90 | 0.016 / 0.043 | 1.36° / 4.25° | 1.045 / 1.141 | 0.480 |
| ×4 degraded | YuNet | 0.998 | 3.76 | 0.015 / 0.039 | 1.35° / 4.30° | 1.035 / 1.118 | 0.578 |
| ×8 clean | CelebA | 0.995 | 4.26 | 0.019 / 0.047 | 1.35° / 4.31° | 1.049 / 1.159 | 0.411 |
| ×8 clean | YuNet | 0.995 | 4.09 | 0.018 / 0.043 | 1.37° / 4.39° | 1.042 / 1.136 | 0.491 |
| ×8 degraded | CelebA | 0.943 | 5.84 | 0.030 / 0.084 | 1.45° / 4.94° | 1.076 / 1.227 | 0.194 |
| ×8 degraded | YuNet | 0.943 | 5.36 | 0.026 / 0.076 | 1.46° / 5.02° | 1.052 / 1.177 | 0.313 |

**Reasons and consequences:**
- **Upscale:** without one, YuNet misses most faces of the ×8 input size (10% found). An
  upscale by the model scale finds 95–99%. Larger upscales lose degraded faces again.
- **Templates:** the fitted templates reduce the error on every axis, most in scale, but the
  per-image error dominates.
- **Alignment error versus training jitter:** it often exceeds the training jitter, mostly in
  scale. Only 31% (×8, degraded) to 63% (×4, clean) of faces fall inside it.
- **What follows:** the restoration models will often see crops outside their training jitter.
  The cost is measured end to end in PR 4.2. Training with jitter matched to these errors is
  the candidate remedy.
- **The measurement is optimistic:** CelebA faces are near-frontal and centred, the backgrounds
  are flat, and the degradation is light.
- **No real-photo measurement:** YuNet was trained on WIDER FACE, so the WIDER real-world set
  (training photos) cannot measure it.

### Inference pipeline and its end-to-end evaluation (PR 4.2)
**Decision** (defaults taken without a separate question, 2026-11-26): `facehall.pipeline`
follows `CLAUDE.md` §6.
- **Steps:**
  - the background is the bicubic upscale of the photo by the model scale;
  - faces come from YuNet (D31) or from landmarks given by the caller;
  - each face gets an aligned LR crop of the model input size, and the crops are restored in
    batches;
  - each restored crop is blended into the background through a feathered ellipse, by the
    inverse warp, in a window around the face. Faces are pasted in detection order.
- **Templates:** detected points align to `YUNET_TEMPLATES[scale]` (D31), given points to
  `CELEBA_TEMPLATE`, because given landmarks follow the CelebA convention.
- **LR crop:** a face at or below the crop scale is warped directly (bicubic). A larger face is
  warped to an integer multiple k of the crop size and averaged over k × k blocks.
  - The first version used `ceil(size / scale)` px and a non-integer area resize, so a face just
    above the crop scale went through a 33 → 32 px resize that blurred it. Found and fixed in
    this PR (`11c5f09`).
- **Face-size limit** `max_downscale: 2.0`: a face more than twice the model input size is left
  to the background. Its own pixels hold detail that the model would have to invent from the
  downscaled crop. Initial value, not measured.
- **Mask:** an ellipse centred at (0.5, 0.55) of the crop, radii 0.42 × 0.44, with a smoothstep
  fade over the outer quarter of the radius.
  - It covers the forehead to the chin; hair, ears and background stay bicubic.
  - Initial values, not tuned: checked visually on 6 scenes per scale (no seams).
- **Colour correction** (mean shift of the restored face to the bicubic LR upscale, inside the
  mask): optional, off by default. It has no measurable effect (below).
- **Not in this PR:**
  - the face-box fallback moves to PR 4.3, where the CLI takes user input. A box gives no
    landmarks, so it needs the detector restricted to the box or mean landmarks in it;
  - a background model (optional in `CLAUDE.md` §6).

**End-to-end evaluation** (`scripts/evaluate_pipeline.py` from commit `6320817`, in a separate
worktree; 1,000 CelebA val scenes per scale and input; local `outputs/pipeline_eval/`):
- **Scenes:** built as in D31, with 32 / scale px between the eyes. Degraded scenes add Gaussian
  blur σ 0.5 px, noise σ 3 levels and JPEG quality 70.
- **Variants** (`facehall.eval.endtoend`), each one step closer to the product:
  - `bicubic`: the background alone;
  - `aligned`: the model on an LR crop on the template's pixel grid, the area downscale of the HR
    crop, as in training;
  - `direct`: the model on the LR crop that the annotated landmarks cut from the scene. That
    crop is resampled at LR resolution;
  - `annotated`: the full pipeline with the annotated landmarks;
  - `detected`: the full pipeline with YuNet.
- **Scoring:** against the dataset's HR crop. The frames are scored on the crop that the
  annotated landmarks align in them.

PSNR-Y / SSIM-Y / LPIPS / ID similarity:

| Variant | ×4 clean | ×4 degraded | ×8 clean | ×8 degraded |
|---|---|---|---|---|
| bicubic | 27.14 / 0.796 / 0.326 / 0.712 | 25.11 / 0.699 / 0.505 / 0.438 | 23.47 / 0.638 / 0.544 / 0.340 | 21.87 / 0.572 / 0.684 / 0.129 |
| aligned | 28.80 / 0.842 / 0.144 / 0.675 | 26.65 / 0.760 / 0.220 / 0.363 | 25.09 / 0.716 / 0.268 / 0.317 | 23.40 / 0.651 / 0.333 / 0.143 |
| direct | 27.79 / 0.817 / 0.163 / 0.617 | 26.38 / 0.758 / 0.221 / 0.375 | 24.12 / 0.689 / 0.284 / 0.262 | 22.98 / 0.645 / 0.331 / 0.146 |
| annotated | 27.64 / 0.814 / 0.217 / 0.618 | 25.93 / 0.740 / 0.289 / 0.371 | 24.00 / 0.675 / 0.343 / 0.263 | 22.66 / 0.620 / 0.397 / 0.140 |
| detected | 27.62 / 0.813 / 0.217 / 0.613 | 25.90 / 0.739 / 0.292 / 0.367 | 23.99 / 0.674 / 0.347 / 0.261 | 22.50 / 0.610 / 0.421 / 0.128 |

Models: ×4 `srresnet_id_perc_robust_x4-20261126-121325`, ×8
`srresnet_id_robust_x8-20261125-174436` (the references). The face was detected (landmark NME
below 0.5, the D31 rule) in 0.996 / 0.990 of the ×4 scenes and 0.987 / 0.925 of the ×8 scenes
(clean / degraded). D31 found 0.943 on ×8 degraded scenes: that was another sample of 800 val
rows.

Restored: 0.990 / 0.985 (×4) and 0.982 / 0.923 (×8). The other detected faces had landmarks
that implied a face 2.1–3.0 times the model input, so the face-size limit left them to the
background. In the six ×4 clean cases the mouth points sat far below the eyes. So the limit also
filters implausible landmark sets.

**Observations:**
- **Aligning at LR resolution is the largest cost.** On clean scenes `aligned` → `direct`
  loses 0.058 ID at ×4 (0.675 → 0.617) and 0.055 at ×8, and about 1 dB PSNR-Y.
  - **Not the interpolation:** a one-off check on 250 val scenes compared bicubic, Lanczos and
    bilinear LR warps and a ×4 supersampled warp. They differ by at most 0.011 ID.
  - **Not avoidable by skipping the resampling:** an integer-shift crop without resampling
    helps on clean scenes (+0.016 ID at ×4, +0.029 at ×8) but loses on degraded ones (−0.026,
    −0.009).
  - **Part of the information is lost:** plain bicubic upsampling of the aligned LR crop loses
    too (×4 clean 0.695 → 0.657). The rest of the loss is the mismatch with training: every
    real photo needs a subpixel resampling of an 8 or 4 px face, and the models never saw one.
- **On degraded scenes `aligned` and `direct` are not comparable:** the degradation lands on
  another grid (the crop against the scene), and the LR warp smooths noise and JPEG blocks;
  `direct` is even slightly higher in ID.
- **The paste-back costs almost no identity:** ID within 0.006 of `direct`. It costs 0.12–0.45 dB
  PSNR-Y and 0.05–0.07 LPIPS, because the mask leaves hair and background to the bicubic
  background, and PSNR and LPIPS score the whole crop.
- **Detection costs little where it finds the face:** ID −0.002 to −0.005 on clean scenes and
  ×4 degraded ones. On ×8 degraded scenes it costs −0.012 ID and +0.024 LPIPS: 7.5% of the faces
  are missed and stay bicubic.
- **Against plain bicubic upscaling of the photo:**
  - the pipeline is better in PSNR-Y, SSIM-Y and LPIPS at both scales (×4 clean LPIPS
    0.326 → 0.217);
  - in ID it is lower everywhere except ×8 degraded with annotated landmarks: ×4 0.712 → 0.613
    clean and 0.438 → 0.367 degraded, ×8 clean 0.340 → 0.261;
  - the `bicubic` variant upsamples the scene on its own grid, so it avoids the LR resampling;
  - in the gate protocols the ×4 reference is already below bicubic in ID on clean and light
    inputs (0.682 / 0.477 against 0.711 / 0.547). The pipeline widens that gap.
- **Colour correction:** every metric within 0.0004 of the run without it (×4, commit
  `11c5f09`). The model keeps the mean colour already.
- **Grids** (6 scenes per scale, local, not measured):
  - no visible seams;
  - the restored face is visibly sharper than the bicubic hair around it;
  - at ×8, faces get younger and some look like another person, as in PR 2.5.

**Consequences:**
- The protocol numbers (`RESULTS.md`) overstate the pipeline on real photos, mostly in ID,
  because the protocols cut the LR crop on the template's grid.
- **Candidate remedy:** fine-tune the references on LR crops made like the pipeline's:
  degradation at the scene's resolution, then alignment at LR resolution. Together with the
  detector-matched jitter (D31). Whether it comes before PR 4.3 is Q15.

### D32 — Fine-tune on pipeline-like LR crops before PR 4.3 (closes Q15)
**Decision** (the user, 2026-11-26): fine-tune the restoration models on LR crops made like the
inference pipeline's before PR 4.3.

**Reason:** in the pipeline, the alignment at LR resolution costs about 0.06 ID at both scales
(PR 4.2). The CLI, the API and video will all use these models.

This work becomes PR 4.3. The `facehall enhance` CLI with output marking moves to PR 4.4 and
video to PR 4.5.

### LR crops made like the pipeline's (PR 4.3)
**Data** (`data.scene`, `facehall.data.config.SceneConfig`; defaults taken without a separate
question, 2026-11-26):
- **How a sample is made:**
  - the source is warped to a larger HR scene, in which the face has a random sub-pixel
    position (±0.5 LR px per axis), rotation (±10°) and size (0.8–1.25 times the model input,
    log-uniform);
  - the scene is degraded to LR with the training degradation;
  - the LR crop is cut from it by the inverse transform with `facehall.align.resample_crop`,
    the function the pipeline uses;
  - the HR target is the usual HR crop.
- **Ranges:** the scene's size range is the one of the WIDER real-world crops (D20). The
  rotation range is an initial value.
- **Every split:** `prob: 1.0` applies to every split of the training run, so validation sees
  such crops too. The evaluation protocols do not set it and stay unchanged; probability 0
  draws nothing, and the HR-embedding cache key leaves the setting out.
- **Check:** on 100 val images, the scene LR crop differs from the grid crop by 2.6 (×4) and
  3.8 (×8) levels on average, against 32–45 for a 2 px shift. The geometry is right; what is
  left is the resampling.
- **Loading:** about 2,900 (×4) and 1,900 (×8) samples/s with 8 workers, above the training
  speed.

**Runs:** two variants per scale. Both use the reference recipe (`srresnet_id_perc_robust_x4`,
`srresnet_id_robust_x8`) from the same L1 start, 2 epochs.
- **A** (`srresnet_scene_x{4,8}`): scene crops; nothing else changes.
- **B** (`srresnet_scene_jitter_x{4,8}`): A plus alignment jitter matched to the face detector.
  It uses the p95 errors of YuNet on degraded scenes (D31): ×4 shift 0.04, rotation 5°, scale
  0.12; ×8 shift 0.08, rotation 5°, scale 0.18.

**Selection rule** (set before the runs):
- **Where:** `scripts/evaluate_pipeline.py` on 1,000 val scenes, the `detected` variant (the
  product path), clean and degraded inputs, against the scale's reference.
- **A variant qualifies** when, on both inputs:
  - ID similarity is higher;
  - PSNR-Y is lower by at most 0.3 dB;
  - LPIPS is higher by at most 0.005.
- **The winner:** the qualifying variant with the higher mean ID over both inputs.
- **Then, on test:** the winner and the reference are scored end to end on the test split.
  The winner also goes through the D23/D26/D28 gate on the protocols, which use LR crops on the
  template grid, so a loss there is possible and is reported, not hidden.
- **The reference:** whether the winner replaces it is the user's decision.

**Runs** on 2026-11-26 from commit `0309ebe` in a separate worktree, all four at once, 2 epochs
each (19–23 min each in parallel): `srresnet_scene_x4-20261126-164336`,
`srresnet_scene_jitter_x4-20261126-164416`, `srresnet_scene_x8-20261126-164356`,
`srresnet_scene_jitter_x8-20261126-164436`.

**Selection on val** (1,000 scenes per input, all models scored from `0309ebe`; local
`outputs/pipeline_eval/val_q15/`): **no variant qualifies at either scale.**

`detected` variant, ID similarity / PSNR-Y / LPIPS:

| Model | ×4 clean | ×4 degraded | ×8 clean | ×8 degraded |
|---|---|---|---|---|
| reference | 0.6131 / 27.62 / 0.2173 | 0.3670 / 25.90 / 0.2918 | 0.2614 / 23.99 / 0.3472 | 0.1281 / 22.50 / 0.4209 |
| A: scene crops | 0.6065 / 27.67 / 0.2145 | 0.3611 / 25.90 / 0.2898 | 0.2566 / 24.03 / 0.3485 | 0.1271 / 22.50 / 0.4213 |
| B: scene crops + detector jitter | 0.6073 / 27.67 / 0.2149 | 0.3619 / 25.90 / 0.2903 | 0.2596 / 24.03 / 0.3452 | 0.1323 / 22.56 / 0.4182 |

ID similarity of the other variants, clean / degraded scenes (`aligned`: LR crop on the template
grid; `direct`: LR crop resampled at LR resolution):

| Model | ×4 `aligned` | ×4 `direct` | ×8 `aligned` | ×8 `direct` |
|---|---|---|---|---|
| reference | 0.675 / 0.363 | 0.617 / 0.375 | 0.317 / 0.143 | 0.262 / 0.146 |
| A | 0.648 / 0.351 | 0.611 / 0.370 | 0.293 / 0.136 | 0.260 / 0.145 |
| B | 0.649 / 0.349 | 0.612 / 0.368 | 0.293 / 0.136 | 0.259 / 0.145 |

**Observations:**
- **No gain on the pipeline's crops.** On the crops resampled at LR resolution (`direct`) the
  fine-tuned models score about the same ID as the references (−0.001 to −0.006). On grid crops
  (`aligned`) they lose 0.024–0.027 ID. They adapted to the resampled input by losing on the
  sharp one, not by gaining on the resampled one.
- **What this says about the cost of the alignment at LR** (PR 4.2):
  - after the fine-tune, the gap between grid and resampled crops is 0.037 ID at ×4, the same
    as for plain bicubic upsampling (0.695 → 0.657);
  - together with the interpolation check of PR 4.2, this suggests the loss is mostly
    information that the sub-pixel resampling of an 8 or 4 px face takes away, not a mismatch
    with training. PR 4.2's reading ("the rest is the mismatch with training") is not supported;
  - not tested: training the L1 start from scratch on scene crops (30 epochs, about 30 min per
    scale) instead of 2 fine-tuning epochs.
- **Detector-matched jitter** helps a little only where the detector errs most: on ×8 degraded
  scenes the gap between detected and annotated landmarks shrinks from 0.012 to 0.008 ID
  (0.1323 against 0.1281). Elsewhere B equals A.
- **PSNR-Y and LPIPS** barely move (within 0.05 dB and 0.003).

**Result:** the references stay (D25, D29); no test evaluation or gate run. The `data.scene`
option, `resample_crop` and the four configs stay in the repository as the record of this
negative result.

### `facehall enhance` and output marking (PR 4.4)
**Decision** (defaults taken without a separate question, 2026-11-26; the user confirmed the new
`enhance` subcommand of the public CLI): `facehall enhance` restores the faces of photos and
folders with `facehall.pipeline.Enhancer` and writes marked outputs (`CLAUDE.md` §11).
- **Interface:** `facehall enhance INPUT… -o OUTPUT [--scale 4|8 | --ckpt CKPT]`, as planned in
  `docs/PROJECT.md` §12.
  - **Inputs:** image files or folders, searched recursively.
  - **Output:** one file for one input image; otherwise a folder that mirrors the inputs.
  - **Format:** PNG by default; `--format jpg|webp` with `--quality`.
  - **Safety:** an existing output is kept unless `--overwrite`; an output never replaces an
    input; two inputs may not map to one output.
  - **Errors:** an image that fails is reported, and the others go on; the exit code is 1.
  - **Messages:** only file names and face counts are printed, never image content. The notice
    follows every run.
- **Models:** `--scale` takes the checkpoint from `models.x4` / `models.x8` in
  `configs/pipeline/default.yaml`.
  - These are the gate references (D25, D29), under `${FACEHALL_RUNS:-runs}`, so their run
    folder can move.
  - `--ckpt` takes any checkpoint, and its scale comes from the checkpoint.
- **Face boxes** (`--box`, the fallback planned in PR 4.2) for faces the detector misses:
  - **Inside a box:** the highest-scoring YuNet face at threshold 0.3 (the pipeline's is 0.8,
    D33) whose
    centre lies in the box and whose size is within 3× of the box's. The size factor is an
    initial value for hand-drawn boxes, not tuned.
  - **Measured on the threshold** (300 CelebA val scenes per cell, commit `903919e`): 0.3
    instead of 0.6 finds at most 3 points more faces, with the same NME. Detected clean /
    degraded:
    - 4 px between the eyes, ×8 upscale: 0.990 / 0.950 → 0.993 / 0.960;
    - 8 px, ×4 upscale: 0.997 / 0.993 → 1.000 / 0.993.
  - **Without a detection:** the landmarks come from the box itself, as YuNet's mean landmarks
    in its own box (`YUNET_BOX_LANDMARKS`). They are fitted on the 1,500 train scenes per scale
    of `YUNET_TEMPLATES` (1,495 and 1,489 detected) and mirror-averaged.
  - **Cost** on 800 val scenes (`scripts/measure_detection.py --parts box`, commit `903919e`),
    detected landmarks against landmarks from the detected box:

    | Scale, input | Shift p95 | Rotation p95 | Scale p95 | Within jitter |
    |---|---|---|---|---|
    | ×4 clean | 0.030 / 0.100 | 4.1° / 1.5° | 1.105 / 1.103 | 63% / 35% |
    | ×4 degraded | 0.039 / 0.098 | 4.3° / 1.5° | 1.118 / 1.121 | 58% / 29% |
    | ×8 clean | 0.043 / 0.099 | 4.4° / 1.5° | 1.136 / 1.131 | 49% / 27% |
    | ×8 degraded | 0.076 / 0.094 | 5.0° / 1.5° | 1.177 / 1.181 | 31% / 20% |

    - A box fixes the face's size about as well as the landmarks, but its centre less well.
      The face's own rotation is unknown, so the small rotation error only reflects the
      upright CelebA faces.
    - These numbers hold for YuNet's tight boxes. A hand-drawn box twice as large gives a face
      twice the size, which the face-size limit then leaves to the background; the help text
      describes the box YuNet draws.
  - **The risk:** a box on a place without a face makes the model draw one there. The user
    asked for it, and the output is marked.
- **Tiled detection:** OpenCV's YuNet crashed (segfault) on a 1920 × 1080 image upscaled ×8
  (133 MP input), and on a 1100 × 700 one at ×8 (49 MP); it ran at 33 MP.
  - So a larger image is detected in overlapping tiles whose upscale has at most 16 MP:
    `detector.max_pixels`; `tile_overlap` 128 input px, more than any face the pipeline restores.
  - The detections are merged by non-maximum suppression (IoU 0.3).
  - **Check:** on canvases of CelebA val scenes, tiles of 300 px (×4) and 200 px (×8) found the
    same faces as the whole image (17 of 17 and 30 of 30).
  - **Time and memory:** a 1920 × 1080 frame with 34 faces at ×8 takes 8.1 s end to end on the
    RTX 6000 Ada, including the 133 MP PNG, with 3.3 GB peak memory; 33 faces are restored.
- **Output size limit** `max_output_pixels: 200_000_000` (600 MB of RGB); a larger output is
  refused.
- **Lower face-size limit** `max_upscale: 2.0`, the mirror of `max_downscale` (PR 4.2): a face
  more than twice smaller than the model input is left to the background.
  - **Why:** the models were trained on faces of 0.8–1.25 times their input size (`data.scene`,
    PR 4.3) and the detector's p95 scale error is 1.18 (D31).
  - **Seen:** on a real photo (WIDER FACE), a ×4 detection with score 0.60, 2.9 times smaller
    than the model input, lay on background texture. The model drew a dark blob there.
  - Initial value, not tuned.
- **Paste-back in uint8:** the background stays uint8, and only a window around each face is
  blended in float and rounded. Same output as before for single faces. On 320 frames of the
  PR 4.2 protocol it differed in one frame, with two overlapping detections, by 1 level.
- **Marks** (`facehall.marking`):
  - **PNG text chunks:** `AI-enhanced` = `true`, `Description` = the notice, `Software`, and
    `facehall` = the record as JSON.
  - **XMP** (PNG iTXt `XML:com.adobe.xmp`; JPEG and WebP through Pillow):
    - `Iptc4xmpExt:DigitalSourceType` =
      `http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia`;
    - `dc:description` = the notice, `xmp:CreatorTool`;
    - `facehall:AIEnhanced`, `Model`, `ModelSHA256`, `InputSHA256` and `Record` (the JSON
      record), in the namespace `https://github.com/dronja/robodream_cv/xmp/1.0/`.
  - **EXIF** in every format: `ImageDescription` = the notice and `Software`.
  - **The record:**
    - the notice, the package version and the UTC time;
    - the model: run, checkpoint, weights sha256, architecture, scale;
    - the pipeline settings without local paths;
    - the input sha256, its size and where the faces came from;
    - the output size;
    - per face: source, score, restored or why not, and the changed region.

    It holds no file names, landmarks or image content.
  - **Visible mark** (`--watermark`): "AI-enhanced" on a dark band in the bottom-right corner.
    Optional and off by default, as `CLAUDE.md` §11 allows.
- **The IPTC term:** chosen from the IPTC Digital Source Type vocabulary (cv.iptc.org, read on
  2026-11-26).
  - Verbatim: `compositeWithTrainedAlgorithmicMedia`, "Edited using Generative AI":
    "Augmentation, correction or enhancement using a Generative AI model, such as with
    inpainting or outpainting operations".
  - The alternatives do not fit:
    - `algorithmicallyEnhanced` is "without changing the main content of the media", and the
      restored faces are new content;
    - `trainedAlgorithmicMedia` is for wholly generated media.
  - No IPTC page names super-resolution, so the choice is ours.
  - *Needs verification:* the `Iptc4xmpExt` namespace URI was not read verbatim from the IPTC
    specification; it is the commonly used one.
- **Real photos** (two WIDER group photos, then two crowd photos; local only):
  - At the D31 threshold 0.6, all 9 faces restored at ×8 in the group photos were false
    detections: tie patterns, a button and printed letters. The model drew faces on them. This
    led to the measure below and D33.
  - At 0.8, no false face was restored in those photos. In a team photo at ×4, all 66 restored
    faces were real, and the restorations looked plausible (seen, not measured).
- **C2PA: not now.** A C2PA manifest has to be signed. A self-signed local certificate is not
  trusted by validators, and the product stays local (D13). It also needs a new dependency.
  Revisit with Phase 6, if outputs are ever published.

### D33 — Detector score threshold 0.8 in the pipeline (amends D31)
**Decision** (the user, 2026-11-26): `facehall.pipeline` detects faces with YuNet at score
threshold **0.8 at both scales**, instead of D31's 0.6. Inside a face box given by the user it
stays 0.3 (PR 4.4).

**Why:** D31 chose 0.6 on synthetic CelebA scenes, whose flat backgrounds have nothing to
mistake for a face (a limitation D31 recorded). On real photos, a ×4 or ×8 upscale makes
textures look like tiny faces, and the model then draws a face on them. A missed face only
stays bicubic, and `--box` can add it; a false one is invented content (`CLAUDE.md` §11).

**Measured** (`scripts/measure_detection.py --parts wider`, commit `e76633e`):
- **Data:** 200 photos of the WIDER FACE `val` part (D20), against the official face boxes.
- **Restorable detections:** those of a size the pipeline restores (crop scale ½–2).
- **A face:** a detection whose box overlaps an official box with IoU ≥ 0.3.
- **Recall:** official faces of restorable size, judged from the box width, that are found.
- **Bias:** YuNet was trained on the WIDER FACE training photos, these included, so all the
  numbers are optimistic.

| Threshold | ×4: faces among detections | false per photo | faces found | ×8: faces among detections | false per photo | faces found |
|---|---|---|---|---|---|---|
| 0.3 | 0.527 | 5.38 | 0.809 | 0.180 | 19.00 | 0.701 |
| 0.5 | 0.761 | 1.68 | 0.733 | 0.390 | 5.48 | 0.584 |
| 0.6 (D31) | 0.837 | 0.98 | 0.690 | 0.513 | 2.88 | 0.506 |
| 0.7 | 0.917 | 0.42 | 0.629 | 0.666 | 1.27 | 0.425 |
| **0.8** | **0.971** | **0.12** | **0.517** | **0.845** | **0.32** | **0.293** |
| 0.9 | 1.000 | 0.00 | 0.191 | 1.000 | 0.00 | 0.037 |

(1,473 official faces of restorable size at ×4, 1,105 at ×8.) Some "false" detections may be
faces that the official boxes miss, or match with IoU below 0.3; the ones seen were not faces.

**Cost on synthetic CelebA scenes** (300 val scenes per cell, as in D31), detected clean /
degraded at 0.6 → 0.8:
- ×4 (8 px between the eyes, ×4 upscale): 0.997 / 0.993 → 0.997 / 0.987;
- ×8 (4 px, ×8 upscale): 0.990 / 0.950 → 0.990 / 0.900.

**Consequences:**
- On real photos the pipeline restores about half the small faces at ×4 and under a third at
  ×8; even at 0.8, about one restorable ×8 detection in seven is not a face.
- The PR 4.2 and PR 4.3 end-to-end numbers (`detected` variant) were measured at 0.6.
- D31's measurements (detection rate, alignment error, templates) are unchanged.

### Video: frame by frame, face tracks and the flicker measure (PR 4.5)
**Decision** (defaults taken without a separate question, 2026-11-26; the face-track settings
were chosen by rules set before the runs, one of them amended as described below):
`facehall enhance` also takes videos. Each frame is restored on its own by the PR 4.2
pipeline. The faces are tracked from frame to frame with an alpha-beta filter (alpha 0.5,
beta 0.1) and a detection-score hysteresis (a track begins at 0.8, D33, and continues down to
0.3). The output is encoded by the system `ffmpeg`, keeps the input's audio and carries the
marks as container tags.

- **Interface:** video files (`.mp4 .mov .mkv .avi .webm .m4v`) are inputs like images, also
  inside folders.
  - **Outputs:** MP4 by default; `--video-format mkv|mov` in an output folder, or the suffix of a
    single output file. The output must be of the input's kind.
  - **Faces given by hand:** `--landmarks` and `--box` are refused for videos.
  - **Errors:** a failing video does not stop the other inputs, as for images.
- **Reading:** OpenCV `VideoCapture` (the FFmpeg backend of `opencv-python-headless`).
  - A rotation stored in the file is applied, so the frames are upright, and the output carries
    no rotation (tested with `-display_rotation 90`).
  - The frame rate is OpenCV's `CAP_PROP_FPS`, and the output is written at that constant rate.
    For variable-rate inputs this is *needs verification*: no such input was tried.
- **Encoding:** the restored RGB frames go to `ffmpeg` through a pipe: libx264 (or libx265),
  CRF 18, preset `medium`, `yuv420p` (`video.codec`, `crf`, `preset`). A second `ffmpeg` pass
  copies that stream, adds the input's audio (`video.audio`) and writes the tags.
  - **Audio:** copied into MKV; re-encoded to AAC in MP4 and MOV, which accept few audio codecs.
  - **Why ffmpeg:** the pip build of OpenCV writes no H.264 here (`avc1` and `H264` fail; it has
    MPEG-4 part 2, MJPG, FFV1 and VP9), no audio and no tags. The system `ffmpeg` 8.0.1 has
    libx264, libx265 and AAC.
  - `ffmpeg` is a separate program on `PATH`, needed for videos only; its license is noted in
    `THIRD_PARTY_LICENSES.md` §4.4 (the Ubuntu build is GPL).
- **Marks** (`facehall.marking.video_tags`): container tags `ai_enhanced`, `comment` and
  `description` (the notice), `digital_source_type` (the IPTC term of PR 4.4) and `facehall`
  (the record).
  - MP4 and MOV keep custom keys only with `-movflags +use_metadata_tags`; checked with
    `ffprobe`.
  - **The record** holds the model, the parameters, the input sha256, the frame rate and count,
    and face counts over all frames instead of one entry per face.
  - No XMP in videos, and C2PA stays deferred (PR 4.4).
- **Face tracks** (`facehall.video.FaceTracker`):
  - **Matching:** detections are matched to the predicted tracks greedily by box IoU ≥ 0.3.
  - **Filter:** each track's five landmarks and box go through an alpha-beta filter. A new
    track starts with zero velocity. In the first version, the second detection set the
    velocity; its jitter (a box growing by 1 px) pushed a profile face over the face-size limit
    for several frames.
  - **Hysteresis:** YuNet runs at 0.3 (`video.track_threshold`). A detection at the pipeline's
    threshold (0.8, D33) may begin a track; a weaker one may only continue one, and is dropped
    otherwise.
    - The detections at or above 0.8 from the 0.3 run equal those of a 0.8 run: 960 of 960
      frames identical at both scales (NMS keeps the higher-scoring box).
  - **No face without a detection:** `video.hold` can keep restoring a track that has no
    detection at all at its predicted place. It is off by default: after a cut, or when a person
    turns away, it would draw a face on whatever is there. The synthetic clips cannot show
    this, because they always contain a face.

**Flicker measure** (`facehall.eval.temporal`, `scripts/measure_temporal.py`):
- **Clips:** an aligned CelebA image as a small LR scene over 24 frames, with the face at the
  model input size (8 px between the eyes at ×4, 4 px at ×8). Each frame is rendered from the
  source: padded with its mean colour, shifted (bicubic) at the source resolution, then area
  downscaled. The degradation is that of D31 (blur 0.5, noise 3, JPEG 70), drawn anew per frame.
- **Conditions:**
  - `static_degraded`: no motion;
  - `drift_clean` and `drift_degraded`: (0.37, 0.13) LR px per frame;
  - `sway_degraded`: a sideways sine, 2 LR px amplitude, 16-frame period;
  - `fast_degraded`: (1.5, 0.5) LR px per frame (added as described below).
- **Flicker:** the face crops of consecutive frames, each aligned by the frame's true landmarks
  so that the motion cancels. The measure is the mean absolute luma difference inside the
  paste-back mask, in 8-bit levels, averaged over frame pairs. The HR clip, rendered the same
  way at ×scale, gives the floor: 0 when static, 0.7–1.3 when moving.
- **Also measured:** PSNR-Y of each crop against the frame's HR crop, and the fraction of frames
  in which the face was restored (NME below 0.5, as in D31).
- **Variants:** bicubic; `annotated` (true landmarks in every frame); `detected` (YuNet at 0.8
  in every frame); `track:<alpha>,<beta>,<hold>[,<low>]`.
- **Data:** 100 val clips for the selection and 100 test clips for the report; the two gate
  references (D25, D29).

**Round 1, smoothing** (val, commit `780193f`; the ×4 run's tree differed from it only in tests
and the README):
- **Rule** (set before the runs): a setting qualifies if, at both scales and in every condition,
  its PSNR-Y is at most 0.05 dB below `detected`'s and its restored fraction is not lower. Of
  those, the lowest mean flicker over all cells wins. If none qualifies, tracking stays off.
- **Mean flicker** over 2 scales × 4 conditions (`fast_degraded` came later): `detected` 5.982,
  `annotated` 5.339.

  | alpha, beta, hold | Mean flicker | Rule |
  |---|---|---|
  | 1, 0, 2 | 5.926 | qualifies |
  | 0.5, 0.1, 2 | 5.561 | qualifies |
  | 0.3, 0.05, 2 | 5.479 | qualifies |
  | 0.2, 0.02, 2 | 5.425 | fails: restored ×8 sway 0.849 < 0.886 |
  | 0.5, 0, 2 | 5.556 | qualifies |
  | **0.3, 0, 2** | **5.418** | **qualifies, lowest** |
  | 0.3, 0.05, 0 | 5.545 | fails: restored ×8 sway 0.863 < 0.886 |

- **Amendment** (made after these results and before the next runs): with beta 0 the filter has
  no velocity. It trails a face moving at speed v by (1 − alpha) / alpha · v, 2.3 v at
  alpha 0.3. The four conditions move at most 0.79 LR px per frame, so they cannot show this,
  but a 16 px face crossing a frame easily moves 2 px per frame.
  - So `fast_degraded` was added, and the rule now also requires the winner to qualify there.
    Otherwise the next setting by flicker that does is taken.
  - A first `fast_degraded` run is void: at ×8 the face left its scene, which was padded by one
    image size (18 LR px). The fast scene is now padded by three.
- **`fast_degraded`** (val, commit `2969305`), restored fraction and PSNR-Y against `detected`.
  `detected` restored 0.867 at ×8 and 0.998 at ×4:

  | alpha, beta, hold | ×8 restored | ×8 PSNR-Y Δ | ×4 restored | ×4 PSNR-Y Δ | Rule |
  |---|---|---|---|---|---|
  | 1, 0, 2 | 0.907 | +0.02 | 1.000 | 0.00 | qualifies |
  | **0.5, 0.1, 2** | **0.905** | **+0.02** | **1.000** | **0.00** | **qualifies; the winner** |
  | 0.3, 0.05, 2 | 0.753 | −0.04 | 0.999 | −0.01 | fails |
  | 0.2, 0.02, 2 | 0.512 | −0.16 | 0.883 | −0.06 | fails |
  | 0.5, 0, 2 | 0.846 | −0.08 | 1.000 | −0.02 | fails |
  | 0.3, 0, 2 | 0.534 | −0.15 | 0.909 | −0.08 | fails |
  | 0.3, 0.05, 0 | 0.727 | −0.06 | 0.997 | −0.01 | fails |

  The strong filters trail the face: the crop is then off the face, and the face-size and
  alignment checks leave it bicubic.

**Round 2, hysteresis instead of hold** (val, commit `e9b3224`, all five conditions):
- **Rule** (set before the runs): the round-1 rule against `detected`, for candidates without
  hold (hysteresis down to 0.3 or 0.5, alpha 0.5, beta 0.1); of those that qualify, the lower
  mean flicker wins. If neither qualifies, the round-1 winner with hold 2 stays. The hold
  variants are shown for comparison.

  | Variant | Mean flicker | ×8 restored: static / drift / sway / fast | Rule |
  |---|---|---|---|
  | `detected` | 6.274 | 0.884 / 0.884 / 0.886 / 0.867 | — |
  | `annotated` | 5.647 | 1 / 1 / 1 / 1 | — |
  | 0.5, 0.1, hold 2 (round 1) | 5.859 | 0.922 / 0.940 / 0.926 / 0.905 | qualifies |
  | 0.5, 0.1, no hold, no hysteresis | 5.929 | 0.886 / 0.890 / 0.884 / 0.863 | fails: restored |
  | **0.5, 0.1, hysteresis 0.3** | **5.865** | **0.953 / 0.977 / 0.954 / 0.963** | **qualifies; the winner** |
  | 0.5, 0.1, hysteresis 0.5 | 5.873 | 0.949 / 0.965 / 0.948 / 0.956 | qualifies |
  | 0.5, 0.1, hold 2, hysteresis 0.3 | 5.862 | 0.953 / 0.980 / 0.954 / 0.965 | (has hold) |

  ("Drift" is `drift_degraded`.) At ×4 every tracked variant restores 0.995–1.000 of the frames.
  Hysteresis flickers about as little as hold and restores more faces, without restoring any
  face that is not detected in its frame.

**Test** (100 test clips per condition, commit `132cc0a`). The tracker with the defaults
(alpha 0.5, beta 0.1, hysteresis 0.3, no hold) against per-frame detection. Flicker in levels /
PSNR-Y in dB / restored fraction:

| Scale, condition | bicubic | `annotated` | `detected` | tracked (defaults) |
|---|---|---|---|---|
| ×4 static_degraded | 3.12 / 24.85 / — | 3.51 / 25.82 / 1 | 4.03 / 25.77 / 0.992 | 3.71 / 25.78 / 0.995 |
| ×4 drift_clean | 3.55 / 27.40 / — | 4.31 / 27.99 / 1 | 4.66 / 27.96 / 1.000 | 4.43 / 27.97 / 1.000 |
| ×4 drift_degraded | 4.63 / 24.79 / — | 4.97 / 25.74 / 1 | 5.29 / 25.71 / 0.993 | 5.12 / 25.71 / 0.993 |
| ×4 sway_degraded | 4.39 / 24.83 / — | 4.72 / 25.78 / 1 | 5.08 / 25.75 / 0.991 | 4.93 / 25.75 / 0.990 |
| ×4 fast_degraded | 5.33 / 24.79 / — | 5.62 / 25.75 / 1 | 5.85 / 25.72 / 0.991 | 5.71 / 25.72 / 0.993 |
| ×8 static_degraded | 3.53 / 21.20 / — | 4.55 / 22.05 / 1 | 5.75 / 21.85 / 0.869 | 4.84 / 21.88 / 0.922 |
| ×8 drift_clean | 5.35 / 22.97 / — | 6.44 / 23.60 / 1 | 6.86 / 23.55 / 0.985 | 6.45 / 23.57 / 0.990 |
| ×8 drift_degraded | 5.79 / 21.14 / — | 7.11 / 22.01 / 1 | 8.07 / 21.79 / 0.873 | 7.41 / 21.85 / 0.956 |
| ×8 sway_degraded | 5.40 / 21.17 / — | 6.66 / 22.03 / 1 | 7.57 / 21.82 / 0.863 | 7.01 / 21.85 / 0.933 |
| ×8 fast_degraded | 6.70 / 21.14 / — | 8.07 / 22.02 / 1 | 8.97 / 21.80 / 0.868 | 8.34 / 21.84 / 0.949 |

The HR floor is 0 when static and 0.73–1.23 when moving. The one cell where tracking restores
fewer frames than per-frame detection is ×4 sway_degraded, 0.990 against 0.991 (about 2 of
2,400 frames).

**Observations:**
- **The model flickers more than bicubic, even with perfect landmarks:** ×8 static 4.55
  against 3.53, ×4 3.51 against 3.12. Each frame is restored on its own, and the model
  amplifies the frame-to-frame noise, compression and sub-pixel changes into changing detail.
  Tracking cannot remove this.
- **Detection jitter adds flicker on top:** `detected` against `annotated`, +0.5 at ×4 and
  +1.2 at ×8 on static faces. The tracker removes most of it (×8 static 5.75 → 4.84) and
  restores the face in more frames (×8 0.87 → 0.92–0.96), at equal or better PSNR-Y.
- **A strong filter trails a moving face:** see the `fast_degraded` check above. The defaults
  keep a velocity (beta 0.1), so they follow constant motion.
- **Real video** (a 10 s pan over a WIDER group photo, local): six consecutive ×4 frames show
  the same faces without jumps or dropouts, with small changes in the eyes and mouths (seen, not
  measured).
- **Speed** (RTX 6000 Ada, about 62 faces per frame): 250 frames of 640 × 360 at ×4 (output
  2560 × 1440) took 30.1 s end to end, 8.3 frames/s; 320 × 180 at ×8 took 29.1 s. Peak memory
  was 2.4 GB in both cases.

**Next steps** (proposals, not decided): a temporal model, or blending each face's restored
crop over time in the template frame; cut detection; a real video set for the measure.
