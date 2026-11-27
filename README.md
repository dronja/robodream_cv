# 🧠 Face Hallucination

Low-resolution face images are a common issue in many fields, such as video surveillance systems, forensics, old photo restoration, face recognition systems, and more.

**Face Hallucination** is a type of *super-resolution*, specifically designed to restore details and increase the resolution of face images based on knowledge of facial structure.

---

## 🎯 Goal

Transform blurry or low-quality face images into high-resolution ones using knowledge about typical facial features — while **preserving the identity** of the person.

- **×4:** an aligned 32×32 face crop becomes 128×128 (about 8 px between the eyes in the input).
- **×8:** an aligned 16×16 face crop becomes 128×128 (about 4 px between the eyes).
- **In photos and videos,** `facehall` finds such small faces, restores each one and pastes it
  back into an upscale of the whole image.

> 🔥 **Main challenge:** identity and realism preservation

---

## ⚠️ Responsible use

The output is a **plausible reconstruction, not a recovery of real details**. The smaller the input
(especially ×8 from 16 px), the more detail is invented. The results must not be used to identify
a person or as evidence; at most they are a hint, to be shown next to the original.

- **Invented details:** the models make older people look younger, can drop glasses, and at ×8
  can draw a whole face from a dark blur (seen in the evaluation grids, see
  [Limitations](#limitations)).
- **Every output is marked** as AI-enhanced in its metadata, and `--watermark` adds a visible
  mark.
- **Use:** non-commercial research only, because of the training data and the third-party models
  ([Licenses](#-licenses)).

---

## 🚧 Project status

The research notebook of this repository has become the package `facehall`: training,
evaluation, a command-line tool, ONNX models, a local REST API and Docker images. Phases 0–5 of
the plan are done; Phase 6 (release documentation) is in progress. The current state and the plan
are in [`docs/PROJECT.md`](docs/PROJECT.md), decisions in [`docs/DECISIONS.md`](docs/DECISIONS.md)
and all metrics in [`docs/RESULTS.md`](docs/RESULTS.md).

**The trained weights are not published.** They are derived from CelebA, whose terms allow
non-commercial research only, so they stay local. To use the tools, train the models yourself
(`docs/PROJECT.md` §12) on a copy of CelebA obtained under its terms.

```bash
uv sync                  # PyTorch from PyPI (CUDA build on Linux)
uv sync --extra cpu      # or: CPU-only PyTorch (then use `uv run --extra cpu ...`)
uv run pytest
uv run facehall --version
```

---

## 🚀 Usage

### Enhancing photos and videos

`facehall enhance` detects the faces of a photo or of every video frame, restores each one with
a model and pastes it back into a bicubic upscale:

```bash
uv run facehall enhance photo.jpg -o out.png --scale 4          # one photo
uv run facehall enhance photos/ -o enhanced/ --scale 8 --watermark   # a folder, visible mark
uv run facehall enhance photo.jpg -o out.png --box 120,80,18,22  # a face the detector misses
uv run facehall enhance photo.jpg -o out.png --ckpt runs/<id>/best.safetensors
uv run facehall enhance clip.mp4 -o clip_x4.mp4 --scale 4        # a video (needs ffmpeg)
```

- **Face sizes:** the models restore small faces: about 8 px between the eyes at ×4 and 4 px at
  ×8. Faces more than twice as large or as small are left to the bicubic background.
- **Marks:** every output is marked as AI-enhanced:
  - PNG text, XMP (IPTC digital source type `compositeWithTrainedAlgorithmicMedia`) and EXIF;
  - videos: container tags (MP4, MOV, MKV);
  - the model, its checksum, the parameters and the sha256 of the input;
  - no file names and no landmarks.

  `--watermark` adds a visible mark.
- **Videos** are H.264 MP4 by default, with the input's audio. Each frame is restored on its own,
  so the restored faces flicker more than the input does (see `docs/PROJECT.md`).

### Exporting a model to ONNX

```bash
uv run facehall export onnx --scale 4 -o models/x4.onnx   # the x4 reference
uv run facehall export onnx --ckpt runs/<id>/best.safetensors -o models/<name>.onnx
```

The export is checked against PyTorch and refused if the outputs differ by more than 1e-3. The
ONNX file records its source model and checksum in its metadata. `facehall enhance` runs ONNX
models through ONNX Runtime, without PyTorch:

```bash
uv run facehall enhance photo.jpg -o out.png --ckpt models/x4.onnx
uv run facehall enhance photo.jpg -o out.png --scale 8 --set models.backend=onnx   # models/x8.onnx
```

`--device cuda` needs `onnxruntime-gpu` instead of `onnxruntime`.

### A local REST API

`facehall serve` runs the ONNX models of both scales behind a small HTTP API, for local use:

```bash
uv run facehall serve                                   # http://127.0.0.1:8000, try it at /docs
curl --data-binary @photo.jpg "http://127.0.0.1:8000/enhance?scale=4" -o enhanced.png
```

- **The request body is the photo itself**, not a form. The photo is processed in memory: it is
  not written to disk and is not kept after the answer. The log holds no image data, file names,
  face boxes or face positions.
- **The answer** is marked like the output of `facehall enhance`. Its `X-Facehall-Notice` header
  repeats that the faces are a plausible reconstruction.
- **No authentication:** keep it on `127.0.0.1`.
- **Limits:** upload size, output pixels and a short queue (`serve` in
  `configs/pipeline/default.yaml`).
- **No telemetry:** ONNX Runtime sends usage telemetry to Microsoft by default, also on Linux.
  facehall turns it off (`ORT_DISABLE_TELEMETRY=1`) before ONNX Runtime is loaded. FastAPI's
  telemetry is off too.

### Docker

Two images run the same API without PyTorch: `docker/Dockerfile.cpu` and
`docker/Dockerfile.gpu`, which uses the CUDA provider of ONNX Runtime and needs the NVIDIA
Container Toolkit. They are meant to be built and used locally.

```bash
docker build -f docker/Dockerfile.cpu -t facehall:cpu .
docker run --rm -p 127.0.0.1:8000:8000 --read-only --tmpfs /tmp \
    -v "$PWD/models:/models:ro" -v "$FACEHALL_WEIGHTS/yunet:/weights/yunet:ro" facehall:cpu
```

- **No weights in the images:** mount the exported ONNX models at `/models` and the YuNet
  detector at `/weights/yunet`. Both are read-only, and the files must be readable by the
  container's user (uid 10001).
- **Keep the port on `127.0.0.1`.** The service listens on all addresses inside the container,
  and it has no authentication.
- **The GPU image** runs with `--gpus all`, and it stops at start if it finds no GPU.

**Speed** on the development machine (Ryzen 9 9950X, RTX 6000 Ada), one request at a time:

| Request | CPU image | GPU image |
|---|---|---|
| A small photo with one face | 18 ms | 12 ms |
| A photo with 64 small faces | 2.3 s | 2.0 s |
| 1920 × 1080 with 8 faces, ×4, PNG | 2.6 s | 2.6 s |
| 1920 × 1080 with 8 faces, ×8, PNG | 6.9 s | 7.1 s |

- **Detection sets the speed:** most of the time goes to finding the faces, which runs on the
  CPU in both images. The GPU only speeds up the restoration itself.
- **Smaller machines:** limited to 4 cores, the CPU image is at most 8% slower.
- **Method and full numbers:** [`docs/DECISIONS.md`](docs/DECISIONS.md), "Latency and memory of
  the service images (PR 5.5)". The test images were synthetic.

---

## 📊 Results

Numbers from real runs only; every table below is in [`docs/RESULTS.md`](docs/RESULTS.md) or in
[`docs/PROJECT.md`](docs/PROJECT.md) §9 with its full context.

### Models

| Scale | Model | Parameters | Training |
|---|---|---:|---|
| ×4 | SRResNet without BatchNorm, 16 residual blocks | 1,517,571 | L1, then fine-tuned with an identity loss and a perceptual loss, both made robust to pixel patterns |
| ×8 | the same architecture | 1,665,283 | L1, then fine-tuned with the robust identity loss |

Both were trained on the CelebA train split with realistic degradations: blur, resizing,
noise, JPEG and a second degradation stage, as in Real-ESRGAN.

### Synthetic test set

The frozen CelebA test split (19,961 faces of identities never seen in training), degraded at
three levels. **PSNR-Y** (dB, higher is better) / **LPIPS** (lower is better) / **ID similarity**
(cosine of ArcFace embeddings of the output and the original, higher is better):

| Input | ×4 bicubic upscale | ×4 facehall | ×8 bicubic upscale | ×8 facehall |
|---|---|---|---|---|
| clean (bicubic downscale) | 27.25 / 0.341 / 0.711 | 29.18 / 0.141 / 0.682 | 23.42 / 0.546 / 0.316 | 25.37 / 0.262 / 0.312 |
| light | 26.24 / 0.437 / 0.547 | 27.83 / 0.179 / 0.477 | 22.74 / 0.642 / 0.192 | 24.39 / 0.294 / 0.197 |
| medium | 24.57 / 0.548 / 0.358 | 26.25 / 0.222 / 0.313 | 21.38 / 0.689 / 0.088 | 23.05 / 0.333 / 0.119 |
| heavy | 22.83 / 0.630 / 0.198 | 24.55 / 0.261 / 0.196 | 19.89 / 0.716 / 0.043 | 21.48 / 0.381 / 0.073 |

- **Sharper and closer to the original:** much better LPIPS and better PSNR-Y than a bicubic
  upscale on every level.
- **Identity is the hard part:** at ×4 the ID similarity stays below a bicubic upscale on every
  level (within 0.002 on heavy inputs). At ×8 it is above it from light inputs on, but low in
  absolute terms: the output face is mostly invented.

### End to end in a picture

In a real photo, the face must first be found and aligned at the low resolution. On 1,000
synthetic scenes (CelebA validation faces placed in a larger image; degraded: blur, noise, JPEG
70) the whole pipeline with detected faces gives **ID similarity / LPIPS**:

| Scale, input | Bicubic upscale of the image | facehall |
|---|---|---|
| ×4 clean | 0.712 / 0.326 | 0.613 / 0.217 |
| ×4 degraded | 0.438 / 0.505 | 0.367 / 0.292 |
| ×8 clean | 0.340 / 0.544 | 0.261 / 0.347 |
| ×8 degraded | 0.129 / 0.684 | 0.128 / 0.421 |

Cutting the face out at low resolution costs about 0.06 ID at both scales, compared with crops
on the template's pixel grid like those of the test set above. Most of it is information lost in
the resampling. So the faces look much better (LPIPS),
but **they are less like the person than a plain bicubic upscale**, except at ×8 on degraded
inputs, where both are equally low.

### Real low-resolution photos

On 8,645 (×4) and 2,172 (×8) faces from WIDER FACE photos that are natively that small, there
is no reference. FID / KID ×10³ against real CelebA faces (lower is closer to real faces):

| Scale | Bicubic upscale | facehall |
|---|---|---|
| ×4 | 201.2 / 206.8 | 65.8 / 61.9 |
| ×8 | 317.3 / 379.4 | 105.0 / 98.4 |

These metrics reward realistic faces whether or not they are the right person, so they do not
measure faithfulness.

### Speed

See [Docker](#docker): 18 ms for a small photo with one face on the CPU, 2–7 s for a Full HD
frame, mostly spent finding the faces.

### Limitations

- **Invented details:** older people look younger, glasses can disappear, and at ×8 whole faces
  can be drawn from near-featureless inputs (seen in the evaluation grids, not measured).
- **False faces:** the detector also fires on textures. On real photos, 3% (×4) and 15% (×8) of
  the restored detections were not faces, and only 52% / 29% of the small faces were found. Both
  numbers are optimistic: the detector was trained on those photos.
- **Videos** flicker more than the input, because each frame is restored on its own.
- **Subgroups:** ID similarity is lower for faces annotated `Male` = 0 than `Male` = 1 (CelebA's
  binary attribute) on every test level, bicubic upscale included. Skin tone is not measured: CelebA has no such annotation.
- **Training data:** CelebA celebrities, near-frontal and well lit, with JPEG artifacts in the
  targets.

---

## 📂 Data

- **Training and synthetic test:** [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  aligned, with its official identity-disjoint split; non-commercial research only, not
  redistributed. The repository holds the code that prepares it and the checksums
  ([`docs/DATA.md`](docs/DATA.md)), no images or annotations.
- **Real-world test:** native small faces from the WIDER FACE training photos with the
  RetinaFace landmarks; also non-commercial, kept local.
- **The earlier dataset** of this repository (CelebA images mixed with Internet images of unknown
  origin) is not used and no longer linked.

---

## 📐 Background

| Model | Description | Link |
|-------|-------------|------|
| **SRGAN** | Super-Resolution Generative Adversarial Network | [arXiv:1609.04802](https://arxiv.org/abs/1609.04802) |
| **SRResNet** | Residual-based Super-Resolution network | [arXiv:1609.04802](https://arxiv.org/abs/1609.04802) |
| **EDSR** | Residual SR without BatchNorm | [arXiv:1707.02921](https://arxiv.org/abs/1707.02921) |
| **ESRGAN** | Perceptual and relativistic adversarial losses for SR | [arXiv:1809.00219](https://arxiv.org/abs/1809.00219) |
| **Real-ESRGAN** | Enhanced SRGAN with real-world degradation handling | [arXiv:2107.10833](https://arxiv.org/abs/2107.10833) |
| **FSRNet** | Face SR with facial priors | [arXiv:1711.10703](https://arxiv.org/abs/1711.10703) |
| **PULSE** | Self-Supervised Upsampling via Latent Space | [GitHub](https://github.com/alex-damian/pulse) |
| **Attention-FH** | Attention-Aware Face Hallucination | [arXiv:1708.03132](https://arxiv.org/abs/1708.03132) |
| **C-Face** | Identity-aware face hallucination with a face-recognition (compare-face) loss; the "shape control" wording in earlier README versions needs verification | [JAIR 2022, doi:10.1613/JAIR.1.13816](https://doi.org/10.1613/JAIR.1.13816) |

**The original research** (`legacy/FH.ipynb`, read only) compared four approaches on small crops
of the earlier dataset (24×24 → 96×96 for the trained models). PULSE gave realistic faces of other people; the SRGAN generator
did poorly; a simplified SRResNet (FNet) did best; a simplified C-Face (CNet) slightly worse.
Those results are not comparable with the protocol above. FNet and CNet were re-trained on it as
baselines (`docs/RESULTS.md`), and the SRGAN result most likely came from its training recipe
(no L1 pretraining, few epochs), not from the approach.

---

## 📌 Plans

- A prior-guided model (coarse SR → facial landmarks or parsing → fine SR, as in FSRNet), for
  better identity at ×8.
- A landmark error metric (NME) and a skin-tone subgroup.
- Faster face detection, which takes most of the time of a larger photo.

---

## 📜 Licenses

- **Code:** Apache-2.0 ([`LICENSE`](LICENSE)).
- **Data, weights and third-party models** have their own terms, most of them non-commercial:
  [`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md).
