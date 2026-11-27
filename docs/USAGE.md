# Using facehall

> A practical guide to facehall 0.1.0: installation, models, photos, videos, the REST API, the
> Docker images, configuration and troubleshooting. Last updated: **2026-11-27**.
>
> - What the models are, how well they work and their limits: [`MODEL_CARD.md`](MODEL_CARD.md).
> - Training, evaluation and every command of the project: [`PROJECT.md`](PROJECT.md) §12.

**Before you start.** facehall restores **small faces**: about 8 px between the eyes (×4) or
4 px (×8). The output is a plausible reconstruction, not a recovery of the real face. Do not use
it to identify a person or as evidence; show it next to the original. Every output is marked as
AI-enhanced. Use is limited to non-commercial research (`MODEL_CARD.md` §2–3).

## Contents

1. [What you need](#1-what-you-need)
2. [Install](#2-install)
3. [Get the models](#3-get-the-models)
4. [Photos](#4-photos)
5. [Videos](#5-videos)
6. [Reading the marks](#6-reading-the-marks)
7. [The REST API](#7-the-rest-api)
8. [Docker](#8-docker)
9. [Configuration](#9-configuration)
10. [Troubleshooting](#10-troubleshooting)

## 1. What you need

| What | Needed for | Notes |
|---|---|---|
| Linux, x86-64 | everything | Developed and tested on Linux only |
| [uv](https://docs.astral.sh/uv/) | installing and running | uv installs Python 3.12 itself |
| The restoration models | restoring faces | Not published: see [§3](#3-get-the-models) |
| YuNet face detector (230 kB) | finding faces | Downloaded by a script, [§3.1](#31-the-face-detector) |
| NVIDIA GPU | optional | Faster restoration only (see [§4.6](#46-speed)). The PyTorch build needs a driver for CUDA 13.2 |
| `ffmpeg` | videos only | e.g. `sudo apt install ffmpeg` |
| Docker | the images only | plus the NVIDIA Container Toolkit for the GPU image |

## 2. Install

```bash
git clone https://github.com/dronja/robodream_cv.git && cd robodream_cv
uv sync                      # with PyTorch for CUDA (Linux)
uv sync --extra cpu          # or: CPU-only PyTorch, much smaller; then `uv run --extra cpu ...`
uv run facehall --version    # facehall 0.1.0
```

Every command below runs inside the repository with `uv run`.

## 3. Get the models

What each use needs:

| Use | Restoration model | Detector |
|---|---|---|
| `facehall enhance` (default, PyTorch) | `runs/<run>/best.safetensors` and the `best.json` next to it | YuNet |
| `facehall enhance` with ONNX, `facehall serve`, Docker | `models/x4.onnx`, `models/x8.onnx` | YuNet |

### 3.1 The face detector

```bash
uv run python scripts/download_weights.py yunet    # into $FACEHALL_WEIGHTS/yunet/
```

Set `FACEHALL_WEIGHTS` to the folder you want, for example `export
FACEHALL_WEIGHTS=$HOME/facehall-weights`; the default is the development machine's
`/mnt/storage/AI/Face/weights`. The file is checked against a pinned sha256. YuNet is MIT
licensed (`THIRD_PARTY_LICENSES.md` §3).

### 3.2 The restoration models

The trained weights are not published: they are derived from CelebA, whose terms allow
non-commercial research only (D39). There are two ways to get them.

**A. You have copies of the reference runs** (for example from the machine that trained them):

```bash
# the two run folders, or at least best.safetensors and best.json of each:
#   runs/srresnet_id_perc_robust_x4-20261126-121325/   (x4)
#   runs/srresnet_id_robust_x8-20261125-174436/        (x8)
export FACEHALL_RUNS=/path/to/runs      # if they are not in ./runs
sha256sum -c docs/models.sha256         # run in the repository root: the versions of release 0.1.0
```

**B. You train them yourself.** This needs CelebA and about 70 minutes on an RTX 6000 Ada:

1. Download the 20 parquet shards of the Hugging Face dataset `eurecom-ds/celeba` (8.92 GB),
   only if you accept the CelebA terms. Their checksums are in [`DATA.md`](DATA.md) §2.
2. Prepare the dataset (about 2 minutes):

   ```bash
   uv run python scripts/prepare_dataset.py --src /path/to/celeba_parquet --out /data/celeba
   export FACEHALL_DATA=/data            # the folder that holds celeba/
   ```

3. Download the networks the losses use. The facenet weights state no license
   (`THIRD_PARTY_LICENSES.md` §3), so check whether you may use them first.

   ```bash
   uv run python scripts/download_weights.py vgg19 facenet_vggface2
   ```

4. Train: first a short check, then the L1 start and the fine-tuning of each scale. Times are
   from an RTX 6000 Ada.

   ```bash
   uv run facehall train --config configs/train/srresnet_x4.yaml --smoke    # seconds
   uv run facehall train --config configs/train/srresnet_x4.yaml            # ~27 min
   uv run facehall train --config configs/train/srresnet_id_perc_robust_x4.yaml \
       --set train.init=runs/<your srresnet_x4 run>/best.safetensors        # ~9 min
   uv run facehall train --config configs/train/srresnet_x8.yaml            # ~27 min
   uv run facehall train --config configs/train/srresnet_id_robust_x8.yaml \
       --set train.init=runs/<your srresnet_x8 run>/best.safetensors        # ~7 min
   ```

   Every run fits a GPU with 8 GB. An interrupted run continues with
   `uv run facehall train --resume runs/<run>`.
5. Point the tools at your runs, either per command with `--ckpt runs/<run>/best.safetensors`, or
   for `--scale` with `--set models.x4=runs/<run>/best.safetensors` (and `models.x8`).

Your own runs are new weights, so their checksums differ from `docs/models.sha256`, and their
quality is not the measured one until you evaluate them (`PROJECT.md` §12, `facehall eval`).

### 3.3 ONNX exports

The REST API and the Docker images run ONNX files, without PyTorch:

```bash
uv run facehall export onnx --scale 4 -o models/x4.onnx
uv run facehall export onnx --scale 8 -o models/x8.onnx
uv run facehall export onnx --ckpt runs/<run>/best.safetensors -o models/<name>.onnx   # any run
```

Each export is compared with PyTorch and refused if they differ by more than 1e-3. Add
`--overwrite` to replace a file.

## 4. Photos

### 4.1 Basic use

```bash
uv run facehall enhance photo.jpg -o out.png --scale 4           # one photo
uv run facehall enhance photos/ -o enhanced/ --scale 8           # a folder, mirrored
uv run facehall enhance a.jpg b.png -o enhanced/ --format jpg --quality 92
```

- **Output:** the whole image upscaled ×4 or ×8 (bicubic), with every restored face pasted in.
  A 1000 × 800 photo becomes 4000 × 3200 at ×4. Outputs above 200 million pixels are refused.
- **Formats:** PNG by default; `--format jpg` or `webp` for a folder; for one file the suffix of
  `-o` decides. Inputs: PNG, JPEG, WebP, BMP, TIFF.
- **Existing outputs** are not replaced without `--overwrite`. A failed file does not stop the
  others.
- **`--device`:** `auto` (a GPU when the installed runtime can use one), `cpu` or `cuda`.

For each input it prints a line like:

```
photo.jpg -> out.png: 65 faces, 64 restored, 1 too small for the model
```

followed by the notice that the faces are a plausible reconstruction.

### 4.2 Which scale

The scale is decided by the size of the faces, not by how large you want the picture:

| Distance between the eyes in the photo | Scale |
|---|---|
| about 4–16 px | ×4 (best near 8 px) |
| about 2–8 px | ×8 (best near 4 px) |
| more than 16 px | none: such faces are left as they are (upscaled bicubically) |

A face more than twice as large or as small as the model input is reported as `too large for the
model` or `too small for the model` and left to the bicubic background. For normal portraits
facehall therefore changes nothing but the size.

### 4.3 Faces the detector misses

```bash
uv run facehall enhance photo.jpg -o out.png --scale 4 --box 120,80,18,22
uv run facehall enhance photo.jpg -o out.png --scale 4 --landmarks 124,88,132,88,128,93,125,97,131,97
```

- **`--box LEFT,TOP,WIDTH,HEIGHT`** in photo pixels, tight like a detector's box: cheek to cheek
  and forehead to chin, about 2.2 eye distances wide. The detector first looks inside the box at
  a lower threshold; if it finds nothing, the landmarks come from the box.
- **`--landmarks`:** five points: the eye centres, the nose tip and the mouth corners, in photo
  pixels.
- **One input image** only; both options can be repeated for several faces.
- **Warning:** a box without a face in it still gets a face drawn into it.

### 4.4 Options

| Option | Effect |
|---|---|
| `--watermark` | a visible "AI-enhanced" mark on the image |
| `--set color_correction=true` | match the colours of each restored face to the input |
| `--set detector.score_threshold=0.9` | fewer false faces, but more faces missed (default 0.8) |
| `--set models.backend=onnx` | use `models/x4.onnx` / `x8.onnx` instead of the checkpoints (no PyTorch) |
| `--ckpt FILE` | any model: a checkpoint (`.safetensors`) or an ONNX export (`.onnx`) |
| `--pipeline FILE` | your own pipeline config ([§9](#9-configuration)) |

On ONNX, `--device cuda` needs `onnxruntime-gpu` instead of `onnxruntime` in the environment.

### 4.5 Check the result

Look at every output before you use it:
- **Changed identity:** faces can look younger, lose glasses, or look like someone else,
  especially at ×8.
- **False faces:** a face may be drawn on a texture (a tie, a button, printed letters). On real
  photos, about 3% (×4) and 15% (×8) of the restored detections were not faces.
- **The record** lists each restored region ([§6](#6-reading-the-marks)).

### 4.6 Speed

On a Ryzen 9 9950X with an RTX 6000 Ada:
- a small photo with one face takes about 20 ms on the CPU;
- a Full HD frame with 8 faces takes 2–7 s.

Most of that time goes to finding the faces, on the CPU, with or without a GPU (`PROJECT.md` §9,
PR 5.5). A 4-core machine is at most 8% slower.

## 5. Videos

```bash
uv run facehall enhance clip.mp4 -o clip_x4.mp4 --scale 4
uv run facehall enhance videos/ -o out/ --scale 8 --video-format mkv
```

- **Needs** the `ffmpeg` program on `PATH`.
- **Inputs:** `.mp4 .mov .mkv .avi .webm .m4v`, also inside folders.
- **Output:** H.264 (CRF 18), MP4 by default, with the input's audio. The marks are written as
  container tags.
- **Processing:** each frame is restored on its own. Face tracks smooth the landmarks between
  frames, and a face is restored only in frames where it is detected.
- **Expect flicker:** the restored faces flicker more than the input, even with perfect
  landmarks.
- **Speed:** about 8 frames/s for 640 × 360 at ×4 with about 60 faces per frame (RTX 6000 Ada).
- **Settings:** `video` in the pipeline config ([§9](#9-configuration)), for example
  `--set video.crf=23` for smaller files or `--set video.audio=false`.

## 6. Reading the marks

Every output says that it is AI-enhanced:

| Format | Marks |
|---|---|
| PNG | text chunks `AI-enhanced`, `Description`, `Software`, `facehall` (the record); XMP; EXIF |
| JPEG, WebP | XMP and EXIF |
| Videos | container tags `ai_enhanced`, `description`, `digital_source_type`, `facehall` |

- **XMP** carries the IPTC digital source type `compositeWithTrainedAlgorithmicMedia`, which
  XMP-aware tools show.
- **The record** holds:
  - the notice and the facehall version;
  - the model: its name, the sha256 of its weights and, for ONNX, of the file;
  - the parameters;
  - the sha256 and size of the input;
  - for each face: how it was found, its score, whether it was restored, and the changed region.

  It holds no file names, landmarks or image content.

Read the record of an image:

```bash
uv run python -c "from facehall import marking; import json, sys; \
print(json.dumps(marking.read_record(sys.argv[1]), indent=1))" out.png
```

The `model.sha256` and `model.onnx_sha256` values identify the model version
(`MODEL_CARD.md` §12). Re-encoding the image in another program may drop the marks.

## 7. The REST API

```bash
uv run facehall serve                      # http://127.0.0.1:8000, both scales, ONNX models
uv run facehall serve --scale 4 --port 9000 --device cpu
```

- **Models:** it loads `models/x4.onnx` and `models/x8.onnx` (or `$FACEHALL_MODELS/...`) and
  warms them up. It answers within about a second on the CPU and two on a GPU.
- **No authentication:** keep it on `127.0.0.1`. Bound to another address, anyone who can reach
  it can send photos, and the service only logs a warning.
- **Try it by hand:** at `http://127.0.0.1:8000/docs`. That page loads its script from a CDN.

### 7.1 Endpoints

| Endpoint | Answer |
|---|---|
| `POST /enhance` | the enhanced photo |
| `GET /health` | `{"status": "ok", "scales": [4, 8], "pending": 0}` |
| `GET /version` | the facehall version, the models with their checksums and device, the limits, the notice |

`POST /enhance` takes **the photo's bytes as the request body**, not a form. Query parameters:

| Parameter | Values | Default |
|---|---|---|
| `scale` | `4` or `8` | the smallest scale served |
| `format` | `png`, `jpeg`, `webp` | `png` |
| `quality` | 1–100 (JPEG and WebP) | 95 |
| `watermark` | `true` / `false` | `false` |
| `box` | `LEFT,TOP,WIDTH,HEIGHT`, repeatable | detection |

```bash
curl --data-binary @photo.jpg "http://127.0.0.1:8000/enhance?scale=4" -o enhanced.png
curl --data-binary @photo.jpg \
    "http://127.0.0.1:8000/enhance?scale=8&format=jpeg&box=120,80,18,22" -o enhanced.jpg
curl -s -D - -o /dev/null --data-binary @photo.jpg "http://127.0.0.1:8000/enhance"   # headers only
```

From Python, without extra packages:

```python
import urllib.request

with open("photo.jpg", "rb") as f:
    request = urllib.request.Request(
        "http://127.0.0.1:8000/enhance?scale=4", data=f.read(), method="POST"
    )
with urllib.request.urlopen(request, timeout=150) as answer:
    print(answer.headers["X-Facehall-Faces"])  # e.g. "3 faces, 3 restored"
    open("enhanced.png", "wb").write(answer.read())
```

**Answer headers:**
- `X-AI-Enhanced: true`;
- `X-Facehall-Notice`: the plausible-reconstruction notice;
- `X-Facehall-Model`: the model and its scale;
- `X-Facehall-Faces`: the face counts, as printed by `facehall enhance`.

The image carries the same marks as the command's outputs.

**Errors:**

| Status | Meaning |
|---|---|
| 400 | bad parameters, a scale that is not served, or a photo that cannot be decoded |
| 413 | the upload is larger than 25 MB, or the output would exceed 2^27 pixels (a 4K photo at ×4, a 1080p one at ×8) |
| 415 | not a supported image, or a form upload (use `--data-binary`) |
| 503 | busy: the queue of 8 is full, or the answer took more than 120 s; retry after the `Retry-After` seconds |

The limits are in `serve` of the pipeline config, for example `--set serve.queue=2`.

**Privacy:**
- Uploads are processed in memory and dropped with the answer; nothing is written to disk.
- The log holds one line per request (method, path, status, time), without images, file names,
  boxes or face positions.
- ONNX Runtime's and FastAPI's telemetry are off.

## 8. Docker

Two images run the REST API without PyTorch: `facehall:cpu` (about 0.7 GB) and `facehall:gpu`
(about 4.6 GB, CUDA through ONNX Runtime). They hold no weights: you mount them.

```bash
docker build -f docker/Dockerfile.cpu -t facehall:cpu .   # about 40 s the first time
docker build -f docker/Dockerfile.gpu -t facehall:gpu .   # about 8 min the first time

# the container runs as uid 10001: the mounted files, and the folders above them, must be
# readable by it
chmod 755 models "$FACEHALL_WEIGHTS/yunet" && chmod 644 models/*.onnx "$FACEHALL_WEIGHTS"/yunet/*

docker run --rm -p 127.0.0.1:8000:8000 --read-only --tmpfs /tmp \
    -v "$PWD/models:/models:ro" -v "$FACEHALL_WEIGHTS/yunet:/weights/yunet:ro" facehall:cpu
docker run --rm --gpus all -p 127.0.0.1:8000:8000 --read-only --tmpfs /tmp \
    -v "$PWD/models:/models:ro" -v "$FACEHALL_WEIGHTS/yunet:/weights/yunet:ro" facehall:gpu
```

- **Ports:** always publish the port on `127.0.0.1`, as above. Inside the container the service
  listens on all addresses, and it has no authentication.
- **Use:** the same `curl` commands as in [§7](#7-the-rest-api), about a second (CPU) or two
  (GPU) after the start. Docker's own health check runs every 30 s, so `docker ps` shows
  `healthy` after the first check.
- **GPU image:** needs `--gpus all` and the NVIDIA Container Toolkit. Without a GPU it stops at
  start with "could not start the CUDA provider".
- **Options:** give the whole command after the image name; keep `--host 0.0.0.0` inside the
  container, for example `... facehall:cpu facehall serve --host 0.0.0.0 --port 8000 --scale 4`.
- **Self-check:** `uv run python scripts/smoke_image.py --image facehall:cpu --models models`.
  It checks the answer, the marks, the user and the logs.

## 9. Configuration

All settings are in `configs/pipeline/default.yaml`. Change one per command with
`--set KEY=VALUE` (repeatable), or copy the file and pass `--pipeline my.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `detector.score_threshold` | 0.8 | detection score for a face; lower finds more faces and more false ones |
| `detector.box_score_threshold` | 0.3 | the same inside a `--box` |
| `max_downscale`, `max_upscale` | 2.0 | how much larger or smaller than the model input a face may be |
| `max_output_pixels` | 200,000,000 | largest output of `facehall enhance` |
| `color_correction` | false | match the colours of each restored face to the input |
| `models.backend` | torch | `torch` (checkpoints) or `onnx` (exports) for `--scale` |
| `models.x4`, `models.x8` | the reference runs | checkpoints for `--scale` |
| `models.onnx_x4`, `models.onnx_x8` | `models/x4.onnx`, `models/x8.onnx` | ONNX files for `--scale` and `serve` |
| `video.*` | see the file | tracking, codec (`libx264`, `libx265`), `crf`, `preset`, `audio` |
| `serve.*` | see the file | host, port, scales, upload and output limits, queue, timeout |

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `FACEHALL_WEIGHTS` | `/mnt/storage/AI/Face/weights` | third-party weights (YuNet and the training and metric networks) |
| `FACEHALL_RUNS` | `runs` | the folder of the reference runs |
| `FACEHALL_MODELS` | `models` | the folder of the ONNX exports |
| `FACEHALL_DATA` | `/mnt/storage/AI/Face` | datasets, for training and evaluation |
| `ORT_DISABLE_TELEMETRY` | set to `1` by facehall | ONNX Runtime telemetry; an explicit value is kept |

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| An error naming `face_detection_yunet_2026may.onnx` or a `best.safetensors` / `.onnx` file | The weights are not where the config looks: [§3](#3-get-the-models), `FACEHALL_WEIGHTS`, `FACEHALL_RUNS`, `FACEHALL_MODELS` |
| `0 faces` | The detector found nothing: try the other scale, or give a `--box` |
| `too large for the model` | The face is larger than the model handles: use ×4 instead of ×8, or leave it; such faces need no restoration |
| `too small for the model` | Use ×8 instead of ×4 |
| Faces drawn where there are none | False detections: raise `detector.score_threshold`, or give the faces with `--box` |
| `this onnxruntime has no CUDA provider` | `--device cuda` with ONNX needs `onnxruntime-gpu`; or use `--device cpu` |
| `video needs the ffmpeg program on PATH` | Install `ffmpeg` |
| API answers 415 | Send the bytes: `curl --data-binary @photo.jpg`, not `-F` |
| API answers 413 | A smaller photo, a smaller scale, or a higher `serve.max_output_pixels` (more memory) |
| API answers 503 | Busy; retry later, or raise `serve.queue` |
| Container: `Permission denied: /models/...` | Make the mounted files readable by uid 10001 ([§8](#8-docker)) |
| GPU container stops at start | Run it with `--gpus all`; check the NVIDIA Container Toolkit |
