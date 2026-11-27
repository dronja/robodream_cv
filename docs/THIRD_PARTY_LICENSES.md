# Third-party licenses

> This is not legal advice. Last reviewed: **2026-11-24**.
>
> - Package licenses come from the installed package metadata, generated with
>   `scripts/license_report.py` and reviewed by hand.
> - Dataset and model terms come from their official pages.
> - Every PR that adds a dependency, a dataset or weights updates this file before merge
>   (`CLAUDE.md` §8). Review the whole file again before every release.

## 1. This project

The code is licensed under Apache-2.0 (`LICENSE`). Trained weights are a different matter: the
CelebA terms also cover "derived data" (§2). Until that is reviewed, weights trained on CelebA,
including the legacy weights, are treated as **non-commercial research only** and are not covered
by the code license.

## 2. Datasets

| Dataset | Used for | Terms | Checked |
|---|---|---|---|
| CelebA: aligned images, identities, landmarks, attributes | Training and evaluation, Phases 1–3 | "available for non-commercial research purposes only"; "You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data"; "You agree not to further copy, publish or distribute any portion of the CelebA dataset", except for copies for internal use at a single site within the same organization | [Official page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 2026-11-23 |
| Hugging Face mirror `eurecom-ds/celeba` | Download source for CelebA | No license is stated in the dataset card or in the Hub metadata; the CelebA terms above apply | Hub API, 2026-11-23 |
| WIDER FACE: training images, boxes and attributes | Real-world test set (D20) | Official page, section "License": "Creative Common License (cc by-nc-nd)". The same page: "images are selected from the publicly available WIDER dataset". The page links to the Hugging Face dataset `CUHK-CSE/wider_face`, whose Hub metadata states `cc-by-nc-nd-4.0` | [Official page](http://shuoyang1213.me/WIDERFACE/) and Hub API, 2026-11-24 |
| RetinaFace annotations of WIDER FACE (`retinaface_gt.zip`: boxes and five landmarks) | Alignment of the real-world test set (D20) | No license file comes with the annotations. The [InsightFace README](https://github.com/deepinsight/insightface), section "License": "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." The [RetinaFace README](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) links the download | READMEs, 2026-11-24 |

Consequence for this public repository: CelebA images, annotations and per-image manifests are not
committed (D14); `docs/DATA.md` holds only counts and checksums. Public figures that show CelebA
faces need a separate decision (`docs/PROJECT.md` §11, Q2 and Q3). The same applies to WIDER FACE:
its crops, the manifest and any figure that shows its faces stay local. "NoDerivatives" forbids
sharing adapted material such as crops, and the people in the photos gave no consent to this use
(D20).

## 3. Pretrained weights

| Weights | Shipped by | Status |
|---|---|---|
| LPIPS linear layers (`alex.pth`; `vgg.pth` and `squeeze.pth` are not used) | Bundled in torchmetrics 1.9.0 (`torchmetrics/functional/image/lpips_models/`). Used by the LPIPS metric (PR 1.7) | torchmetrics' LPIPS code cites the LPIPS project, [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity). Its [LICENSE](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE) is BSD-style with two conditions (BSD 2-Clause; the SPDX name is not stated): "Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang". That the bundled files are exactly the LPIPS release weights *needs verification*. Checked 2026-11-24 |
| LPIPS backbone: torchvision `AlexNet_Weights.DEFAULT` (`alexnet-owt-7be5be79.pth`, 233 MB, ImageNet) | Downloaded by torchvision into the torch hub cache (`~/.cache/torch/hub/checkpoints/`) on first use (D11) | [torchvision docs](https://docs.pytorch.org/vision/stable/models.html): "The pre-trained models provided in this library may have their own licenses or terms and conditions derived from the dataset used for training. It is your responsibility to determine whether you have permission to use the models for your use case." The ImageNet terms *need verification*. Used for non-commercial research only (D2). Checked 2026-11-24 |
| InsightFace `buffalo_l` recognition model `w600k_r50.onnx` (ArcFace, ResNet50 trained on WebFace600K) | ID-similarity metric (PR 1.7, D18). Downloaded by `scripts/download_weights.py` from `github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip`: 288,621,354 bytes, sha256 `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`. Only `w600k_r50.onnx` is kept (sha256 `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`), in `$FACEHALL_WEIGHTS/buffalo_l/` | **Non-commercial research only.** [README](https://github.com/deepinsight/insightface): "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." [python-package README](https://github.com/deepinsight/insightface/tree/master/python-package): "The pretrained models provided with this library are for non-commercial research only, whether downloaded automatically or manually." **Open point:** the README's "2025-11-24 Update" adds "For open-sourced face recognition models (e.g., buffalo_l package), please contact recognition-oss-pack@insightface.ai for licensing." It does not say whether research use needs that separate license. The user chose to use the model for non-commercial research (D2) and record this point (D18). The InsightFace code (MIT) is not used. Checked 2026-11-24 |
| InceptionV3 for FID and KID (`weights-inception-2015-12-05-6726825d.pth`, 95,628,359 bytes, sha256 `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`) | Real-world evaluation (PR 2.5). Downloaded by `scripts/download_weights.py inception` from the torch-fidelity v0.2.0 GitHub release into `$FACEHALL_WEIGHTS/inception/`; loaded through `torchmetrics.image.fid.NoTrainInceptionV3`. Download approved by the user on 2026-11-24 (D21) | torch-fidelity 0.4.0 (`feature_extractor_inceptionv3.py`): "InceptionV3 weights converted from the official TensorFlow weights"; "Original weights distributed under Apache License 2.0: https://github.com/tensorflow/models/blob/master/LICENSE". The TensorFlow LICENSE itself was not opened. Trained on ImageNet: the ImageNet terms *need verification*, as for AlexNet. Used for non-commercial research only (D2). Checked 2026-11-24 |
| VGG19 for the perceptual loss: torchvision `VGG19_Weights.IMAGENET1K_V1` (`vgg19-dcbb9e9d.pth`, 574,673,361 bytes, sha256 `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44`) | Training with the ESRGAN recipe (PR 3.2). Downloaded by `scripts/download_weights.py vgg19` from `download.pytorch.org` into `$FACEHALL_WEIGHTS/vgg19/`; the feature layers up to `conv5_4` are used. Download approved by the user on 2026-11-25 | The same torchvision statement as for AlexNet above: the terms derive from the training data. Trained on ImageNet: the ImageNet terms *need verification*. Used for non-commercial research only (D2). Checked 2026-11-25 |
| facenet Inception-ResNet-v1 for the identity loss: model `20180402-114759` trained on VGGFace2, as ported to PyTorch by facenet-pytorch (`20180402-114759-vggface2.pt`, 111,898,327 bytes, sha256 `281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b`) | Training with the identity loss (PR 3.3). Downloaded by `scripts/download_weights.py facenet_vggface2` from the facenet-pytorch v2.2.9 GitHub release into `$FACEHALL_WEIGHTS/facenet_vggface2/`; the classification layer is not used. The network code is re-implemented in `facehall.losses.facenet` with the facenet-pytorch MIT notice. Download approved by the user on 2026-11-25, with the weight license unknown | **No license is stated for the weights.** The code of both projects is MIT: [facenet-pytorch](https://github.com/timesler/facenet-pytorch) ("Copyright (c) 2019 Timothy Esler") and [davidsandberg/facenet](https://github.com/davidsandberg/facenet) ("Copyright (c) 2016 David Sandberg"). The facenet-pytorch README: "Pytorch model weights were initialized using parameters ported from David Sandberg's tensorflow facenet repo". Neither README says under which terms the pretrained models are released. The [VGGFace2 page](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) states no license and says "The download links for the VGGFace2 dataset are no longer available from this website"; the dataset terms *need verification*. Used for non-commercial research only (D2); not to be distributed with any release without a separate review. Checked 2026-11-25 |
| YuNet face detector `face_detection_yunet_2026may.onnx` (229,738 bytes, sha256 `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`) | Face detection with five landmarks (Phase 4, D31). Downloaded by `scripts/download_weights.py yunet` from the [opencv_zoo](https://github.com/opencv/opencv_zoo) repository at commit `26cc381e4d2594bb9f47a26eb8fd96c94a13660d` (git LFS, `media.githubusercontent.com`) into `$FACEHALL_WEIGHTS/yunet/`; run through `cv2.FaceDetectorYN` of the existing OpenCV dependency. Download approved by the user on 2026-11-26 | **MIT.** [README](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet): "All files in this directory are licensed under MIT License." [LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE): "MIT License", "Copyright (c) 2020 Shiqi Yu". Trained with [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train) (BSD-3-Clause per its repository page; the LICENSE file was not opened): "YuNet face detection training on WIDER Face", and "The `labelv2` annotations come from SCRFD". So the training data include WIDER FACE (CC BY-NC-ND 4.0, §2) and landmark annotations from InsightFace's SCRFD (non-commercial research only, as for `buffalo_l`). Whether these terms bind the released weights *needs verification*. Used for non-commercial research (D2). Checked 2026-11-26 |
| Legacy weights (`legacy/*.keras`, `legacy/*.pth`) | This project (legacy notebook) | Trained on the legacy dataset, which includes CelebA, so treated as non-commercial research only (§1) |

## 4. Python runtime dependencies

Dev-only tools (ruff, mypy, pytest, pre-commit) are not distributed and are not listed.

### 4.1 Direct dependencies

| Package | Version | License | Notes |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | |
| omegaconf | 2.4.0 | BSD License | From classifiers |
| onnx | 1.23.2 | Apache-2.0 | Exporting models to ONNX only (`facehall export onnx`, PR 5.1); adds ml-dtypes (Apache-2.0). Inference reads ONNX files with onnxruntime alone (D34) |
| onnxruntime | 1.32.0 | MIT License | Runs the recognition model (PR 1.7). Adds flatbuffers and protobuf. CPU build; [PyPI](https://pypi.org/project/onnxruntime/) also states MIT |
| onnxscript | 0.7.2 | MIT License | Needed by PyTorch's dynamo ONNX exporter (PR 5.1); adds onnx-ir (Apache-2.0) |
| opencv-python-headless | 5.1.0.95 | Apache 2.0 | The wheel bundles OpenCV (Apache-2.0), packaging code (MIT), libvpx (BSD-style) and **FFmpeg (LGPL-2.1)**; see `LICENSE-3RD-PARTY.txt` in the wheel. The LGPL obligations apply to distribution; none is planned (D13, local use only) |
| pillow | 13.0.0 | MIT-CMU | |
| pyarrow | 26.0.1 | Apache-2.0 | |
| safetensors | 0.8.0 | Apache Software License | From classifiers |
| torch | 2.15.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib | Same metadata for the `+cpu` build |
| torchmetrics[image] | 1.9.0 | Apache-2.0 | The `image` extra adds scipy and torch-fidelity; bundles LPIPS weights (§3) |
| torchvision | 0.29.2 | BSD | Same metadata for the `+cpu` build |

### 4.2 CUDA build

On Linux, PyPI torch (the default, and the `cuda` extra) installs the NVIDIA CUDA libraries
(`nvidia-*`) under **NVIDIA proprietary licenses** (`LicenseRef-NVIDIA-Proprietary`). It also
installs the `cuda-toolkit` meta-package, which states no license and only pulls in those NVIDIA
packages. The product is used locally only (D13), so no GPU Docker image is published; review the
redistribution terms if that changes. The CPU build (`cpu` extra, used in CI) contains no NVIDIA
packages.

### 4.3 Full list (generated)

Generated from the `cuda` export, which is the superset of both builds. Manual corrections are
marked with †.

| Package | Version | License (from package metadata) |
|---|---|---|
| colorama | 0.4.6 | BSD License † (PyPI classifier; installed on Windows only) |
| cuda-bindings | 13.5.0 | Apache-2.0 |
| cuda-pathfinder | 1.9.1 | Apache-2.0 |
| cuda-toolkit | 13.2.2 | Not stated † (meta-package for the NVIDIA packages below) |
| filelock | 4.2.4 | MIT |
| flatbuffers | 25.12.19 | Apache 2.0 |
| fsspec | 2026.9.0 | BSD-3-Clause |
| jinja2 | 3.1.6 | BSD License |
| lightning-utilities | 0.15.3 | Apache-2.0 |
| markupsafe | 3.0.4 | BSD-3-Clause |
| ml-dtypes | 0.7.0 | Apache-2.0 |
| mpmath | 1.3.0 | BSD |
| networkx | 3.7 | BSD-3-Clause |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-cublas | 13.4.1.3 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-cccl | 13.3.5.0.6 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-cupti | 13.2.86 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-nvrtc | 13.2.86 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-runtime | 13.2.86 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cudnn-cu13 | 9.26.0.51 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cufft | 12.2.0.57 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cufile | 1.17.1.22 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cusolver | 12.2.0.11 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cusparse | 12.7.10.12 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cusparselt-cu13 | 0.8.1 | NVIDIA Proprietary Software |
| nvidia-nccl-cu13 | 2.30.7 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvjitlink | 13.5.37 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvshmem-cu13 | 3.7.2 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvtx | 13.2.86 | LicenseRef-NVIDIA-Proprietary |
| omegaconf | 2.4.0 | BSD License |
| onnx | 1.23.2 | Apache-2.0 |
| onnx-ir | 1.1.0 | Apache-2.0 |
| onnxruntime | 1.32.0 | MIT License |
| onnxscript | 0.7.2 | MIT License |
| opencv-python-headless | 5.1.0.95 | Apache 2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pillow | 13.0.0 | MIT-CMU |
| protobuf | 7.36.2 | 3-Clause BSD License |
| pyarrow | 26.0.1 | Apache-2.0 |
| pyyaml | 6.0.3 | MIT |
| safetensors | 0.8.0 | Apache Software License |
| scipy | 1.18.1 | BSD License |
| setuptools | 84.2.1 | MIT |
| sympy | 1.14.0 | BSD |
| torch | 2.15.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib |
| torch-fidelity | 0.4.0 | Apache License 2.0 |
| torchmetrics | 1.9.0 | Apache-2.0 |
| torchvision | 0.29.2 | BSD |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| triton | 3.9.0 | MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |

### 4.4 External programs

These are not Python dependencies: they are found on `PATH` and run as separate processes, and
nothing is linked against them or distributed (D13).

| Program | Used for | Version checked | License |
|---|---|---|---|
| `ffmpeg`, `ffprobe` | Videos only (`facehall.video`, PR 4.5): encoding the restored frames (libx264 or libx265), copying the audio and writing the marks; `ffprobe` reads the marks back. CI installs the same Ubuntu package on its runner for the video tests (PR #33) | Ubuntu package `7:8.0.1-3ubuntu2` (here and on the CI runner) | FFmpeg is LGPL-2.1+, but this build is configured with `--enable-gpl` and links libx264 and libx265 (both GPL-2.0+), so the program as built is GPL. That matters for distributing it, which is not planned; a Docker image with it (Phase 5) would need this reviewed |

## 5. How to update

```bash
uv export --no-dev --no-hashes --no-emit-project --extra cuda --format requirements.txt \
    | uv run python scripts/license_report.py
```

Replace §4.3 with the output, then:
- re-apply the † corrections;
- update §4.1 when direct dependencies change;
- look up every `UNKNOWN` or `not installed` row by hand.
