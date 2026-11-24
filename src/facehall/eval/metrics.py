"""Per-image metrics for SR vs HR (docs/PROJECT.md §9, PR 1.7).

Inputs are (B, 3, H, W) RGB tensors in [0, 1]; every metric returns one value per image, shape
(B,), so results can be split by subgroup afterwards.

- ``psnr_y`` and ``ssim_y`` use the luma Y of ITU-R BT.601 in [16, 235] (``rgb_to_y``), the
  convention of the SR literature (MATLAB ``rgb2ycbcr``), with no border crop.
- ``Lpips``: LPIPS (arXiv:1801.03924) from torchmetrics, AlexNet backbone by default.
- ``IdentitySimilarity``: cosine similarity of face-recognition embeddings of SR and HR.
  ``ArcFaceOnnx`` runs the InsightFace ``buffalo_l`` recognition model (D18).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure

from facehall.align import umeyama, warp
from facehall.align.similarity import Matrix, Points

# InsightFace face_align.arcface_dst: the 5-point template of the 112x112 recognition crop
# (eyes, nose, mouth corners), in OpenCV pixel-centre coordinates.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ]
)
ARCFACE_SIZE = 112
Y_WEIGHTS = (65.481, 128.553, 24.966)  # BT.601 for RGB in [0, 1] -> Y in [16, 235]
PSNR_MAX_MSE_FLOOR = 1e-10  # identical images give 10 * log10(255^2 / 1e-10) ~ 148 dB


def rgb_to_y(images: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) RGB in [0, 1] -> (B, 1, H, W) luma in [16, 235]."""
    weights = images.new_tensor(Y_WEIGHTS).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True) + 16.0


def psnr_y(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """PSNR on Y with peak 255, per image."""
    mse = (rgb_to_y(sr) - rgb_to_y(hr)).square().flatten(1).mean(dim=1)
    return 10.0 * torch.log10(255.0**2 / mse.clamp_min(PSNR_MAX_MSE_FLOOR))


def ssim_y(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """SSIM on Y (Gaussian window 11, sigma 1.5, K1 0.01, K2 0.03, range 255), per image."""
    value = structural_similarity_index_measure(
        rgb_to_y(sr), rgb_to_y(hr), data_range=255.0, reduction="none"
    )
    assert isinstance(value, torch.Tensor)
    return value


class Lpips:
    """LPIPS per image (lower is better), on inputs in [0, 1].

    Uses the network inside torchmetrics' ``LearnedPerceptualImagePatchSimilarity``: the LPIPS
    linear weights ship with torchmetrics, the backbone weights come from torchvision.
    """

    def __init__(self, net: str = "alex", device: torch.device | str = "cpu") -> None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True)  # type: ignore[arg-type]
        self.net: nn.Module = metric.net.to(device).eval()
        self.name = net

    @torch.no_grad()
    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor = self.net(sr, hr, normalize=True)
        return value.flatten()


class Embedder(Protocol):
    """Face embeddings of aligned crops: (B, 112, 112, 3) uint8 RGB -> (B, D) float."""

    def __call__(self, faces: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating]: ...


class ArcFaceOnnx:
    """The InsightFace ``buffalo_l`` recognition model (``w600k_r50.onnx``, ResNet50 trained
    on WebFace600K) through ONNX Runtime.

    Preprocessing follows InsightFace ``ArcFaceOnnx.get_feat``: RGB, (x - 127.5) / 127.5. That
    model's first graph nodes contain no Sub/Mul normalisation, so InsightFace selects these
    values for it (checked 2026-11-24).
    """

    def __init__(
        self,
        path: Path | str,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        self.path = Path(path)
        options = ort.SessionOptions()
        options.log_severity_level = 3  # the model declares batch 1; larger batches only warn
        options.intra_op_num_threads = threads  # 0: ONNX Runtime's default
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=list(providers)
        )
        self.input = self.session.get_inputs()[0].name

    def __call__(self, faces: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating]:
        blob = (faces.astype(np.float32) - 127.5) / 127.5
        output = self.session.run(
            None, {self.input: np.ascontiguousarray(blob.transpose(0, 3, 1, 2))}
        )
        embeddings: npt.NDArray[np.floating] = np.asarray(output[0])
        return embeddings


def recognition_matrix(template: Points, size: int) -> Matrix:
    """Similarity from a ``size`` crop aligned to ``template`` to the 112 px ArcFace crop.

    In the aligned crop, landmark k sits at ``template[k] * size - 0.5`` (pixel centres,
    ``facehall.align``). Every crop of a dataset shares the template, so one matrix serves all.
    """
    points = np.asarray(template, dtype=np.float64) * size - 0.5
    return umeyama(points, ARCFACE_TEMPLATE_112)


class IdentitySimilarity:
    """Cosine similarity of the embeddings of SR and HR, per image (higher is better).

    Both crops are warped to the ArcFace template with the fixed ``recognition_matrix``
    (bilinear, black border, as InsightFace ``norm_crop``); no face detection is involved.
    """

    def __init__(self, embedder: Embedder, template: Points, size: int) -> None:
        self.embedder = embedder
        self.matrix = recognition_matrix(template, size)

    def embed(self, images: torch.Tensor) -> npt.NDArray[np.float64]:
        """Unit-length embeddings of (B, 3, S, S) crops in [0, 1]."""
        pixels = (images.clamp(0, 1) * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        faces = np.stack(
            [
                warp(image, self.matrix, ARCFACE_SIZE, interpolation="bilinear", border="constant")
                for image in pixels
            ]
        ).astype(np.uint8)
        embeddings = np.asarray(self.embedder(faces), dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        unit: npt.NDArray[np.float64] = embeddings / np.maximum(norms, 1e-12)
        return unit

    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(cosine(self.embed(sr), self.embed(hr)))


def cosine(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Row-wise cosine similarity of unit-length embeddings."""
    result: npt.NDArray[np.float64] = (a * b).sum(axis=1)
    return result
