"""Config schemas for datasets and degradations (``configs/data``, ``configs/degrade``)."""

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class JitterConfig:
    """Alignment jitter limits (``facehall.align.Jitter``)."""

    shift: float = 0.0  # fraction of the crop side, per axis
    rotation: float = 0.0  # degrees
    scale: float = 0.0  # factor in [1 / (1 + scale), 1 + scale]


@dataclass
class DataConfig:
    """An aligned face dataset built from a prepared manifest (docs/DATA.md)."""

    root: str = MISSING  # dataset directory with manifest.parquet
    hr_size: int = 128  # side of the HR crop
    template: str = "celeba"  # name in facehall.data.dataset.TEMPLATES
    interpolation: str = "bicubic"  # source -> HR warp (facehall.align.INTERPOLATIONS)
    border: str = "reflect"  # filling for crops that reach outside the image (D16)
    exclude_blurry: list[str] = field(default_factory=list)  # splits without Blurry = 1 (D15)
    hflip: float = 0.0  # probability of a horizontal flip, augmented splits only
    jitter: JitterConfig = field(default_factory=JitterConfig)  # augmented splits only


@dataclass
class BlurConfig:
    """Blur before a resize. Kind weights are relative sampling weights.

    Sizes are in LR pixels: at a working resolution r times the LR size, the kernel is r times
    larger, so one config gives the same blur in the LR image at every scale.
    """

    prob: float = 0.0
    gaussian: float = 0.0  # isotropic Gaussian, sigma from ``sigma``
    anisotropic: float = 0.0  # Gaussian with two sigmas from ``sigma`` and a random angle
    motion: float = 0.0  # straight motion path of length ``motion_length``, random angle
    defocus: float = 0.0  # uniform disk of radius ``defocus_radius``
    sigma: list[float] = field(default_factory=lambda: [0.0, 0.0])  # [low, high], LR px
    motion_length: list[float] = field(default_factory=lambda: [0.0, 0.0])  # LR px
    defocus_radius: list[float] = field(default_factory=lambda: [0.0, 0.0])  # LR px


@dataclass
class ResizeConfig:
    """Resize to the stage's output size. Mode weights are relative sampling weights."""

    bicubic: float = 1.0
    bilinear: float = 0.0
    area: float = 0.0
    nearest: float = 0.0
    antialias: float = 1.0  # probability of antialiasing a bicubic or bilinear downscale


@dataclass
class NoiseConfig:
    """Additive noise at the stage's output size. Kind weights are relative sampling weights."""

    prob: float = 0.0
    gaussian: float = 0.0  # standard deviation from ``sigma``, in 8-bit levels
    poisson: float = 0.0  # shot noise, ``peak`` expected photons at white (log-uniform)
    sigma: list[float] = field(default_factory=lambda: [0.0, 0.0])
    peak: list[float] = field(default_factory=lambda: [1.0, 1.0])
    gray: float = 0.0  # probability of the same noise in every channel (computed on luma)


@dataclass
class JpegConfig:
    prob: float = 0.0
    quality: list[int] = field(default_factory=lambda: [95, 95])  # [low, high], inclusive


@dataclass
class StageConfig:
    """One degradation stage: blur -> resize -> noise -> JPEG."""

    blur: BlurConfig = field(default_factory=BlurConfig)
    resize: ResizeConfig = field(default_factory=ResizeConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    jpeg: JpegConfig = field(default_factory=JpegConfig)


@dataclass
class SecondOrderConfig:
    """An optional second stage (Real-ESRGAN style, arXiv:2107.10833).

    When it runs, the first stage ends at ``size`` times the LR size and the second stage resizes
    from there to the LR size.
    """

    prob: float = 0.0
    size: list[float] = field(default_factory=lambda: [1.0, 1.0])  # [low, high]
    stage: StageConfig = field(default_factory=StageConfig)


@dataclass
class ColorShiftConfig:
    """Per-channel offset of the LR image; the HR target keeps its colours."""

    prob: float = 0.0
    shift: float = 0.0  # largest offset, in 8-bit levels


@dataclass
class DegradeConfig:
    """HR -> LR degradation (``facehall.data.degrade``).

    ``kind: bicubic`` is degradation v0 (PR 1.4) and uses only ``antialias`` and ``quantize``.
    ``kind: pipeline`` (PR 2.1) draws a random degradation per sample from the other fields.
    """

    kind: str = "bicubic"
    antialias: bool = True  # bicubic only
    quantize: bool = True  # round LR to 8-bit levels, like a stored image
    first: StageConfig = field(default_factory=StageConfig)
    second: SecondOrderConfig = field(default_factory=SecondOrderConfig)
    grayscale: float = 0.0  # probability; the HR target is turned grey as well
    color_shift: ColorShiftConfig = field(default_factory=ColorShiftConfig)
