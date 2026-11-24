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
class DegradeConfig:
    """HR -> LR degradation (``facehall.data.degrade``)."""

    kind: str = "bicubic"
    antialias: bool = True
    quantize: bool = True  # round LR to 8-bit levels, like a stored image
