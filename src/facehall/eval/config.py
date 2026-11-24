"""Config schema of an evaluation protocol (``configs/eval/*.yaml``)."""

from dataclasses import dataclass, field

from omegaconf import MISSING

from facehall.config import require_choice
from facehall.data.config import DataConfig, DegradeConfig
from facehall.data.manifest import SPLITS

LPIPS_NETS = ("alex", "vgg", "squeeze")


@dataclass
class MetricsConfig:
    lpips: str = "alex"  # LPIPS backbone (LPIPS_NETS); "" disables LPIPS
    identity: str = ""  # path of the recognition model (ONNX); "" disables ID similarity
    identity_threads: int = 0  # CPU threads for the recognition model; 0: ONNX Runtime default


@dataclass
class EvalConfig:
    """A frozen evaluation protocol: every model is scored on exactly these samples.

    The protocol, not the checkpoint, defines the data and the degradation, so models trained
    differently are compared on the same inputs.
    """

    name: str = MISSING  # protocol name; results go to <output>/<name>/<split>/<model label>/
    level: str = ""  # degradation level for report tables (e.g. "light"); "": the protocol name
    scale: int = MISSING
    split: str = "test"
    seed: int = 0  # degradation randomness and the grid samples
    device: str = "auto"
    batch_size: int = 64
    workers: int = 8
    output: str = "outputs/eval"
    grid_samples: int = 8  # images saved for comparison grids, local only (D19); 0 disables
    subgroups: list[str] = field(default_factory=list)  # boolean manifest columns
    # Frozen inputs: split -> sha256 of every LR and HR crop (facehall.eval.runner.InputDigest).
    # A split listed here must reproduce its digest exactly, or the evaluation fails.
    inputs_sha256: dict[str, str] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    degrade: DegradeConfig = field(default_factory=DegradeConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def validate_eval(config: EvalConfig) -> None:
    require_choice(config.split, SPLITS, "split")
    for split, digest in config.inputs_sha256.items():
        require_choice(split, SPLITS, "split in inputs_sha256")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"inputs_sha256.{split} must be a lowercase sha256, got {digest!r}")
    if config.metrics.lpips:
        require_choice(config.metrics.lpips, LPIPS_NETS, "LPIPS network")
    for name, value, lowest in (
        ("batch_size", config.batch_size, 1),
        ("workers", config.workers, 0),
        ("grid_samples", config.grid_samples, 0),
    ):
        if value < lowest:
            raise ValueError(f"{name} must be at least {lowest}, got {value}")
