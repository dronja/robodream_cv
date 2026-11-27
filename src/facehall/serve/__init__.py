"""``facehall serve``: the REST API for photos (Phase 5, PR 5.3, D36); see ``facehall.serve.app``.

The service runs the ONNX exports of the reference models (``models.onnx_x4``, ``onnx_x8``)
through ONNX Runtime, without PyTorch (D34); ``--set models.backend=torch`` serves the
checkpoints instead.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from facehall.pipeline import Enhancer, PipelineConfig
from facehall.serve.app import Busy, Worker, check_image, create_app, read_body, warm_up

__all__ = [
    "Busy",
    "Worker",
    "check_image",
    "create_app",
    "load_enhancers",
    "read_body",
    "run_serve",
    "warm_up",
]

LOOPBACK = ("127.0.0.1", "localhost", "::1")
logger = logging.getLogger("facehall.serve")


def load_enhancers(
    config: PipelineConfig, scales: Sequence[int], pipeline: Path, device: str
) -> dict[int, Enhancer]:
    """One enhancer per scale, from the configured models (``models.backend``)."""
    from facehall.enhance import model_path

    enhancers: dict[int, Enhancer] = {}
    for scale in dict.fromkeys(scales):
        path = model_path(config, scale, pipeline)
        enhancer = Enhancer.load(path, config, device)
        if enhancer.scale != scale:
            raise ValueError(f"{path} is a x{enhancer.scale} model, not x{scale}")
        logger.info(
            "x%d: %s on %s (%s)", scale, enhancer.info.get("name"), enhancer.device, path.name
        )
        enhancers[scale] = enhancer
    return enhancers


def run_serve(
    *,
    pipeline: Path,
    overrides: Sequence[str] = (),
    scales: Sequence[int] = (),
    host: str | None = None,
    port: int | None = None,
    device: str = "auto",
) -> int:
    """The ``facehall serve`` command: load the models and serve until interrupted."""
    import uvicorn

    from facehall.config import load_config

    config = load_config(pipeline, PipelineConfig, ["models.backend=onnx", *overrides])
    serve = config.serve
    enhancers = load_enhancers(config, scales or serve.scales, pipeline, device)
    host = host or serve.host
    if host not in LOOPBACK:
        logger.warning(
            "serving on %s: the API has no authentication, so anyone who can reach this "
            "address can use it (meant for local use only, D13; in Docker, publish the port on "
            "127.0.0.1 only: -p 127.0.0.1:8000:8000)",
            host,
        )
    uvicorn.run(
        create_app(enhancers, serve),
        host=host,
        port=port or serve.port,
        access_log=False,  # its lines carry the query string, which may hold face boxes
        server_header=False,
    )
    return 0
