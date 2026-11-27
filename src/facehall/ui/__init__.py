"""``facehall ui``: the local web interface (D41); see ``facehall.ui.app``.

It runs the models of ``facehall enhance`` (``models.backend``: the checkpoints by default, or
their ONNX exports), loaded when a batch first needs them.
"""

import logging
import secrets
import shutil
import socket
import threading
import time
import warnings
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from facehall.pipeline import Enhancer, PipelineConfig
from facehall.ui.app import Settings, create_app
from facehall.ui.batches import Batches, Options

__all__ = ["Batches", "Options", "Settings", "create_app", "run_ui"]

logger = logging.getLogger("facehall.ui")
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def open_when_listening(host: str, port: int, url: str, timeout: float = 10.0) -> None:
    """Open ``url`` in the browser once the server accepts connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    webbrowser.open(url)


def run_ui(
    *,
    pipeline: Path,
    overrides: Sequence[str] = (),
    host: str | None = None,
    port: int | None = None,
    device: str = "auto",
    browser: bool = True,
) -> int:
    """The ``facehall ui`` command: serve the page until interrupted."""
    import uvicorn

    from facehall.config import load_config
    from facehall.enhance import model_path
    from facehall.ui import media

    config = load_config(pipeline, PipelineConfig, overrides)
    ui = config.ui
    host = host or ui.host
    port = port or ui.port
    # Outputs reach max_output_pixels (200 MP), beyond PIL's guard against decompression bombs.
    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, config.max_output_pixels)
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)

    def available(scale: int) -> bool:
        return Path(config.models.path(scale)).is_file()

    def load(scale: int) -> Enhancer:
        enhancer = Enhancer.load(model_path(config, scale, pipeline), config, device)
        logger.info("x%d: %s on %s", scale, enhancer.info.get("name"), enhancer.device)
        return enhancer

    home = Path.home()
    settings = Settings(
        home=home,
        output=Path(ui.output).expanduser() if ui.output else media.default_output(home),
        scales={s: available(s) for s in (4, 8)},
        backend=config.models.backend,
        device=device,
        ffmpeg=shutil.which("ffmpeg") is not None,
    )
    if not any(settings.scales.values()):
        logger.warning(
            "no model found (%s, %s): see docs/USAGE.md §3",
            config.models.path(4),
            config.models.path(8),
        )
    if host not in LOOPBACK:
        logger.warning(
            "serving on %s: the token travels unencrypted over the network; the interface "
            "is meant for local use only (D13)",
            host,
        )
    token = secrets.token_urlsafe(24)
    shown = f"[{host}]" if ":" in host else host
    url = f"http://{shown}:{port}/#token={token}"

    def started() -> None:
        print(f"facehall ui: open {url}", flush=True)
        if browser:
            threading.Thread(
                target=open_when_listening, args=(host, port, url), daemon=True
            ).start()

    app = create_app(Batches(load, available), settings, ui, token=token, on_start=started)
    uvicorn.run(app, host=host, port=port, access_log=False, server_header=False)
    return 0
