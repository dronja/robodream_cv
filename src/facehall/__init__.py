"""Face hallucination: identity-preserving face super-resolution."""

import os

__version__ = "0.1.0"

# ONNX Runtime's official builds send telemetry to Microsoft by default, Linux included (the
# package's Privacy.md). The variable stops the uploader, the events and the persistent device
# id, but only if it is set before onnxruntime is imported, so it is set here, before any facehall
# module imports it. An explicit value in the environment is kept.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
