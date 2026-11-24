"""Evaluation: per-image metrics, protocol runs, the results report (docs/PROJECT.md §9)."""

from facehall.eval.config import EvalConfig, MetricsConfig, validate_eval
from facehall.eval.runner import evaluate_model

__all__ = ["EvalConfig", "MetricsConfig", "evaluate_model", "validate_eval"]
