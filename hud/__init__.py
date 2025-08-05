"""
HUD SDK for interacting with the HUD evaluation platform.
"""

from __future__ import annotations

import warnings
from typing import Any

from . import agent, datasets, env, gym, settings, task, taskset, types, utils
from .adapters import ResponseAction as Response
from .datasets import run_dataset, to_taskconfigs
from .job import create_job, load_job, run_job

# Import deprecated items with deferred warning
from .task import Task as _Task
from .taskset import load_taskset as _load_taskset
from .telemetry import flush, job, trace, trace_open  # New context-based job
from .version import __version__


def __getattr__(name: str) -> Any:
    """Emit deprecation warnings for deprecated imports."""
    if name == "Task":
        warnings.warn(
            "Importing Task from hud is deprecated. "
            "Use hud.datasets.TaskConfig instead. "
            "Task will be removed in v0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _Task
    elif name == "load_taskset":
        warnings.warn(
            "Importing load_taskset from hud is deprecated. "
            "Use hud-evals HuggingFace datasets instead. "
            "load_taskset will be removed in v0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _load_taskset
    raise AttributeError(f"module 'hud' has no attribute '{name}'")


def init_telemetry() -> None:
    from .telemetry import init_telemetry as _init_telemetry

    _init_telemetry()


if settings.settings.fancy_logging:
    import logging
    import sys

    hud_logger = logging.getLogger("hud")
    hud_logger.setLevel(logging.INFO)

    if not hud_logger.handlers:
        # Use the configured stream (defaults to stderr)
        stream = sys.stderr if settings.settings.log_stream.lower() == "stderr" else sys.stdout
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        hud_logger.addHandler(handler)
        hud_logger.propagate = False

__all__ = [
    "Response",
    "__version__",
    "agent",
    "create_job",
    "datasets",
    "env",
    "flush",
    "gym",
    "init_telemetry",
    "job",
    "load_job",
    "run_dataset",
    "run_job",
    "settings",
    "task",
    "taskset",
    "to_taskconfigs",
    "trace",
    "trace_open",
    "types",
    "utils",
]
