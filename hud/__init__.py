"""
HUD SDK for interacting with the HUD evaluation platform.
"""

from __future__ import annotations

from . import agent, datasets, env, gym, settings, task, taskset, types, utils
from .adapters import ResponseAction as Response
from .datasets import run_dataset, to_taskconfigs
from .job import create_job, load_job, run_job
from .task import Task
from .taskset import load_taskset
from .telemetry import flush, job, trace, trace_open  # New context-based job
from .version import __version__


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
    "Task",
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
    "load_taskset",
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
