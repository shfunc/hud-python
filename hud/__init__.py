"""
HUD SDK for interacting with the HUD evaluation platform.
"""

from __future__ import annotations

from . import agent, env, gym, settings, task, taskset, types, utils
from .adapters import ResponseAction as Response
from .job import create_job, load_job, run_job
from .job import job as register_job
from .task import Task
from .taskset import load_taskset
from .telemetry import flush, trace, trace_open
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
    "env",
    "flush",
    "gym",
    "init_telemetry",
    "load_job",
    "load_taskset",
    "register_job",
    "run_job",
    "settings",
    "task",
    "taskset",
    "trace",
    "trace_open",
    "types",
    "utils",
]
