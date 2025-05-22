"""
HUD SDK for interacting with the HUD evaluation platform.
"""
from __future__ import annotations

# Initialize telemetry
from hud.telemetry import init_telemetry

init_telemetry()

# Import version
from hud.telemetry import async_trace, trace
from hud.version import __version__

from . import agent, env, gym, settings, task, taskset, types, utils
from .job import create_job, load_job, run_job
from .job import job as register_job
from .taskset import load_taskset

__all__ = [
    "__version__",
    "agent",
    "async_trace",
    "create_job",
    "env",
    "gym",
    "load_job",
    "load_taskset",
    "register_job",
    "settings",
    "task",
    "taskset",
    "trace",
    "types",
    "utils",
]
