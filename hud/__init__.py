"""
HUD SDK for interacting with the HUD evaluation platform.
"""
from __future__ import annotations

# Initialize telemetry
from hud.telemetry import init_telemetry
init_telemetry()

# Import version
from hud.version import __version__

from . import agent, env, gym, settings, task, taskset, types, utils
from .job import create_job, load_job, run_job
from .job import job as register_job
from .taskset import load_taskset
from hud.telemetry import trace, async_trace

__all__ = [
    "__version__",
    "agent",
    "env",
    "gym",
    "settings",
    "task",
    "taskset",
    "types",
    "utils",
    "load_job",
    "create_job",
    "register_job",
    "load_taskset",
    "trace",
    "async_trace",
]
