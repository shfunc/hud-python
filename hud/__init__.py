"""
HUD SDK for interacting with the HUD evaluation platform.
"""
from __future__ import annotations

from . import agent, env, gym, settings, task, taskset, types, utils
from .job import create_job, load_job, run_job
from .job import job as register_job
from .task import Task
from .taskset import load_taskset
from .telemetry import init_telemetry, trace
from .version import __version__

init_telemetry()

__all__ = [
    "Task",
    "__version__",
    "agent",
    "create_job",
    "env",
    "gym",
    "load_job",
    "load_taskset",
    "register_job",
    "run_job",
    "settings",
    "task",
    "taskset",
    "trace",
    "types",
    "utils",
]
