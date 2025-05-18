"""
HUD SDK for interacting with the HUD evaluation platform.
"""

# Initialize telemetry
from hud.telemetry import init_telemetry
init_telemetry()

# Import version
from hud.version import __version__

# Import API components for top-level access
from hud.gym import make
from hud.taskset import TaskSet, load_taskset
from hud.telemetry import trace, async_trace

__all__ = [
    "make",
    "__version__",
    "TaskSet",
    "load_taskset",
    "trace",
    "async_trace",
]
