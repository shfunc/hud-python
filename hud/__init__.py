"""Human Union Data SDK.

The HUD SDK provides tools for building, evaluating, and deploying AI agents.
"""

from __future__ import annotations

# Import telemetry functions directly for clean access
from .telemetry import clear_trace, create_job, get_trace, job, trace

__all__ = [
    "clear_trace",
    "create_job",
    "get_trace",
    "job",
    "trace",
]

# Version will be added by setuptools_scm
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"