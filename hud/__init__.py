"""Human Union Data SDK.

The HUD SDK provides tools for building, evaluating, and deploying AI agents.
"""

from __future__ import annotations

# Import telemetry functions directly for clean access
from .telemetry import trace, job, get_trace, clear_trace, create_job

__all__ = [
    "trace",
    "job",
    "get_trace", 
    "clear_trace",
    "create_job",
]

# Version will be added by setuptools_scm
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"