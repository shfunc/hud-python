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
    from .version import __version__
except ImportError:
    __version__ = "unknown"

# Apply client-side patches for known issues
import logging
try:
    from .agent_patches import apply_all_patches
    apply_all_patches()
except Exception as e:
    logging.getLogger(__name__).debug(f"Failed to apply agent patches: {e}")
