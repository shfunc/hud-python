"""Human Union Data SDK.

The HUD SDK provides tools for building, evaluating, and deploying AI agents.
"""

from __future__ import annotations

# Import trace function directly for clean access
from .trace import trace, get_trace

__all__ = [
    "trace",
    "get_trace",
]

# Version will be added by setuptools_scm
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"