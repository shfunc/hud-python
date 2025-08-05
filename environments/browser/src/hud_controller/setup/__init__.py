"""Setup tools package for browser environment.

This package provides environment-specific setup functions that can be used
as MCP resources and for direct setup calls.
"""

from .registry import SetupRegistry, setup
from .todo import *

__all__ = ["SetupRegistry", "setup"]
