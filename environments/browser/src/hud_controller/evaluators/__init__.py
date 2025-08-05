"""Evaluators package for browser environment.

This package provides environment-specific evaluators that can be used
as MCP resources and for direct evaluation calls.
"""

from .registry import EvaluatorRegistry, evaluator
from .context import BrowserEnvironmentContext, BrowserEvaluationContext

# Import all environment evaluators to trigger registration
from .todo import *

__all__ = [
    "EvaluatorRegistry",
    "evaluator",
    "BrowserEnvironmentContext",
    "BrowserEvaluationContext",  # Backward compatibility
]
