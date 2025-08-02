"""Problems package for browser environment.

This package provides complete problem definitions that combine setup and evaluation.
"""

from .registry import ProblemRegistry, problem
from .todo import *

__all__ = ["ProblemRegistry", "problem"]
