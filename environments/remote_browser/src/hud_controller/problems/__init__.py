"""Problem definitions for remote browser environment."""

from .registry import ProblemRegistry, problem

# Import problem definitions to trigger registration
from .navigate_and_verify import NavigateAndVerifyProblem

__all__ = [
    "ProblemRegistry",
    "problem",
]
