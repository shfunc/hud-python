"""Evaluate module for 2048 environment."""

from hud.tools import EvaluateTool

# Create the evaluate tool instance
evaluate_tool = EvaluateTool(
    name="evaluate", title="Game Evaluator", description="Evaluate the current game state"
)

# Create decorator for registering to this tool
evaluator = evaluate_tool.register

# Import registry to trigger registration
from . import registry

__all__ = ["evaluate_tool"]
