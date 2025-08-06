"""Evaluators module for remote browser environment."""

from hud.tools import EvaluateTool

# Create the evaluate tool instance
evaluate_tool = EvaluateTool(
    name="evaluate",
    description="Evaluate the browser state"
)

# Create decorator for registering to this tool
evaluator = evaluate_tool.register

# Import all evaluator modules to trigger registration
from . import (
    url_match,
    page_contains,
    element_exists,
    cookie_exists,
    cookie_match,
    history_length,
    selector_history,
    raw_last_action_is,
    verify_type_action,
    sheet_contains,
    sheets_cell_values,
)

__all__ = ['evaluate_tool', 'evaluator']