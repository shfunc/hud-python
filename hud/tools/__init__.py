"""HUD tools for computer control, file editing, and bash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseTool, ToolError, ToolResult
from .bash import BashTool
from .edit import EditTool
from .evaluate import BaseEvaluator, EvaluateTool, EvaluationResult
from .playwright import PlaywrightTool
from .setup import BaseSetup, SetupResult, SetupTool

if TYPE_CHECKING:
    from .computer import AnthropicComputerTool, HudComputerTool, OpenAIComputerTool

__all__ = [
    "AnthropicComputerTool",
    "BaseEvaluator",
    "BaseSetup",
    "BaseTool",
    "BashTool",
    "EditTool",
    # Evaluate tools
    "EvaluateTool",
    "EvaluationResult",
    "HudComputerTool",
    "OpenAIComputerTool",
    "PlaywrightTool",
    "SetupResult",
    # Setup tools
    "SetupTool",
    "ToolError",
    "ToolResult",
]


def __getattr__(name: str) -> Any:
    """Lazy import computer tools to avoid importing pyautogui unless needed."""
    if name in ("AnthropicComputerTool", "HudComputerTool", "OpenAIComputerTool"):
        from . import computer

        return getattr(computer, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
