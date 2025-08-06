"""HUD tools for computer control, file editing, and bash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseTool, ToolError, ToolResult
from .bash import BashTool
from .edit import EditTool
from .evaluate import BaseEvaluator, EvaluateTool, EvaluationResult
from .playwright_tool import PlaywrightTool
from .setup import BaseSetup, SetupResult, SetupTool

if TYPE_CHECKING:
    from .computer import AnthropicComputerTool, HudComputerTool, OpenAIComputerTool

__all__ = [
    "AnthropicComputerTool",
    "BashTool",
    "EditTool",
    "HudComputerTool",
    "OpenAIComputerTool",
    "PlaywrightTool",
    "ToolError",
    "ToolResult",
    "BaseTool",
    # Setup tools
    "SetupTool",
    "BaseSetup",
    "SetupResult",
    # Evaluate tools
    "EvaluateTool",
    "BaseEvaluator",
    "EvaluationResult",
]


def __getattr__(name: str) -> Any:
    """Lazy import computer tools to avoid importing pyautogui unless needed."""
    if name in ("AnthropicComputerTool", "HudComputerTool", "OpenAIComputerTool"):
        from . import computer

        return getattr(computer, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
