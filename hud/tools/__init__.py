"""HUD tools for computer control, file editing, and bash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import ToolError, ToolResult, tool_result_to_content_blocks
from .bash import BashTool
from .edit import EditTool
from .playwright_tool import PlaywrightTool

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
    "tool_result_to_content_blocks",
]


def __getattr__(name: str) -> Any:
    """Lazy import computer tools to avoid importing pyautogui unless needed."""
    if name in ("AnthropicComputerTool", "HudComputerTool", "OpenAIComputerTool"):
        from . import computer

        return getattr(computer, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
