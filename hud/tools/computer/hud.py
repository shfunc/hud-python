# flake8: noqa: B008
from __future__ import annotations

import logging
import platform
from typing import Any, Literal

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ImageContent, TextContent
from pydantic import Field

from hud.tools.base import ToolError, ToolResult, tool_result_to_content_blocks
from hud.tools.executors.base import BaseExecutor
from hud.tools.executors.pyautogui import PyAutoGUIExecutor
from hud.tools.executors.xdo import XDOExecutor

logger = logging.getLogger(__name__)


class HudComputerTool:
    """
    A tool that allows the agent to control the computer.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        display_num: int | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
    ) -> None:
        """
        Initialize the HUD computer tool.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            display_num: X display number
            platform_type: Which executor to use:
                - "auto": Automatically detect based on platform
                - "xdo": Use XDOExecutor (Linux/X11 only)
                - "pyautogui": Use PyAutoGUIExecutor (cross-platform)
        """
        self.width = width
        self.height = height

        # Choose executor based on platform_type
        if platform_type == "auto":
            # Auto-detect based on platform
            system = platform.system().lower()
            if system == "linux":
                # Try XDO first on Linux
                if XDOExecutor.is_available():
                    self.executor = XDOExecutor(display_num=display_num)
                    logger.info("Using XDOExecutor")
                elif PyAutoGUIExecutor.is_available():
                    self.executor = PyAutoGUIExecutor(display_num=display_num)
                    logger.info("Using PyAutoGUIExecutor")
                else:
                    self.executor = BaseExecutor(display_num=display_num)
                    logger.info("No display available, using BaseExecutor (simulation mode)")
            else:
                # Windows/macOS - try PyAutoGUI
                if PyAutoGUIExecutor.is_available():
                    self.executor = PyAutoGUIExecutor(display_num=display_num)
                    logger.info("Using PyAutoGUIExecutor")
                else:
                    self.executor = BaseExecutor(display_num=display_num)
                    logger.info("PyAutoGUI not available, using BaseExecutor (simulation mode)")

        elif platform_type == "xdo":
            if XDOExecutor.is_available():
                self.executor = XDOExecutor(display_num=display_num)
                logger.info("Using XDOExecutor")
            else:
                self.executor = BaseExecutor(display_num=display_num)
                logger.warning("XDO not available, using BaseExecutor (simulation mode)")

        elif platform_type == "pyautogui":
            if PyAutoGUIExecutor.is_available():
                self.executor = PyAutoGUIExecutor(display_num=display_num)
                logger.info("Using PyAutoGUIExecutor")
            else:
                self.executor = BaseExecutor(display_num=display_num)
                logger.warning("PyAutoGUI not available, using BaseExecutor (simulation mode)")
        else:
            raise ValueError(f"Invalid platform_type: {platform_type}")

    async def __call__(
        self,
        action: str = Field(..., description="The action name (click, type, move, etc.)"),
        # Click parameters
        x: int | None = Field(None, description="X coordinate for click/move/scroll actions"),
        y: int | None = Field(None, description="Y coordinate for click/move/scroll actions"),
        button: Literal["left", "right", "middle", "back", "forward"] | None = Field(
            None, description="Mouse button for click actions"
        ),
        pattern: list[int] | None = Field(
            None, description="Click pattern for multi-clicks (e.g., [100] for double-click)"
        ),
        # Key/Type parameters
        text: str | None = Field(None, description="Text for type/response actions"),
        keys: list[str] | None = Field(None, description="Keys for press/keydown/keyup actions"),
        enter_after: bool | None = Field(None, description="Whether to press Enter after typing"),
        # Scroll parameters
        scroll_x: int | None = Field(
            None, description="Horizontal scroll amount (positive = right)"
        ),
        scroll_y: int | None = Field(None, description="Vertical scroll amount (positive = down)"),
        # Move parameters
        offset_x: int | None = Field(None, description="X offset for relative move"),
        offset_y: int | None = Field(None, description="Y offset for relative move"),
        # Drag parameters
        path: list[tuple[int, int]] | None = Field(
            None, description="Path for drag actions as list of (x, y) coordinates"
        ),
        # Wait parameter
        time: int | None = Field(None, description="Time in milliseconds for wait action"),
        # General parameters
        hold_keys: list[str] | None = Field(None, description="Keys to hold during action"),
        # hold_key specific
        duration: float | None = Field(None, description="Duration in seconds for hold_key action"),
    ) -> list[ImageContent | TextContent]:
        """
        Execute a computer control action by name.

        Returns:
            List of MCP content blocks
        """
        logger.info("HudComputerTool executing action: %s", action)

        try:
            # Delegate to executor based on action
            if action == "click":
                result = await self.executor.click(
                    x=x, y=y, button=button or "left", pattern=pattern, hold_keys=hold_keys
                )

            elif action == "press":
                if keys is None:
                    raise ToolError("keys parameter is required for press")
                result = await self.executor.press(keys=keys)

            elif action == "keydown":
                if keys is None:
                    raise ToolError("keys parameter is required for keydown")
                result = await self.executor.keydown(keys=keys)

            elif action == "keyup":
                if keys is None:
                    raise ToolError("keys parameter is required for keyup")
                result = await self.executor.keyup(keys=keys)

            elif action == "type":
                if text is None:
                    raise ToolError("text parameter is required for type")
                result = await self.executor.type(text=text, enter_after=enter_after or False)

            elif action == "scroll":
                result = await self.executor.scroll(
                    x=x, y=y, scroll_x=scroll_x, scroll_y=scroll_y, hold_keys=hold_keys
                )

            elif action == "move":
                result = await self.executor.move(x=x, y=y, offset_x=offset_x, offset_y=offset_y)

            elif action == "wait":
                if time is None:
                    raise ToolError("time parameter is required for wait")
                result = await self.executor.wait(time=time)

            elif action == "drag":
                if path is None:
                    raise ToolError("path parameter is required for drag")
                result = await self.executor.drag(path=path, pattern=pattern, hold_keys=hold_keys)

            elif action == "response":
                if text is None:
                    raise ToolError("text parameter is required for response")
                return [TextContent(text=text, type="text")]

            elif action == "screenshot":
                screenshot = await self.executor.screenshot()
                if screenshot:
                    result = ToolResult(base64_image=screenshot)
                else:
                    result = ToolResult(error="Failed to take screenshot")

            elif action == "position":
                result = await self.executor.position()

            elif action == "hold_key":
                if text is None:
                    raise ToolError("text parameter is required for hold_key")
                if duration is None:
                    raise ToolError("duration parameter is required for hold_key")
                result = await self.executor.hold_key(key=text, duration=duration)

            elif action == "mouse_down":
                result = await self.executor.mouse_down(button=button or "left")

            elif action == "mouse_up":
                result = await self.executor.mouse_up(button=button or "left")

            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown action: {action}"))

            # Convert result to content blocks
            return tool_result_to_content_blocks(result)

        except TypeError as e:
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message=f"Invalid parameters for {action}: {e!s}")
            ) from e

    async def click(self, **kwargs: Any) -> ToolResult:
        """Click at specified coordinates."""
        return await self.executor.click(**kwargs)

    async def press(self, keys: list[str]) -> ToolResult:
        """Press a key combination."""
        return await self.executor.press(keys)

    async def keydown(self, keys: list[str]) -> ToolResult:
        """Press and hold keys."""
        return await self.executor.keydown(keys)

    async def keyup(self, keys: list[str]) -> ToolResult:
        """Release keys."""
        return await self.executor.keyup(keys)

    async def type(self, text: str, enter_after: bool = False) -> ToolResult:
        """Type text using the keyboard."""
        return await self.executor.type(text, enter_after=enter_after)

    async def scroll(self, **kwargs: Any) -> ToolResult:
        """Scroll at specified position."""
        return await self.executor.scroll(**kwargs)

    async def move(self, **kwargs: Any) -> ToolResult:
        """Move mouse cursor."""
        return await self.executor.move(**kwargs)

    async def wait(self, time: int) -> ToolResult:
        """Wait for specified time."""
        return await self.executor.wait(time)

    async def drag(self, **kwargs: Any) -> ToolResult:
        """Drag along a path."""
        return await self.executor.drag(**kwargs)

    async def response(self, text: str) -> list[ImageContent | TextContent]:
        """Return a text response."""
        return [TextContent(text=text, type="text")]

    async def screenshot(self) -> ToolResult:
        """Take a screenshot of the current screen."""
        screenshot_base64 = await self.executor.screenshot()
        if screenshot_base64:
            return ToolResult(base64_image=screenshot_base64)
        else:
            return ToolResult(error="Failed to take screenshot")

    async def position(self) -> ToolResult:
        """Get current cursor position."""
        return await self.executor.position()

    async def mouse_down(
        self, button: Literal["left", "right", "middle", "back", "forward"] = "left"
    ) -> ToolResult:
        """Press and hold a mouse button."""
        return await self.executor.mouse_down(button)

    async def mouse_up(
        self, button: Literal["left", "right", "middle", "back", "forward"] = "left"
    ) -> ToolResult:
        """Release a mouse button."""
        return await self.executor.mouse_up(button)

    async def hold_key(self, text: str, duration: float) -> ToolResult:
        """Hold a key for a specified duration."""
        return await self.executor.hold_key(text, duration)
