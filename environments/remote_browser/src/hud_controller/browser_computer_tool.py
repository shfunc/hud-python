"""Browser-based computer tool that executes all actions within the browser viewport."""

from __future__ import annotations

import base64
import logging
from typing import Literal, Any
from io import BytesIO

from pydantic import Field
from mcp.types import ImageContent, TextContent

from hud.tools.base import ToolResult, tool_result_to_content_blocks

logger = logging.getLogger(__name__)


class BrowserComputerTool:
    """
    A computer tool that executes all actions within a browser viewport.

    This provides the same interface as HudComputerTool but all actions
    (click, type, screenshot, etc.) happen inside the browser rather than
    on the OS level. Perfect for remote browser environments.
    """

    def __init__(self, playwright_tool, width: int = 1920, height: int = 1080):
        """
        Initialize the browser computer tool.

        Args:
            playwright_tool: PlaywrightToolWithMemory instance
            width: Browser viewport width
            height: Browser viewport height
        """
        self.playwright_tool = playwright_tool
        self.width = width
        self.height = height
        logger.info(f"BrowserComputerTool initialized with viewport {width}x{height}")

    async def __call__(
        self,
        action: str = Field(
            ...,
            description="Action to perform: click, type, press, scroll, screenshot, move, drag, etc.",
        ),
        # Click/move parameters
        x: int | None = Field(None, description="X coordinate"),
        y: int | None = Field(None, description="Y coordinate"),
        button: Literal["left", "right", "middle"] | None = Field(
            None, description="Mouse button for click"
        ),
        pattern: list[int] | None = Field(None, description="Multi-click pattern (delays in ms)"),
        # Type parameters
        text: str | None = Field(None, description="Text to type"),
        enter_after: bool | None = Field(None, description="Press Enter after typing"),
        # Press parameters
        keys: list[str] | None = Field(None, description="Keys to press"),
        # Scroll parameters
        scroll_x: int | None = Field(None, description="Horizontal scroll amount"),
        scroll_y: int | None = Field(None, description="Vertical scroll amount"),
        # Drag parameters
        path: list[tuple[int, int]] | None = Field(
            None, description="Path for drag as list of (x, y) coordinates"
        ),
        # General parameters
        hold_keys: list[str] | None = Field(None, description="Keys to hold during action"),
    ) -> list[ImageContent | TextContent]:
        """
        Execute a computer control action within the browser viewport.

        All actions are performed using Playwright's browser automation APIs,
        making this suitable for remote browser environments where OS-level
        control is not available or desired.
        """
        logger.info(f"BrowserComputerTool executing action: {action}")

        try:
            # Convert numeric string parameters to integers
            if x is not None and isinstance(x, str):
                x = int(x)
            if y is not None and isinstance(y, str):
                y = int(y)
            if scroll_x is not None and isinstance(scroll_x, str):
                scroll_x = int(scroll_x)
            if scroll_y is not None and isinstance(scroll_y, str):
                scroll_y = int(scroll_y)

            await self.playwright_tool._ensure_browser()
            page = self.playwright_tool._page

            if not page:
                result = ToolResult(error="No browser page available")

            elif action == "screenshot":
                result = await self._screenshot()

            elif action == "click":
                result = await self._click(x, y, button or "left", pattern, hold_keys)

            elif action == "type":
                result = await self._type(text or "", enter_after or False)

            elif action == "press":
                # Handle both single key string and list of keys
                if isinstance(keys, str):
                    keys = [keys]
                result = await self._press(keys or [])

            elif action == "scroll":
                result = await self._scroll(x, y, scroll_x, scroll_y)

            elif action == "move":
                result = await self._move(x, y)

            elif action == "drag":
                result = await self._drag(x, y, path)

            elif action == "position":
                result = await self._get_position()

            else:
                result = ToolResult(error=f"Unknown action: {action}")

            return tool_result_to_content_blocks(result)

        except Exception as e:
            logger.error(f"BrowserComputerTool error: {e}")
            return tool_result_to_content_blocks(
                ToolResult(error=f"Browser computer tool error: {str(e)}")
            )

    async def _screenshot(self) -> ToolResult:
        """Take a screenshot of the browser viewport."""
        try:
            page = self.playwright_tool._page
            screenshot_bytes = await page.screenshot(full_page=False)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

            logger.info("Browser screenshot captured")
            return ToolResult(base64_image=screenshot_b64)
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return ToolResult(error=f"Failed to take screenshot: {e}")

    async def _click(
        self,
        x: int | None,
        y: int | None,
        button: str,
        pattern: list[int] | None,
        hold_keys: list[str] | None,
    ) -> ToolResult:
        """Click at coordinates within the browser."""
        try:
            page = self.playwright_tool._page

            # Handle modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.down(key)

            # Default to center if no coordinates
            if x is None or y is None:
                x = self.width // 2
                y = self.height // 2

            # Handle multi-click patterns
            if pattern:
                click_count = len(pattern) + 1
                await page.mouse.click(x, y, button=button, click_count=click_count)
                logger.info(f"Multi-clicked at ({x}, {y}) with pattern {pattern}")
            else:
                await page.mouse.click(x, y, button=button)
                logger.info(f"Clicked at ({x}, {y}) with {button} button")

            # Release modifier keys
            if hold_keys:
                for key in reversed(hold_keys):
                    await page.keyboard.up(key)

            # Take screenshot after action
            screenshot = await self._get_screenshot_base64()
            return ToolResult(output=f"Clicked at ({x}, {y}) in browser", base64_image=screenshot)

        except Exception as e:
            logger.error(f"Click failed: {e}")
            return ToolResult(error=f"Failed to click: {e}")

    async def _type(self, text: str, enter_after: bool) -> ToolResult:
        """Type text in the browser."""
        try:
            page = self.playwright_tool._page

            # Type the text
            await page.keyboard.type(text)
            logger.info(f"Typed text: {text[:50]}...")

            # Press Enter if requested
            if enter_after:
                await page.keyboard.press("Enter")
                logger.info("Pressed Enter after typing")

            # Take screenshot after action
            screenshot = await self._get_screenshot_base64()
            return ToolResult(
                output=f"Typed '{text}'" + (" and pressed Enter" if enter_after else ""),
                base64_image=screenshot,
            )

        except Exception as e:
            logger.error(f"Type failed: {e}")
            return ToolResult(error=f"Failed to type: {e}")

    async def _press(self, keys: list[str]) -> ToolResult:
        """Press key combination."""
        try:
            page = self.playwright_tool._page

            # Handle key combination (e.g., ["ctrl", "a"])
            key_combo = "+".join(keys)
            await page.keyboard.press(key_combo)
            logger.info(f"Pressed key combination: {key_combo}")

            # Take screenshot after action
            screenshot = await self._get_screenshot_base64()
            return ToolResult(output=f"Pressed {key_combo}", base64_image=screenshot)

        except Exception as e:
            logger.error(f"Press failed: {e}")
            return ToolResult(error=f"Failed to press keys: {e}")

    async def _scroll(
        self, x: int | None, y: int | None, scroll_x: int | None, scroll_y: int | None
    ) -> ToolResult:
        """Scroll the page."""
        try:
            page = self.playwright_tool._page

            # Default to center if no coordinates
            if x is None:
                x = self.width // 2
            if y is None:
                y = self.height // 2

            # Move to position first
            await page.mouse.move(x, y)

            # Perform scroll
            delta_x = scroll_x or 0
            delta_y = scroll_y or 0

            await page.mouse.wheel(delta_x, delta_y)
            logger.info(f"Scrolled at ({x}, {y}) by ({delta_x}, {delta_y})")

            # Take screenshot after action
            screenshot = await self._get_screenshot_base64()
            return ToolResult(
                output=f"Scrolled by ({delta_x}, {delta_y}) at ({x}, {y})", base64_image=screenshot
            )

        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return ToolResult(error=f"Failed to scroll: {e}")

    async def _move(self, x: int | None, y: int | None) -> ToolResult:
        """Move mouse to position."""
        try:
            page = self.playwright_tool._page

            # Default to center if no coordinates
            if x is None:
                x = self.width // 2
            if y is None:
                y = self.height // 2

            await page.mouse.move(x, y)
            logger.info(f"Moved mouse to ({x}, {y})")

            # Take screenshot after action
            screenshot = await self._get_screenshot_base64()
            return ToolResult(output=f"Moved mouse to ({x}, {y})", base64_image=screenshot)

        except Exception as e:
            logger.error(f"Move failed: {e}")
            return ToolResult(error=f"Failed to move mouse: {e}")

    async def _drag(
        self, x: int | None, y: int | None, path: list[tuple[int, int]] | None
    ) -> ToolResult:
        """Drag from current position through a path."""
        try:
            page = self.playwright_tool._page

            if not path or len(path) < 1:
                return ToolResult(error="Path required for drag action")

            # Start position
            start_x = x if x is not None else self.width // 2
            start_y = y if y is not None else self.height // 2

            # Move to start
            await page.mouse.move(start_x, start_y)
            await page.mouse.down()

            # Drag through path
            for px, py in path:
                await page.mouse.move(px, py)
                logger.info(f"Dragging to ({px}, {py})")

            await page.mouse.up()

            # Take screenshot after action
            screenshot = await self._get_screenshot_base64()
            return ToolResult(
                output=f"Dragged from ({start_x}, {start_y}) through {len(path)} points",
                base64_image=screenshot,
            )

        except Exception as e:
            logger.error(f"Drag failed: {e}")
            return ToolResult(error=f"Failed to drag: {e}")

    async def _get_position(self) -> ToolResult:
        """Get current mouse position (simulated for browser)."""
        # Browsers don't expose real mouse position, return viewport center
        return ToolResult(
            output=f"Mouse position: ({self.width // 2}, {self.height // 2}) (viewport center)"
        )

    async def _get_screenshot_base64(self) -> str:
        """Helper to get base64 screenshot."""
        page = self.playwright_tool._page
        screenshot_bytes = await page.screenshot(full_page=False)
        return base64.b64encode(screenshot_bytes).decode()
