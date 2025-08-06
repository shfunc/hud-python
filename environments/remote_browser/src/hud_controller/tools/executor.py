"""Browser-based executor for computer tools that uses Playwright."""

import base64
import logging
from typing import Literal, Optional

from hud.tools.executors.base import BaseExecutor
from hud.tools.base import ToolResult

logger = logging.getLogger(__name__)


class BrowserExecutor(BaseExecutor):
    """
    Executor that performs all actions within a browser viewport using Playwright.

    This allows HudComputerTool (and its subclasses like AnthropicComputerTool
    and OpenAIComputerTool) to work with remote browser environments.

    The executor translates computer control actions into browser page actions,
    making it possible to control web applications as if they were desktop apps.
    """

    def __init__(self, playwright_tool, display_num: int | None = None):
        """
        Initialize the browser executor.

        Args:
            playwright_tool: PlaywrightToolWithMemory instance for browser control
            display_num: Not used for browser executor, kept for compatibility
        """
        super().__init__(display_num)
        self.playwright_tool = playwright_tool
        logger.info("BrowserExecutor initialized with Playwright backend")

    async def _ensure_page(self):
        """Ensure browser and page are available."""
        await self.playwright_tool._ensure_browser()
        if not self.playwright_tool.page:
            raise RuntimeError("No browser page available")
        return self.playwright_tool.page

    async def screenshot(self) -> str | None:
        """Take a screenshot and return base64 encoded image."""
        try:
            page = await self._ensure_page()
            screenshot_bytes = await page.screenshot(full_page=False)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            logger.info("Browser screenshot captured")
            return screenshot_b64
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Click at coordinates in the browser viewport."""
        try:
            page = await self._ensure_page()

            if x is None or y is None:
                return ToolResult(error="Coordinates required for click")

            # Handle modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.down(key)

            # Map button names
            button_map = {
                "left": "left",
                "right": "right",
                "middle": "middle",
                "back": "left",  # Browser doesn't have back button
                "forward": "left",  # Browser doesn't have forward button
            }

            # Perform click(s)
            if pattern:
                # Multi-click pattern
                for delay in pattern:
                    await page.mouse.click(x, y, button=button_map[button])
                    if delay > 0:
                        await page.wait_for_timeout(delay)
            else:
                # Single click
                await page.mouse.click(x, y, button=button_map[button])

            # Release modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.up(key)

            logger.info(f"Clicked at ({x}, {y}) with button {button}")

            result = ToolResult(output=f"Clicked at ({x}, {y})")
            if take_screenshot:
                result = result + ToolResult(base64_image=await self.screenshot())

            return result

        except Exception as e:
            logger.error(f"Click failed: {e}")
            return ToolResult(error=str(e))

    async def type_text(
        self,
        text: str,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Type text in the browser."""
        try:
            page = await self._ensure_page()

            # Handle modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.down(key)

            # Type the text
            await page.keyboard.type(text)

            # Release modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.up(key)

            logger.info(f"Typed text: {text[:50]}...")

            result = ToolResult(output=f"Typed: {text}")
            if take_screenshot:
                result = result + ToolResult(base64_image=await self.screenshot())

            return result

        except Exception as e:
            logger.error(f"Type failed: {e}")
            return ToolResult(error=str(e))

    async def key(
        self,
        keys: list[str],
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Press keyboard keys in the browser."""
        try:
            page = await self._ensure_page()

            # Handle modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.down(key)

            # Press the keys
            for key in keys:
                await page.keyboard.press(key)

            # Release modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.up(key)

            logger.info(f"Pressed keys: {keys}")

            result = ToolResult(output=f"Pressed: {', '.join(keys)}")
            if take_screenshot:
                result = result + ToolResult(base64_image=await self.screenshot())

            return result

        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return ToolResult(error=str(e))

    async def scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        scroll_x: int | None = None,
        scroll_y: int | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Scroll in the browser viewport."""
        try:
            page = await self._ensure_page()

            # Default to center of viewport if coordinates not provided
            if x is None or y is None:
                viewport = page.viewport_size
                x = viewport["width"] // 2 if viewport else 400
                y = viewport["height"] // 2 if viewport else 300

            # Move to position
            await page.mouse.move(x, y)

            # Perform scroll
            delta_x = scroll_x or 0
            delta_y = scroll_y or 0
            await page.mouse.wheel(delta_x, delta_y)

            logger.info(f"Scrolled at ({x}, {y}) by ({delta_x}, {delta_y})")

            result = ToolResult(output=f"Scrolled by ({delta_x}, {delta_y})")
            if take_screenshot:
                result = result + ToolResult(base64_image=await self.screenshot())

            return result

        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return ToolResult(error=str(e))

    async def move(
        self,
        x: int | None = None,
        y: int | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Move mouse to coordinates in the browser."""
        try:
            page = await self._ensure_page()

            if x is None or y is None:
                return ToolResult(error="Coordinates required for move")

            await page.mouse.move(x, y)

            logger.info(f"Moved mouse to ({x}, {y})")

            result = ToolResult(output=f"Moved to ({x}, {y})")
            if take_screenshot:
                result = result + ToolResult(base64_image=await self.screenshot())

            return result

        except Exception as e:
            logger.error(f"Move failed: {e}")
            return ToolResult(error=str(e))

    async def drag(
        self,
        path: list[tuple[int, int]],
        button: Literal["left", "right", "middle"] = "left",
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Drag along a path in the browser."""
        try:
            page = await self._ensure_page()

            if not path or len(path) < 2:
                return ToolResult(error="Path must have at least 2 points")

            # Handle modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.down(key)

            # Start drag
            start_x, start_y = path[0]
            await page.mouse.move(start_x, start_y)
            await page.mouse.down(button=button)

            # Move through path
            for x, y in path[1:]:
                await page.mouse.move(x, y)

            # End drag
            await page.mouse.up(button=button)

            # Release modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.up(key)

            logger.info(f"Dragged from {path[0]} through {len(path)} points")

            result = ToolResult(output=f"Dragged through {len(path)} points")
            if take_screenshot:
                result = result + ToolResult(base64_image=await self.screenshot())

            return result

        except Exception as e:
            logger.error(f"Drag failed: {e}")
            return ToolResult(error=str(e))
