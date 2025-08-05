"""Browser-based executor for HudComputerTool that uses Playwright."""

import base64
import logging
from typing import Literal

from hud.tools.executors.base import BaseExecutor
from hud.tools.base import ToolResult

logger = logging.getLogger(__name__)


class BrowserExecutor(BaseExecutor):
    """
    Executor that performs all actions within a browser viewport using Playwright.

    This allows HudComputerTool (and its subclasses like AnthropicComputerTool
    and OpenAIComputerTool) to work with remote browser environments.
    """

    def __init__(self, playwright_tool, display_num: int | None = None):
        """
        Initialize the browser executor.

        Args:
            playwright_tool: PlaywrightToolWithMemory instance
            display_num: Not used for browser executor, kept for compatibility
        """
        super().__init__(display_num)
        self.playwright_tool = playwright_tool
        logger.info("BrowserExecutor initialized")

    async def _ensure_page(self):
        """Ensure browser and page are available."""
        await self.playwright_tool._ensure_browser()
        if not self.playwright_tool._page:
            raise RuntimeError("No browser page available")
        return self.playwright_tool._page

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
        """Click at specified coordinates in the browser."""
        try:
            page = await self._ensure_page()

            # Handle modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.down(key)

            # Map button names
            button_map = {"back": "x1", "forward": "x2"}
            playwright_button = button_map.get(button, button)

            # Perform click(s)
            if pattern:
                # Multi-click pattern
                for i, delay in enumerate(pattern):
                    if x is not None and y is not None:
                        await page.mouse.click(x, y, button=playwright_button)
                    else:
                        # Click at current position
                        await page.mouse.click(button=playwright_button)

                    if i < len(pattern) - 1:
                        await page.wait_for_timeout(delay)
            else:
                # Single click
                if x is not None and y is not None:
                    await page.mouse.click(x, y, button=playwright_button)
                else:
                    await page.mouse.click(button=playwright_button)

            # Release modifier keys
            if hold_keys:
                for key in reversed(hold_keys):
                    await page.keyboard.up(key)

            msg = f"Clicked at ({x}, {y}) with {button} button in browser"
            if pattern:
                msg += f" (pattern: {pattern})"
            if hold_keys:
                msg += f" while holding {hold_keys}"

            screenshot = await self.screenshot() if take_screenshot else None
            return ToolResult(output=msg, base64_image=screenshot)

        except Exception as e:
            logger.error(f"Click failed: {e}")
            return ToolResult(error=f"Failed to click: {e}")

    async def type(
        self, text: str, enter_after: bool = False, delay: int = 12, take_screenshot: bool = True
    ) -> ToolResult:
        """Type text in the browser."""
        try:
            page = await self._ensure_page()

            # Type with delay between keystrokes
            await page.keyboard.type(text, delay=delay)

            if enter_after:
                await page.keyboard.press("Enter")

            msg = f"Typed '{text}' in browser"
            if enter_after:
                msg += " followed by Enter"

            screenshot = await self.screenshot() if take_screenshot else None
            return ToolResult(output=msg, base64_image=screenshot)

        except Exception as e:
            logger.error(f"Type failed: {e}")
            return ToolResult(error=f"Failed to type: {e}")

    async def press(self, keys: list[str], take_screenshot: bool = True) -> ToolResult:
        """Press a key combination in the browser."""
        try:
            page = await self._ensure_page()

            # Convert single key to Playwright format
            if len(keys) == 1:
                await page.keyboard.press(keys[0])
            else:
                # Handle key combination
                key_combo = "+".join(keys)
                await page.keyboard.press(key_combo)

            msg = f"Pressed keys: {'+'.join(keys)} in browser"
            screenshot = await self.screenshot() if take_screenshot else None
            return ToolResult(output=msg, base64_image=screenshot)

        except Exception as e:
            logger.error(f"Press failed: {e}")
            return ToolResult(error=f"Failed to press keys: {e}")

    async def scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        scroll_amount: int = 5,
        direction: Literal["up", "down", "left", "right"] = "down",
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Scroll in the browser."""
        try:
            page = await self._ensure_page()

            # Calculate scroll deltas
            delta_x = 0
            delta_y = 0

            if direction == "up":
                delta_y = -scroll_amount * 100
            elif direction == "down":
                delta_y = scroll_amount * 100
            elif direction == "left":
                delta_x = -scroll_amount * 100
            elif direction == "right":
                delta_x = scroll_amount * 100

            # Move to position if specified
            if x is not None and y is not None:
                await page.mouse.move(x, y)

            # Perform scroll
            await page.mouse.wheel(delta_x, delta_y)

            msg = f"Scrolled {direction} by {scroll_amount} units in browser"
            if x is not None and y is not None:
                msg += f" at ({x}, {y})"

            screenshot = await self.screenshot() if take_screenshot else None
            return ToolResult(output=msg, base64_image=screenshot)

        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return ToolResult(error=f"Failed to scroll: {e}")

    async def move(self, x: int, y: int, take_screenshot: bool = True) -> ToolResult:
        """Move mouse to coordinates in the browser."""
        try:
            page = await self._ensure_page()
            await page.mouse.move(x, y)

            msg = f"Moved mouse to ({x}, {y}) in browser"
            screenshot = await self.screenshot() if take_screenshot else None
            return ToolResult(output=msg, base64_image=screenshot)

        except Exception as e:
            logger.error(f"Move failed: {e}")
            return ToolResult(error=f"Failed to move: {e}")

    async def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: Literal["left", "right", "middle"] = "left",
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Drag from start to end coordinates in the browser."""
        try:
            page = await self._ensure_page()

            # Move to start position
            await page.mouse.move(start_x, start_y)

            # Press down
            await page.mouse.down(button=button)

            # Move to end position
            await page.mouse.move(end_x, end_y)

            # Release
            await page.mouse.up(button=button)

            msg = f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y}) in browser"
            screenshot = await self.screenshot() if take_screenshot else None
            return ToolResult(output=msg, base64_image=screenshot)

        except Exception as e:
            logger.error(f"Drag failed: {e}")
            return ToolResult(error=f"Failed to drag: {e}")

    # Additional browser-specific actions can override base implementations
    async def scroll_to(
        self, x: int, y: int, smooth: bool = True, take_screenshot: bool = True
    ) -> ToolResult:
        """Scroll page to specific coordinates (browser-specific)."""
        try:
            page = await self._ensure_page()

            # Use JavaScript for precise scrolling
            await page.evaluate(
                f"window.scrollTo({{'left': {x}, 'top': {y}, 'behavior': '{'smooth' if smooth else 'instant'}' }})"
            )

            msg = f"Scrolled page to ({x}, {y}) in browser"
            screenshot = await self.screenshot() if take_screenshot else None
            return ToolResult(output=msg, base64_image=screenshot)

        except Exception as e:
            logger.error(f"Scroll to failed: {e}")
            return ToolResult(error=f"Failed to scroll to position: {e}")
