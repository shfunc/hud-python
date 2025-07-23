from __future__ import annotations

import asyncio
import base64
import logging
from io import BytesIO
import os

if "DISPLAY" not in os.environ:
    try:
        from hud.settings import settings
        os.environ["DISPLAY"] = settings.display
    except (ImportError, AttributeError):
        os.environ["DISPLAY"] = ":0"

try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from hud.tools.base import ToolResult

from .base import BaseExecutor

logger = logging.getLogger(__name__)

class PyAutoGUIExecutor(BaseExecutor):
    """
    Cross-platform executor using PyAutoGUI.
    Works on Windows, macOS, and Linux.
    Falls back to simulation mode when PyAutoGUI is not available.
    """

    def __init__(self, display_num: int | None = None) -> None:
        """
        Initialize the executor.

        Args:
            display_num: X display number (used only on Linux, ignored on Windows/macOS)
        """
        self.display_num = display_num
        self._screenshot_delay = 0.5  # Delay before taking screenshots

        # Check if PyAutoGUI is available and functional
        self.is_simulation = not self._check_pyautogui_available()

        if self.is_simulation:
            logger.warning("PyAutoGUI not available or not functional - running in simulation mode")
            # Initialize parent BaseExecutor
            super().__init__(display_num)
        else:
            logger.info("PyAutoGUI available - running in real mode")
            # Configure PyAutoGUI settings
            pyautogui.FAILSAFE = False  # Disable fail-safe feature
            pyautogui.PAUSE = 0.1  # Small pause between actions

    def _check_pyautogui_available(self) -> bool:
        """
        Check if PyAutoGUI is available and functional.

        Returns:
            True if PyAutoGUI is available and functional, False otherwise
        """
        if not PYAUTOGUI_AVAILABLE:
            return False

        try:
            # Try to get screen size as a simple test
            pyautogui.size()
            return True
        except Exception:
            return False

    async def execute(self, command: str, take_screenshot: bool = True) -> ToolResult:
        """
        Execute a PyAutoGUI command or simulate it if not available.

        Args:
            command: The command description (for logging/simulation)
            take_screenshot: Whether to capture a screenshot after execution

        Returns:
            ToolResult with output, error, and optional screenshot
        """
        # Fall back to simulation if PyAutoGUI not available
        if self.is_simulation:
            return await super().execute(command, take_screenshot)

        # Log the command
        logger.info("Executing PyAutoGUI command: %s", command)

        # Prepare result
        result = ToolResult(output=f"Executed: {command}")

        # Take screenshot if requested
        if take_screenshot:
            await asyncio.sleep(self._screenshot_delay)
            screenshot = await self.screenshot()
            if screenshot:
                result = ToolResult(
                    output=result.output, error=result.error, base64_image=screenshot
                )

        return result

    async def screenshot(self) -> str | None:
        """
        Take a screenshot and return base64 encoded image.

        Returns:
            Base64 encoded PNG image or None if failed
        """
        # Fall back to simulation if PyAutoGUI not available
        if self.is_simulation:
            return await super().screenshot()

        try:
            # Take screenshot using PyAutoGUI
            screenshot = pyautogui.screenshot()

            # Convert to base64
            buffer = BytesIO()
            screenshot.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            return base64.b64encode(image_data).decode()
        except Exception as e:
            logger.error("Failed to take screenshot: %s", e)
            return None

    async def mouse_move(self, x: int, y: int, take_screenshot: bool = True) -> ToolResult:
        """Move mouse to specified coordinates."""
        if self.is_simulation:
            return await self.execute(f"mousemove {x} {y}", take_screenshot=take_screenshot)

        try:
            pyautogui.moveTo(x, y, duration=0.1)
            return await self.execute(f"mousemove {x} {y}", take_screenshot=take_screenshot)
        except Exception as e:
            return ToolResult(error=str(e))

    async def click(
        self,
        button: int = 1,
        x: int | None = None,
        y: int | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Click at specified coordinates or current position."""
        if self.is_simulation:
            if x is not None and y is not None:
                return await self.execute(
                    f"mousemove {x} {y} click {button}", take_screenshot=take_screenshot
                )
            else:
                return await self.execute(f"click {button}", take_screenshot=take_screenshot)

        try:
            # Map button numbers to PyAutoGUI button names
            button_map = {1: "left", 2: "middle", 3: "right"}
            button_name = button_map.get(button, "left")

            if x is not None and y is not None:
                pyautogui.click(x=x, y=y, button=button_name)
                return await self.execute(
                    f"click at ({x}, {y}) with {button_name} button",
                    take_screenshot=take_screenshot,
                )
            else:
                pyautogui.click(button=button_name)
                return await self.execute(
                    f"click with {button_name} button", take_screenshot=take_screenshot
                )
        except Exception as e:
            return ToolResult(error=str(e))

    async def type_text(
        self, text: str, delay: int = 12, take_screenshot: bool = True
    ) -> ToolResult:
        """Type text with specified delay between keystrokes."""
        if self.is_simulation:
            return await self.execute(
                f"type --delay {delay} -- {text}", take_screenshot=take_screenshot
            )

        try:
            # Convert delay from milliseconds to seconds for PyAutoGUI
            interval = delay / 1000.0
            pyautogui.typewrite(text, interval=interval)
            return await self.execute(f"type '{text}'", take_screenshot=take_screenshot)
        except Exception as e:
            return ToolResult(error=str(e))

    async def key(self, key_sequence: str, take_screenshot: bool = True) -> ToolResult:
        """Press a key or key combination."""
        if self.is_simulation:
            return await self.execute(f"key -- {key_sequence}", take_screenshot=take_screenshot)

        try:
            # Handle key combinations (e.g., "ctrl+c")
            if "+" in key_sequence:
                keys = key_sequence.split("+")
                pyautogui.hotkey(*keys)
            else:
                # Map common key names from xdotool to PyAutoGUI
                key_map = {
                    "Return": "enter",
                    "BackSpace": "backspace",
                    "Delete": "delete",
                    "Escape": "escape",
                    "Tab": "tab",
                    "Home": "home",
                    "End": "end",
                    "Page_Up": "pageup",
                    "Page_Down": "pagedown",
                    "Left": "left",
                    "Right": "right",
                    "Up": "up",
                    "Down": "down",
                }
                key = key_map.get(key_sequence, key_sequence.lower())
                pyautogui.press(key)

            return await self.execute(f"key {key_sequence}", take_screenshot=take_screenshot)
        except Exception as e:
            return ToolResult(error=str(e))

    async def scroll(
        self,
        direction: str,
        amount: int = 5,
        x: int | None = None,
        y: int | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """
        Scroll in specified direction.

        Args:
            direction: "up", "down", "left", or "right"
            amount: Number of scroll clicks
            x, y: Optional coordinates to scroll at
            take_screenshot: Whether to capture a screenshot after execution
        """
        if self.is_simulation:
            if x is not None and y is not None:
                return await self.execute(
                    f"mousemove {x} {y} scroll {direction} {amount}",
                    take_screenshot=take_screenshot,
                )
            else:
                return await self.execute(
                    f"scroll {direction} {amount}", take_screenshot=take_screenshot
                )

        try:
            # Move to position if specified
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)

            # Perform scroll
            if direction in ["up", "down"]:
                # Vertical scroll (positive = up, negative = down)
                scroll_amount = amount if direction == "up" else -amount
                pyautogui.scroll(scroll_amount)
            else:
                # Horizontal scroll (if supported)
                scroll_amount = amount if direction == "right" else -amount
                pyautogui.hscroll(scroll_amount)

            return await self.execute(
                f"scroll {direction} {amount}", take_screenshot=take_screenshot
            )
        except Exception as e:
            return ToolResult(error=str(e))

    async def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, take_screenshot: bool = True
    ) -> ToolResult:
        """Drag from start coordinates to end coordinates."""
        if self.is_simulation:
            return await self.execute(
                f"drag from ({start_x}, {start_y}) to ({end_x}, {end_y})",
                take_screenshot=take_screenshot,
            )

        try:
            pyautogui.moveTo(start_x, start_y)
            pyautogui.dragTo(end_x, end_y, duration=0.5, button="left")
            return await self.execute(
                f"drag from ({start_x}, {start_y}) to ({end_x}, {end_y})",
                take_screenshot=take_screenshot,
            )
        except Exception as e:
            return ToolResult(error=str(e))
