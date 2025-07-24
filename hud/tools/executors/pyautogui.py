from __future__ import annotations

import asyncio
import base64
import logging
import os
from io import BytesIO
from typing import Literal

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

# Map CLA standard keys to PyAutoGUI keys (only where they differ)
CLA_TO_PYAUTOGUI = {
    # Most keys are the same in PyAutoGUI, only map the differences
    "escape": "esc",
    "enter": "return",
    "pageup": "pgup",
    "pagedown": "pgdn",
    "printscreen": "prtscr",
    "prtsc": "prtscr",
    "super": "win",
    "command": "cmd",
}


class PyAutoGUIExecutor(BaseExecutor):
    """
    Cross-platform executor using PyAutoGUI.
    Works on Windows, macOS, and Linux.

    This executor should only be instantiated when PyAutoGUI is available and functional.
    """

    def __init__(self, display_num: int | None = None) -> None:
        """
        Initialize the executor.

        Args:
            display_num: X display number (used only on Linux, ignored on Windows/macOS)
        """
        super().__init__(display_num)

        logger.info("PyAutoGUIExecutor initialized")

        # Configure PyAutoGUI settings
        pyautogui.FAILSAFE = False  # Disable fail-safe feature
        pyautogui.PAUSE = 0.1  # Small pause between actions

    def _map_key(self, key: str) -> str:
        """Map CLA standard key to PyAutoGUI key."""
        return CLA_TO_PYAUTOGUI.get(key.lower(), key.lower())

    def _map_keys(self, keys: list[str]) -> list[str]:
        """Map CLA standard keys to PyAutoGUI keys."""
        mapped_keys = []
        for key in keys:
            # Handle key combinations like "ctrl+a"
            if "+" in key:
                parts = key.split("+")
                mapped_parts = [self._map_key(part) for part in parts]
                mapped_keys.append("+".join(mapped_parts))
            else:
                mapped_keys.append(self._map_key(key))
        return mapped_keys

    @classmethod
    def is_available(cls) -> bool:
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

    async def screenshot(self) -> str | None:
        """
        Take a screenshot and return base64 encoded image.

        Returns:
            Base64 encoded PNG image or None if failed
        """
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

    # ===== Helper Methods =====

    def _hold_keys_context(self, keys: list[str] | None) -> None:
        """
        Press and hold keys.

        Args:
            keys: List of keys to hold
        """
        if keys:
            for key in keys:
                pyautogui.keyDown(key)

    def _release_keys(self, keys: list[str] | None) -> None:
        """Release held keys."""
        if keys:
            for key in reversed(keys):  # Release in reverse order
                pyautogui.keyUp(key)

    # ===== CLA Action Implementations =====

    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Click at specified coordinates or current position."""
        try:
            # Map button names (PyAutoGUI doesn't support back/forward)
            button_map = {
                "left": "left",
                "right": "right",
                "middle": "middle",
                "back": "left",
                "forward": "right",
            }  # Fallback for unsupported
            button_name = button_map.get(button, "left")

            # Hold keys if specified
            self._hold_keys_context(hold_keys)

            try:
                # Handle multi-clicks based on pattern
                if pattern:
                    clicks = len(pattern) + 1
                    interval = pattern[0] / 1000.0 if pattern else 0.1  # Convert ms to seconds

                    if x is not None and y is not None:
                        pyautogui.click(
                            x=x, y=y, clicks=clicks, interval=interval, button=button_name
                        )
                    else:
                        pyautogui.click(clicks=clicks, interval=interval, button=button_name)
                else:
                    # Single click
                    if x is not None and y is not None:
                        pyautogui.click(x=x, y=y, button=button_name)
                    else:
                        pyautogui.click(button=button_name)
            finally:
                # Release held keys
                self._release_keys(hold_keys)

            result = ToolResult(
                output=f"Clicked {button} button at ({x}, {y})" if x else f"Clicked {button} button"
            )

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def type(
        self, text: str, enter_after: bool = False, delay: int = 12, take_screenshot: bool = True
    ) -> ToolResult:
        """Type text with specified delay between keystrokes."""
        try:
            # Convert delay from milliseconds to seconds for PyAutoGUI
            interval = delay / 1000.0
            pyautogui.typewrite(text, interval=interval)

            if enter_after:
                pyautogui.press("enter")

            result = ToolResult(
                output=f"Typed: '{text}'" + (" and pressed Enter" if enter_after else "")
            )

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def key(self, key_sequence: str, take_screenshot: bool = True) -> ToolResult:
        """Press a key or key combination."""
        try:
            # Handle key combinations (e.g., "ctrl+c")
            if "+" in key_sequence:
                keys = key_sequence.split("+")
                pyautogui.hotkey(*keys)
                result = ToolResult(output=f"Pressed hotkey: {key_sequence}")
            else:
                # Map common key names from xdotool to PyAutoGUI
                key = key_sequence.lower()
                pyautogui.press(CLA_TO_PYAUTOGUI.get(key, key))
                result = ToolResult(output=f"Pressed key: {key_sequence}")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def press(self, keys: list[str], take_screenshot: bool = True) -> ToolResult:
        """Press a key combination (hotkey)."""
        try:
            # Map CLA keys to PyAutoGUI keys
            mapped_keys = self._map_keys(keys)

            # Handle single key or combination
            if len(mapped_keys) == 1 and "+" not in mapped_keys[0]:
                pyautogui.press(mapped_keys[0])
                result = ToolResult(output=f"Pressed key: {keys[0]}")
            else:
                # For combinations, use hotkey
                hotkey_parts = []
                for key in mapped_keys:
                    if "+" in key:
                        hotkey_parts.extend(key.split("+"))
                    else:
                        hotkey_parts.append(key)
                pyautogui.hotkey(*hotkey_parts)
                result = ToolResult(output=f"Pressed hotkey: {'+'.join(keys)}")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def keydown(self, keys: list[str], take_screenshot: bool = True) -> ToolResult:
        """Press and hold keys."""
        try:
            # Map CLA keys to PyAutoGUI keys
            mapped_keys = self._map_keys(keys)
            for key in mapped_keys:
                pyautogui.keyDown(key)

            result = ToolResult(output=f"Keys down: {', '.join(keys)}")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def keyup(self, keys: list[str], take_screenshot: bool = True) -> ToolResult:
        """Release held keys."""
        try:
            # Map CLA keys to PyAutoGUI keys
            mapped_keys = self._map_keys(keys)
            for key in reversed(mapped_keys):  # Release in reverse order
                pyautogui.keyUp(key)

            result = ToolResult(output=f"Keys up: {', '.join(keys)}")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        scroll_x: int | None = None,
        scroll_y: int | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Scroll at specified position."""
        try:
            # Move to position if specified
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)

            # Hold keys if specified
            self._hold_keys_context(hold_keys)

            try:
                msg_parts = []

                # Perform vertical scroll
                if scroll_y and scroll_y != 0:
                    # PyAutoGUI: positive = up, negative = down (opposite of our convention)
                    pyautogui.scroll(-scroll_y)
                    msg_parts.append(f"vertically by {scroll_y}")

                # Perform horizontal scroll (if supported)
                if scroll_x and scroll_x != 0:
                    # PyAutoGUI horizontal scroll might not work on all platforms
                    try:
                        pyautogui.hscroll(scroll_x)
                        msg_parts.append(f"horizontally by {scroll_x}")
                    except AttributeError:
                        # hscroll not available
                        msg_parts.append(f"horizontally by {scroll_x} (not supported)")

                if not msg_parts:
                    return ToolResult(output="No scroll amount specified")

                msg = "Scrolled " + " and ".join(msg_parts)
                if x is not None and y is not None:
                    msg += f" at ({x}, {y})"
                if hold_keys:
                    msg += f" while holding {hold_keys}"
            finally:
                # Release held keys
                self._release_keys(hold_keys)

            result = ToolResult(output=msg)

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def move(
        self,
        x: int | None = None,
        y: int | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Move mouse cursor."""
        try:
            if x is not None and y is not None:
                # Absolute move
                pyautogui.moveTo(x, y, duration=0.1)
                result = ToolResult(output=f"Moved mouse to ({x}, {y})")
            elif offset_x is not None or offset_y is not None:
                # Relative move
                offset_x = offset_x or 0
                offset_y = offset_y or 0
                pyautogui.moveRel(xOffset=offset_x, yOffset=offset_y, duration=0.1)
                result = ToolResult(output=f"Moved mouse by offset ({offset_x}, {offset_y})")
            else:
                return ToolResult(output="No move coordinates specified")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def drag(
        self,
        path: list[tuple[int, int]],
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Drag along a path."""
        if len(path) < 2:
            return ToolResult(error="Drag path must have at least 2 points")

        try:
            # Hold keys if specified
            self._hold_keys_context(hold_keys)

            try:
                # Move to start
                start_x, start_y = path[0]
                pyautogui.moveTo(start_x, start_y)

                # Handle multi-point drag
                if len(path) == 2:
                    # Simple drag
                    end_x, end_y = path[1]
                    pyautogui.dragTo(end_x, end_y, duration=0.5, button="left")
                    result = ToolResult(
                        output=f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
                    )
                else:
                    # Multi-point drag
                    pyautogui.mouseDown(button="left")
                    for i, (x, y) in enumerate(path[1:], 1):
                        duration = 0.1
                        if pattern and i - 1 < len(pattern):
                            duration = pattern[i - 1] / 1000.0  # Convert ms to seconds
                        pyautogui.moveTo(x, y, duration=duration)
                    pyautogui.mouseUp(button="left")

                    result = ToolResult(output=f"Dragged along {len(path)} points")

                if hold_keys:
                    result = ToolResult(output=f"{result.output} while holding {hold_keys}")
            finally:
                # Release held keys
                self._release_keys(hold_keys)

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def mouse_down(
        self,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Press and hold a mouse button."""
        try:
            # Map button names (PyAutoGUI doesn't support back/forward)
            button_map = {
                "left": "left",
                "right": "right",
                "middle": "middle",
                "back": "left",
                "forward": "right",
            }  # Fallback for unsupported
            button_name = button_map.get(button, "left")

            pyautogui.mouseDown(button=button_name)
            result = ToolResult(output=f"Mouse down: {button} button")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def mouse_up(
        self,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        take_screenshot: bool = True,
    ) -> ToolResult:
        """Release a mouse button."""
        try:
            # Map button names (PyAutoGUI doesn't support back/forward)
            button_map = {
                "left": "left",
                "right": "right",
                "middle": "middle",
                "back": "left",
                "forward": "right",
            }  # Fallback for unsupported
            button_name = button_map.get(button, "left")

            pyautogui.mouseUp(button=button_name)
            result = ToolResult(output=f"Mouse up: {button} button")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def hold_key(self, key: str, duration: float, take_screenshot: bool = True) -> ToolResult:
        """Hold a key for a specified duration."""
        try:
            # Map CLA key to PyAutoGUI key
            mapped_key = self._map_key(key)
            pyautogui.keyDown(mapped_key)
            await asyncio.sleep(duration)
            pyautogui.keyUp(mapped_key)

            result = ToolResult(output=f"Held key '{key}' for {duration} seconds")

            if take_screenshot:
                screenshot = await self.screenshot()
                if screenshot:
                    result = ToolResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ToolResult(error=str(e))

    async def position(self) -> ToolResult:
        """Get current cursor position."""
        try:
            x, y = pyautogui.position()
            return ToolResult(output=f"Mouse position: ({x}, {y})")
        except Exception as e:
            return ToolResult(error=str(e))
