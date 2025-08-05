# flake8: noqa: B008
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ImageContent, TextContent
from pydantic import Field

from hud.tools.base import ToolResult, tool_result_to_content_blocks

from .hud import HudComputerTool

if TYPE_CHECKING:
    from anthropic.types.beta import BetaToolComputerUse20250124Param

logger = logging.getLogger(__name__)

# Map Anthropic key names to CLA standard keys
ANTHROPIC_TO_CLA_KEYS = {
    # Common variations
    "Return": "enter",
    "Escape": "escape",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "Backspace": "backspace",
    "Delete": "delete",
    "Tab": "tab",
    "Space": "space",
    "Control": "ctrl",
    "Alt": "alt",
    "Shift": "shift",
    "Meta": "win",  # Windows key
    "Command": "cmd",  # macOS
    "Super": "win",  # Linux
    "PageUp": "pageup",
    "PageDown": "pagedown",
    "Home": "home",
    "End": "end",
    "Insert": "insert",
    "F1": "f1",
    "F2": "f2",
    "F3": "f3",
    "F4": "f4",
    "F5": "f5",
    "F6": "f6",
    "F7": "f7",
    "F8": "f8",
    "F9": "f9",
    "F10": "f10",
    "F11": "f11",
    "F12": "f12",
}


class AnthropicComputerTool(HudComputerTool):
    """
    Anthropic Computer Use tool for interacting with the computer.
    """

    name: str = "computer"
    api_type: str = "computer_20250124"

    def __init__(
        self,
        width: int = 1400,
        height: int = 850,
        environment_width: int = 1920,
        environment_height: int = 1080,
        display_num: int | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
        rescale_images: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with Anthropic's default dimensions.

        Args:
            width: Target width for rescaling (default: 1400 for Anthropic)
            height: Target height for rescaling (default: 850 for Anthropic)
            environment_width: Environment screen width (default: 1920)
            environment_height: Environment screen height (default: 1080)
            display_num: X display number
            platform_type: Which executor to use:
                - "auto": Automatically detect based on platform
                - "xdo": Use XDOExecutor (Linux/X11 only)
                - "pyautogui": Use PyAutoGUIExecutor (cross-platform)
            rescale_images: If True, rescale screenshots. If False, only rescale action coordinates
            **kwargs: Additional arguments passed to HudComputerTool (e.g., executor)
        """
        super().__init__(
            width=width,
            height=height,
            display_num=display_num,
            environment_width=environment_width,
            environment_height=environment_height,
            platform_type=platform_type,
            rescale_images=rescale_images,
            **kwargs,
        )

    def to_params(self) -> BetaToolComputerUse20250124Param:
        """Convert to Anthropic tool parameters."""
        return cast(
            "BetaToolComputerUse20250124Param",
            {
                "type": self.api_type,
                "name": self.name,
                "display_width_px": self.width,
                "display_height_px": self.height,
            },
        )

    def _map_anthropic_key_to_cla(self, key: str) -> str:
        """Map Anthropic key name to CLA standard key."""
        # Handle key combinations like "ctrl+a"
        if "+" in key:
            parts = key.split("+")
            mapped_parts = []
            for part in parts:
                # Try exact match first, then case-insensitive
                mapped = ANTHROPIC_TO_CLA_KEYS.get(
                    part, ANTHROPIC_TO_CLA_KEYS.get(part.capitalize(), part.lower())
                )
                mapped_parts.append(mapped)
            return "+".join(mapped_parts)
        else:
            # Single key - try exact match first, then case-insensitive
            return ANTHROPIC_TO_CLA_KEYS.get(
                key, ANTHROPIC_TO_CLA_KEYS.get(key.capitalize(), key.lower())
            )

    async def __call__(
        self,
        action: str = Field(..., description="The action to perform on the computer"),
        coordinate: list[int] | tuple[int, int] | None = Field(
            None, description="The coordinate to interact with on the computer [x, y]"
        ),
        text: str | None = Field(
            None, description="The text to type on the computer or key to press"
        ),
        start_coordinate: list[int] | tuple[int, int] | None = Field(
            None, description="The starting coordinate for drag actions [x, y]"
        ),
        scroll_direction: str | None = Field(
            None, description="The direction to scroll (up, down, left, right)"
        ),
        scroll_amount: int | None = Field(None, description="The amount to scroll"),
        duration: float | None = Field(None, description="The duration of the action in seconds"),
        take_screenshot_on_click: bool = Field(
            True, description="Whether to take a screenshot after clicking"
        ),
    ) -> list[ImageContent | TextContent]:
        """
        Handle Anthropic Computer Use API calls.

        This converts Anthropic's action format to HudComputerTool's format.

        Returns:
            List of MCP content blocks
        """
        logger.info("AnthropicComputerTool received action: %s", action)

        # Convert lists to tuples if needed
        coord_tuple = None
        if coordinate:
            coord_tuple = tuple(coordinate) if isinstance(coordinate, list) else coordinate

        start_coord_tuple = None
        if start_coordinate:
            start_coord_tuple = (
                tuple(start_coordinate) if isinstance(start_coordinate, list) else start_coordinate
            )

        # Map Anthropic actions to HudComputerTool actions
        if action == "screenshot":
            screenshot_base64 = await self.executor.screenshot()
            if screenshot_base64:
                # Rescale screenshot if requested
                screenshot_base64 = await self._rescale_screenshot(screenshot_base64)
                result = ToolResult(base64_image=screenshot_base64)
            else:
                result = ToolResult(error="Failed to take screenshot")

        elif action == "left_click" or action == "click":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                logger.info("Scaled coordinates: %s, %s", scaled_x, scaled_y)
                result = await self.executor.click(x=scaled_x, y=scaled_y)
            else:
                result = await self.executor.click()

        elif action == "double_click":
            if coord_tuple and len(coord_tuple) >= 2:
                # Use pattern for double-click
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, pattern=[100])
            else:
                result = await self.executor.click(pattern=[100])

        elif action == "triple_click":
            if coord_tuple and len(coord_tuple) >= 2:
                # Use pattern for triple-click
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, pattern=[100, 100])
            else:
                result = await self.executor.click(pattern=[100, 100])

        elif action == "right_click":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, button="right")
            else:
                result = await self.executor.click(button="right")

        elif action == "middle_click":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, button="middle")
            else:
                result = await self.executor.click(button="middle")

        elif action == "mouse_move" or action == "move":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.move(x=scaled_x, y=scaled_y)
            else:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="coordinate is required for mouse_move")
                )

        elif action == "type":
            if text:
                result = await self.executor.type(text=text)
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for type"))

        elif action == "key":
            if text:
                # Anthropic sends single key or combo like "ctrl+a"
                # Map to CLA standard key format
                mapped_key = self._map_anthropic_key_to_cla(text)
                result = await self.executor.press(keys=[mapped_key])
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for key"))

        elif action == "scroll":
            # Original implementation validates scroll_direction and scroll_amount
            if scroll_direction not in ["up", "down", "left", "right"]:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="scroll_direction must be 'up', 'down', 'left', or 'right'",
                    )
                )

            if scroll_amount is None or scroll_amount < 0:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="scroll_amount must be a non-negative int"
                    )
                )

            # Convert direction to scroll amounts
            scroll_x = None
            scroll_y = None
            if scroll_direction == "down":
                scroll_y = scroll_amount
            elif scroll_direction == "up":
                scroll_y = -scroll_amount
            elif scroll_direction == "right":
                scroll_x = scroll_amount
            elif scroll_direction == "left":
                scroll_x = -scroll_amount

            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.scroll(
                    x=scaled_x, y=scaled_y, scroll_x=scroll_x, scroll_y=scroll_y
                )
            else:
                result = await self.executor.scroll(scroll_x=scroll_x, scroll_y=scroll_y)

        elif action == "left_click_drag" or action == "drag":
            # Anthropic sends drag with start and end coordinates
            if coord_tuple and len(coord_tuple) >= 2:
                if start_coord_tuple and len(start_coord_tuple) >= 2:
                    # Full drag path
                    path = [
                        (start_coord_tuple[0], start_coord_tuple[1]),
                        (coord_tuple[0], coord_tuple[1]),
                    ]
                    scaled_path = self._scale_path(path)
                    result = await self.executor.drag(path=scaled_path)
                else:
                    # Just end coordinate, drag from current position
                    # Original spec allows this
                    current_pos = [(0, 0), (coord_tuple[0], coord_tuple[1])]  # Simplified
                    scaled_path = self._scale_path(current_pos)
                    result = await self.executor.drag(path=scaled_path)
            else:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="coordinate is required for left_click_drag"
                    )
                )

        elif action == "wait":
            # Original spec expects duration in seconds
            if duration is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="duration is required for wait")
                )
            if duration < 0:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="duration must be non-negative")
                )
            if duration > 100:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration is too long"))

            # Convert seconds to milliseconds for HudComputerTool
            result = await self.executor.wait(time=int(duration * 1000))

        elif action == "hold_key":
            # Original spec has hold_key action
            if text is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="text is required for hold_key")
                )
            if duration is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="duration is required for hold_key")
                )
            if duration < 0:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="duration must be non-negative")
                )
            if duration > 100:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration is too long"))

            # Hold key action
            result = await self.executor.hold_key(key=text, duration=duration)

        elif action == "left_mouse_down":
            # These don't accept coordinates in original spec
            if coord_tuple is not None:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="coordinate is not accepted for left_mouse_down",
                    )
                )
            # Use generic mouse_down method
            result = await self.executor.mouse_down(button="left")

        elif action == "left_mouse_up":
            # These don't accept coordinates in original spec
            if coord_tuple is not None:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="coordinate is not accepted for left_mouse_up"
                    )
                )
            # Use generic mouse_up method
            result = await self.executor.mouse_up(button="left")

        elif action == "cursor_position":
            result = await self.executor.position()

        else:
            # Unknown action
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Invalid action: {action}"))

        # Rescale screenshot in result if present
        if isinstance(result, ToolResult) and result.base64_image and self.rescale_images:
            rescaled_image = await self._rescale_screenshot(result.base64_image)
            result = result.replace(base64_image=rescaled_image)

        # Handle screenshot for actions that need it
        screenshot_actions = {
            "screenshot",
            "left_click",
            "click",
            "double_click",
            "triple_click",
            "right_click",
            "middle_click",
            "mouse_move",
            "move",
            "type",
            "key",
            "scroll",
            "left_click_drag",
            "drag",
            "wait",
            "hold_key",
            "left_mouse_down",
            "left_mouse_up",
        }

        if (
            action in screenshot_actions
            and action != "screenshot"
            and take_screenshot_on_click
            and isinstance(result, ToolResult)
            and not result.base64_image
        ):
            screenshot_base64 = await self.executor.screenshot()
            if screenshot_base64:
                # Rescale screenshot if requested
                screenshot_base64 = await self._rescale_screenshot(screenshot_base64)
                result = ToolResult(
                    output=result.output, error=result.error, base64_image=screenshot_base64
                )

        # Convert to content blocks
        return tool_result_to_content_blocks(result)
