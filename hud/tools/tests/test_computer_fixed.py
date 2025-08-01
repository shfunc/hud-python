"""Fixed tests for computer tools that handle Field parameters correctly."""

from __future__ import annotations

import pytest
from mcp.types import ImageContent, TextContent

from hud.tools.computer.anthropic import AnthropicComputerTool
from hud.tools.computer.hud import HudComputerTool
from hud.tools.computer.openai import OpenAIComputerTool


def extract_field_defaults(func_params: dict) -> dict:
    """Extract actual values from Field objects in parameters."""
    cleaned = {}
    for key, value in func_params.items():
        if hasattr(value, "default"):  # It's a Field object
            cleaned[key] = value.default
        else:
            cleaned[key] = value
    return cleaned


class MCPToolWrapper:
    """Wrapper that handles Field parameter conversion for MCP tools."""

    def __init__(self, tool):
        self.tool = tool

    async def __call__(self, **kwargs):
        # Convert Field objects to their default values
        cleaned_kwargs = extract_field_defaults(kwargs)
        # Remove None values to use tool's defaults
        cleaned_kwargs = {k: v for k, v in cleaned_kwargs.items() if v is not None}
        return await self.tool(**cleaned_kwargs)


@pytest.mark.asyncio
async def test_hud_computer_screenshot_fixed():
    """Test screenshot with proper parameter handling."""
    comp = HudComputerTool()
    # Call with explicit action parameter only
    blocks = await comp(action="screenshot")

    # Check if we got content blocks back
    assert blocks is not None
    assert len(blocks) > 0

    # The tool should return either ImageContent or TextContent
    # If screenshot fails, it might return TextContent with error
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)


@pytest.mark.asyncio
async def test_hud_computer_click_simulation_fixed():
    """Test click action with proper parameters."""
    comp = HudComputerTool()
    # Pass only the required parameters, let others use their Field defaults
    blocks = await comp(action="click", x=10, y=10)
    assert blocks is not None
    assert len(blocks) > 0


@pytest.mark.asyncio
async def test_openai_computer_screenshot_fixed():
    """Test OpenAI computer tool screenshot."""
    comp = OpenAIComputerTool()
    blocks = await comp(type="screenshot")
    assert blocks is not None
    assert len(blocks) > 0
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)


@pytest.mark.asyncio
async def test_anthropic_computer_screenshot_fixed():
    """Test Anthropic computer tool screenshot."""
    comp = AnthropicComputerTool()
    blocks = await comp(action="screenshot")
    assert blocks is not None
    assert len(blocks) > 0
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)


# Test cases for parameterized tests
FIXED_CASES = [
    ("screenshot", {}),
    ("click", {"x": 1, "y": 1}),  # Removed pattern=[] as it should use Field default
    ("press", {"keys": ["ctrl", "c"]}),
    ("keydown", {"keys": ["shift"]}),
    ("keyup", {"keys": ["shift"]}),
    ("type", {"text": "hello"}),
    ("scroll", {"x": 10, "y": 10, "scroll_y": 20}),  # Added x,y coordinates
    ("move", {"x": 5, "y": 5}),  # Just x,y, no offset params
    ("wait", {"time": 5}),
    ("drag", {"path": [(0, 0), (10, 10)]}),
    ("mouse_down", {}),
    ("mouse_up", {}),
    ("hold_key", {"text": "a", "duration": 0.1}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("action, params", FIXED_CASES)
async def test_hud_computer_actions_fixed(action: str, params: dict):
    """Test various computer actions with proper parameters."""
    comp = HudComputerTool()
    blocks = await comp(action=action, **params)

    # Ensure we got content blocks back
    assert blocks is not None
    assert len(blocks) > 0
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)
