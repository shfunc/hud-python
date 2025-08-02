from __future__ import annotations

import pytest
from mcp.types import ImageContent, TextContent

from hud.tools.computer.anthropic import AnthropicComputerTool
from hud.tools.computer.hud import HudComputerTool
from hud.tools.computer.openai import OpenAIComputerTool


@pytest.mark.asyncio
async def test_hud_computer_screenshot():
    comp = HudComputerTool()
    blocks = await comp(action="screenshot")
    # Screenshot might return ImageContent or TextContent (if error)
    assert blocks is not None
    assert len(blocks) > 0
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)


@pytest.mark.asyncio
async def test_hud_computer_click_simulation():
    comp = HudComputerTool()
    blocks = await comp(action="click", x=10, y=10)
    # Should return text confirming execution or screenshot block
    assert blocks
    assert len(blocks) > 0


@pytest.mark.asyncio
async def test_openai_computer_screenshot():
    comp = OpenAIComputerTool()
    blocks = await comp(type="screenshot")
    assert blocks is not None
    assert len(blocks) > 0
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)


@pytest.mark.asyncio
async def test_anthropic_computer_screenshot():
    comp = AnthropicComputerTool()
    blocks = await comp(action="screenshot")
    assert blocks is not None
    assert len(blocks) > 0
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)


@pytest.mark.asyncio
async def test_openai_computer_click():
    comp = OpenAIComputerTool()
    blocks = await comp(type="click", x=5, y=5)
    assert blocks
