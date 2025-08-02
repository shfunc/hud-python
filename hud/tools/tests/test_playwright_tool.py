"""Tests for PlaywrightTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import INVALID_PARAMS, ImageContent, TextContent

from hud.tools.playwright_tool import PlaywrightTool


@pytest.mark.asyncio
async def test_playwright_tool_init():
    """Test PlaywrightTool initialization."""
    tool = PlaywrightTool()
    assert tool._browser is None
    assert tool._context is None
    assert tool._page is None


@pytest.mark.asyncio
async def test_playwright_tool_navigate_requires_url():
    """Test that navigate action requires url parameter."""
    tool = PlaywrightTool()

    with pytest.raises(McpError) as exc_info:
        await tool(action="navigate")

    assert exc_info.value.error.code == INVALID_PARAMS
    assert "url parameter is required" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_playwright_tool_click_requires_selector():
    """Test that click action requires selector parameter."""
    tool = PlaywrightTool()

    with pytest.raises(McpError) as exc_info:
        await tool(action="click")

    assert exc_info.value.error.code == INVALID_PARAMS
    assert "selector parameter is required" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_playwright_tool_type_requires_params():
    """Test that type action requires both selector and text."""
    tool = PlaywrightTool()

    # Missing selector
    with pytest.raises(McpError) as exc_info:
        await tool(action="type", text="hello")
    assert "selector parameter is required" in exc_info.value.error.message

    # Missing text
    with pytest.raises(McpError) as exc_info:
        await tool(action="type", selector="#input")
    assert "text parameter is required" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_playwright_tool_screenshot_action():
    """Test screenshot action with mocked page."""
    tool = PlaywrightTool()

    # Mock the page and browser
    mock_page = AsyncMock()
    mock_page.screenshot.return_value = b"fake_screenshot_data"

    with (
        patch.object(tool, "_ensure_browser", new_callable=AsyncMock),
        patch.object(tool, "page", new=mock_page),
    ):
        blocks = await tool(action="screenshot")

    # Should return content blocks
    assert len(blocks) > 0
    assert any(isinstance(b, ImageContent | TextContent) for b in blocks)


@pytest.mark.asyncio
async def test_playwright_tool_get_page_info():
    """Test get_page_info action."""
    tool = PlaywrightTool()

    # Mock the page
    mock_page = AsyncMock()
    mock_page.url = "https://example.com"
    mock_page.title.return_value = "Example Page"
    mock_page.evaluate.return_value = {"height": 1000}

    with (
        patch.object(tool, "_ensure_browser", new_callable=AsyncMock),
        patch.object(tool, "page", new=mock_page),
    ):
        blocks = await tool(action="get_page_info")

    # Should return content blocks with page info
    assert len(blocks) > 0
    assert any(isinstance(b, TextContent) for b in blocks)


@pytest.mark.asyncio
async def test_playwright_tool_wait_for_element():
    """Test wait_for_element action."""
    tool = PlaywrightTool()

    # Missing selector
    with pytest.raises(McpError) as exc_info:
        await tool(action="wait_for_element")

    assert "selector parameter is required" in exc_info.value.error.message

    # With selector (mocked)
    mock_page = AsyncMock()
    mock_page.wait_for_selector.return_value = MagicMock()

    with (
        patch.object(tool, "_ensure_browser", new_callable=AsyncMock),
        patch.object(tool, "page", new=mock_page),
    ):
        blocks = await tool(action="wait_for_element", selector="#test")

    # Should return content blocks
    assert len(blocks) > 0
    mock_page.wait_for_selector.assert_called_once()


@pytest.mark.asyncio
async def test_playwright_tool_unknown_action():
    """Test that unknown action raises error."""
    tool = PlaywrightTool()

    with pytest.raises(McpError) as exc_info:
        await tool(action="unknown_action")

    assert exc_info.value.error.code == INVALID_PARAMS
    assert "Unknown action" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_playwright_tool_cleanup():
    """Test cleanup closes browser properly."""
    tool = PlaywrightTool()

    # Mock browser and context
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    tool._browser = mock_browser
    tool._context = mock_context

    await tool._cleanup()

    mock_context.close.assert_called_once()
    mock_browser.close.assert_called_once()
    assert tool._browser is None
    assert tool._context is None
    assert tool._page is None
