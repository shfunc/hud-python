"""Todo app setup tools."""

import logging
from typing import Dict, Any, List
import httpx
from mcp.types import TextContent
from . import setup

logger = logging.getLogger(__name__)


@setup.tool("todo_seed")
async def todo_seed(num_items: int = 5):
    """Seed database with default test todos.

    Args:
        num_items: Number of test items to create (default: 5)

    Returns:
        Setup result with seeded items info
    """
    try:
        # Get the backend port from persistent context
        persistent_ctx = setup.env
        backend_port = persistent_ctx.get_app_backend_port("todo")

        # Call the app's seed API
        url = f"http://localhost:{backend_port}/api/eval/seed"
        async with httpx.AsyncClient() as client:
            response = await client.post(url)
            response.raise_for_status()
            result = response.json()

        # Refresh the page to show the seeded items
        playwright_tool = persistent_ctx.get_playwright_tool()
        if playwright_tool:
            await playwright_tool.page.reload()
            import asyncio

            await asyncio.sleep(0.5)  # Wait for page to reload

        return TextContent(
            text=f"Seeded database with test data",
            type="text",
        )
    except Exception as e:
        logger.error(f"todo_seed failed: {e}")
        return TextContent(text=f"Failed to seed database: {str(e)}", type="text")


@setup.tool("todo_reset")
async def todo_reset():
    """Reset database to empty state.

    Returns:
        Setup result with reset confirmation
    """
    try:
        # Get the backend port from persistent context
        persistent_ctx = setup.env
        backend_port = persistent_ctx.get_app_backend_port("todo")

        # Call the app's reset API
        url = f"http://localhost:{backend_port}/api/eval/reset"
        async with httpx.AsyncClient() as client:
            response = await client.delete(url)
            response.raise_for_status()
            result = response.json()

        return TextContent(text="Database reset to empty state", type="text")
    except Exception as e:
        logger.error(f"todo_reset failed: {e}")
        return TextContent(text=f"Failed to reset database: {str(e)}", type="text")


@setup.tool("todo_custom_seed")
async def todo_custom_seed(items: List[Dict[str, Any]]):
    """Seed database with custom todo items.

    Args:
        items: List of todo items to create

    Returns:
        Setup result with seeded items info
    """
    try:
        # Ensure all items have required fields
        formatted_items = []
        for item in items:
            formatted_item = {
                "title": item.get("title", ""),
                "description": item.get("description", ""),  # Add empty description if not provided
                "completed": item.get("completed", False),
            }
            formatted_items.append(formatted_item)

        # Get the backend port from persistent context
        persistent_ctx = setup.env
        backend_port = persistent_ctx.get_app_backend_port("todo")

        # Call the app's custom seed API (send list directly, not wrapped in dict)
        url = f"http://localhost:{backend_port}/api/eval/seed_custom"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=formatted_items)
            response.raise_for_status()
            result = response.json()

        return TextContent(
            text=f"Seeded database with {len(items)} custom items",
            type="text",
        )
    except Exception as e:
        logger.error(f"todo_custom_seed failed: {e}")
        return TextContent(text=f"Failed to seed custom items: {str(e)}", type="text")


@setup.tool("todo_navigate")
async def todo_navigate(url: str):
    """Navigate to the Todo app.

    Args:
        url: Optional custom URL to navigate to

    Returns:
        Setup result with navigation confirmation
    """
    try:
        # Get the persistent context
        persistent_ctx = setup.env

        # Get the default URL if not provided
        if not url:
            url = persistent_ctx.get_app_url("todo")

        # Use Playwright to navigate
        playwright_tool = persistent_ctx.get_playwright_tool()
        if playwright_tool:
            nav_result = await playwright_tool.navigate(url)

            return TextContent(
                text=f"Navigated to Todo app at {url}",
                type="text",
            )
        else:
            return TextContent(text="Playwright tool not available for navigation", type="text")
    except Exception as e:
        logger.error(f"todo_navigate failed: {e}")
        return TextContent(text=f"Failed to navigate to Todo app: {str(e)}", type="text")
