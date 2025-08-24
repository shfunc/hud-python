"""Todo app setup tools."""

import logging
from typing import Dict, Any, List
from fastmcp import Context
from hud.tools.types import SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup.tool("todo_seed")
async def todo_seed(ctx: Context, num_items: int = 5):
    """Seed database with default test todos.

    Args:
        num_items: Number of test items to create (default: 5)

    Returns:
        Setup result with seeded items info
    """
    try:
        # Call the app's seed API
        env = setup.env  # Get BrowserEnvironmentContext from hub
        result = await env.call_app_api("todo", "/api/eval/seed", method="POST")

        # Refresh the page to show the seeded items
        if env.playwright:
            await env.playwright.page.reload()
            import asyncio

            await asyncio.sleep(0.5)  # Wait for page to reload

        return SetupResult(
            content=f"Seeded database with test data",
            info={
                "items_added": result.get("items_added", num_items),
            },
        )
    except Exception as e:
        logger.error(f"todo_seed failed: {e}")
        return SetupResult(content=f"Failed to seed database: {str(e)}", isError=True)


@setup.tool("todo_reset")
async def todo_reset(ctx: Context):
    """Reset database to empty state.

    Returns:
        Setup result with reset confirmation
    """
    try:
        # Call the app's reset API
        env = setup.env  # Get BrowserEnvironmentContext from hub
        result = await env.call_app_api("todo", "/api/eval/reset", method="DELETE")

        return SetupResult(content="Database reset to empty state")
    except Exception as e:
        logger.error(f"todo_reset failed: {e}")
        return SetupResult(content=f"Failed to reset database: {str(e)}", isError=True)


@setup.tool("todo_custom_seed")
async def todo_custom_seed(ctx: Context, items: List[Dict[str, Any]]):
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

        # Call the app's custom seed API (send list directly, not wrapped in dict)
        env = setup.env  # Get BrowserEnvironmentContext from hub
        result = await env.call_app_api(
            "todo", "/api/eval/seed_custom", method="POST", json=formatted_items
        )

        return SetupResult(
            content=f"Seeded database with {len(items)} custom items",
            info={
                "items_added": result.get("items_added", len(items)),
            },
        )
    except Exception as e:
        logger.error(f"todo_custom_seed failed: {e}")
        return SetupResult(content=f"Failed to seed custom items: {str(e)}", isError=True)


@setup.tool("todo_navigate")
async def todo_navigate(ctx: Context, url: str = None):
    """Navigate to the Todo app.

    Args:
        url: Optional custom URL to navigate to

    Returns:
        Setup result with navigation confirmation
    """
    try:
        # Get the default URL if not provided
        env = setup.env  # Get BrowserEnvironmentContext from hub
        if not url:
            url = env.get_app_url("todo")

        # Use Playwright to navigate
        if env.playwright:
            nav_result = await env.playwright.navigate(url)

            return SetupResult(
                content=f"Navigated to Todo app at {url}",
                info={
                    "url": url,
                },
            )
        else:
            return SetupResult(content="Playwright tool not available for navigation", isError=True)
    except Exception as e:
        logger.error(f"todo_navigate failed: {e}")
        return SetupResult(content=f"Failed to navigate to Todo app: {str(e)}", isError=True)
