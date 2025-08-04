"""Navigation setup for remote browser environment."""

import logging
from ..setup import setup

logger = logging.getLogger(__name__)


@setup("navigate_to_url", description="Navigate browser to a specific URL")
class NavigateSetup:
    """Setup function to navigate to a URL."""

    def __init__(self, context):
        self.context = context

    async def __call__(self, url: str, wait_for_load_state: str = "networkidle") -> dict:
        """
        Navigate to the specified URL.

        Args:
            url: The URL to navigate to
            wait_for_load_state: State to wait for after navigation

        Returns:
            Setup result dictionary
        """
        logger.info(f"Navigating to URL: {url}")

        # Get the playwright tool from context
        playwright_tool = self.context.playwright_tool
        if not playwright_tool:
            logger.error("No playwright tool available in context")
            return {
                "status": "error",
                "message": "No browser available for navigation",
            }

        # Navigate using the playwright tool (tracking is handled by PlaywrightToolWithMemory)
        result = await playwright_tool.navigate(url, wait_for_load_state)

        if result.get("success"):
            logger.info(f"Successfully navigated to {url}")
            return {
                "status": "success",
                "message": f"Navigated to {url}",
                "url": result.get("url"),
                "title": result.get("title"),
            }
        else:
            logger.error(f"Failed to navigate: {result.get('error')}")
            return {
                "status": "error",
                "message": f"Navigation failed: {result.get('error')}",
            }
