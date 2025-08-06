"""Navigation setup for remote browser environment."""

import logging
from typing import Any
from hud.tools import BaseSetup, SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup("navigate_to_url", "Navigate browser to a specific URL")
class NavigateSetup(BaseSetup):
    """Setup function to navigate to a URL."""

    async def __call__(self, context: Any, url: str, wait_for_load_state: str = "networkidle", **kwargs) -> SetupResult:
        """Navigate to the specified URL.
        
        Args:
            context: Browser context with playwright_tool
            url: The URL to navigate to
            wait_for_load_state: State to wait for after navigation
            **kwargs: Additional arguments
            
        Returns:
            Setup result dictionary
        """
        logger.info(f"Navigating to URL: {url}")
        
        # Get the playwright tool from context
        if not context or not hasattr(context, 'page') or not context.page:
            logger.error("No playwright tool available in context")
            return {
                "status": "error",
                "message": "No browser available for navigation",
            }
        
        # Navigate using the playwright tool (tracking is handled by PlaywrightToolWithMemory)
        result = await context.navigate(url, wait_for_load_state)
        
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