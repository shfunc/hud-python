"""Cookie setup functions for remote browser environment."""

import logging
from typing import Any, List, Dict
from hud.tools import BaseSetup, SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup("set_cookies", "Set cookies in the browser")
class SetCookiesSetup(BaseSetup):
    """Setup function to set cookies."""

    async def __call__(self, context: Any, cookies: List[Dict[str, Any]], **kwargs) -> SetupResult:
        """Set cookies in the browser context.
        
        Args:
            context: Browser context with playwright_tool
            cookies: List of cookie dictionaries with name, value, and optional properties
            **kwargs: Additional arguments
            
        Returns:
            Setup result dictionary
        """
        logger.info(f"Setting {len(cookies)} cookies")
        
        # Get the playwright tool from context
        if not context or not hasattr(context, 'page') or not context.page:
            logger.error("No browser page available")
            return {
                "status": "error",
                "message": "No browser page available",
            }
        
        try:
            # Add cookies to the context
            await context.page.context.add_cookies(cookies)
            
            logger.info(f"Successfully set {len(cookies)} cookies")
            return {
                "status": "success",
                "message": f"Set {len(cookies)} cookies",
                "cookies_set": [c.get("name") for c in cookies],
            }
        except Exception as e:
            logger.error(f"Failed to set cookies: {e}")
            return {
                "status": "error",
                "message": f"Failed to set cookies: {str(e)}",
            }


@setup("clear_cookies", "Clear all cookies from the browser")
class ClearCookiesSetup(BaseSetup):
    """Setup function to clear all cookies."""

    async def __call__(self, context: Any, **kwargs) -> SetupResult:
        """Clear all cookies from the browser context.
        
        Args:
            context: Browser context with playwright_tool
            **kwargs: Additional arguments
            
        Returns:
            Setup result dictionary
        """
        logger.info("Clearing all cookies")
        
        # Get the playwright tool from context
        if not context or not hasattr(context, 'page') or not context.page:
            logger.error("No browser page available")
            return {
                "status": "error",
                "message": "No browser page available",
            }
        
        try:
            # Clear all cookies
            await context.page.context.clear_cookies()
            
            logger.info("Successfully cleared all cookies")
            return {
                "status": "success",
                "message": "Cleared all cookies",
            }
        except Exception as e:
            logger.error(f"Failed to clear cookies: {e}")
            return {
                "status": "error",
                "message": f"Failed to clear cookies: {str(e)}",
            }