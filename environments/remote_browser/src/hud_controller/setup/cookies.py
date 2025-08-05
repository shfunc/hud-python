"""Cookie setup functions for remote browser environment."""

import logging
from typing import List, Dict, Any
from ..setup import setup

logger = logging.getLogger(__name__)


@setup("set_cookies", description="Set cookies in the browser")
class SetCookiesSetup:
    """Setup function to set cookies."""

    def __init__(self, context):
        self.context = context

    async def __call__(self, cookies: List[Dict[str, Any]]) -> dict:
        """
        Set cookies in the browser context.

        Args:
            cookies: List of cookie dictionaries with name, value, and optional properties

        Returns:
            Setup result dictionary
        """
        logger.info(f"Setting {len(cookies)} cookies")

        # Get the browser context
        browser_context = self.context.context
        if not browser_context:
            logger.error("No browser context available")
            return {
                "status": "error",
                "message": "No browser context available",
            }

        try:
            # Add cookies to the context
            await browser_context.add_cookies(cookies)

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


@setup("clear_cookies", description="Clear all cookies from the browser")
class ClearCookiesSetup:
    """Setup function to clear all cookies."""

    def __init__(self, context):
        self.context = context

    async def __call__(self) -> dict:
        """
        Clear all cookies from the browser context.

        Returns:
            Setup result dictionary
        """
        logger.info("Clearing all cookies")

        # Get the browser context
        browser_context = self.context.context
        if not browser_context:
            logger.error("No browser context available")
            return {
                "status": "error",
                "message": "No browser context available",
            }

        try:
            # Clear all cookies
            await browser_context.clear_cookies()

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
