"""Setup function to load custom HTML content."""

import logging
from typing import Any
from hud.tools import BaseSetup, SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup("load_html_content", "Load custom HTML content directly into the browser")
class LoadHtmlContentSetup(BaseSetup):
    """Setup function to load custom HTML content directly into the browser."""

    async def __call__(self, context: Any, html: str, **kwargs) -> SetupResult:
        """Load custom HTML content into the browser.

        Args:
            context: Browser context with playwright_tool
            html: HTML content to load
            **kwargs: Additional arguments

        Returns:
            Status dictionary
        """
        logger.info("Starting load_html_content setup")

        if not html:
            logger.error("No HTML content provided")
            return {"status": "error", "message": "No HTML content provided for load_html_content"}

        # Get playwright tool from context
        if not context or not hasattr(context, "page") or not context.page:
            logger.error("No browser page available")
            return {"status": "error", "message": "No browser page available"}

        page = context.page

        try:
            logger.info("Setting custom HTML content")

            # Set the page content directly
            await page.set_content(html, wait_until="domcontentloaded", timeout=10000)

            # Wait a short time for rendering and scripts to initialize
            await page.wait_for_timeout(1000)

            logger.info("HTML content loaded successfully")

            return {
                "status": "success",
                "message": "HTML content loaded",
                "content_length": len(html),
            }

        except Exception as e:
            logger.error(f"Error loading HTML content: {str(e)}")
            return {"status": "error", "message": f"Failed to load HTML content: {str(e)}"}
