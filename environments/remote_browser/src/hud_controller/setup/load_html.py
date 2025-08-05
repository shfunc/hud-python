"""Setup function to load custom HTML content."""

import logging
from typing import Dict, Any
from .registry import setup
from ..evaluators.context import RemoteBrowserContext

logger = logging.getLogger(__name__)


@setup("load_html_content")
class LoadHtmlContentSetup:
    """Setup function to load custom HTML content directly into the browser."""

    name = "load_html_content"

    def __init__(self, context):
        self.context = context

    async def __call__(self, html: str) -> Dict[str, Any]:
        """
        Load custom HTML content into the browser.

        Args:
            html: HTML content to load

        Returns:
            Status dictionary
        """
        logger.info("Starting load_html_content setup")

        if not html:
            logger.error("No HTML content provided")
            return {"status": "error", "message": "No HTML content provided for load_html_content"}

        # Get page from context
        page = self.context.page
        if not page:
            logger.error("No page available in context")
            return {"status": "error", "message": "No browser page available"}

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
