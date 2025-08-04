"""Interaction setup functions for remote browser environment."""

import logging
from ..setup import setup

logger = logging.getLogger(__name__)


@setup("click_element", description="Click on an element by selector")
class ClickElementSetup:
    """Setup function to click on an element."""

    def __init__(self, context):
        self.context = context

    async def __call__(self, selector: str, timeout: int = 30000) -> dict:
        """
        Click on an element identified by a CSS selector.

        Args:
            selector: CSS selector for the element to click
            timeout: Maximum time to wait for the element (milliseconds)

        Returns:
            Setup result dictionary
        """
        logger.info(f"Clicking element: {selector}")

        # Get the playwright tool from context
        playwright_tool = self.context.playwright_tool
        if not playwright_tool:
            logger.error("No playwright tool available")
            return {
                "status": "error",
                "message": "No browser available",
            }

        # Click the element (tracking is handled by PlaywrightToolWithMemory)
        result = await playwright_tool.click(selector)

        if result.get("success"):
            logger.info(f"Successfully clicked: {selector}")
            return {
                "status": "success",
                "message": f"Clicked element: {selector}",
            }
        else:
            logger.error(f"Failed to click: {result.get('error')}")
            return {
                "status": "error",
                "message": f"Failed to click element: {result.get('error')}",
            }


@setup("type_text", description="Type text into an input element")
class TypeTextSetup:
    """Setup function to type text into an element."""

    def __init__(self, context):
        self.context = context

    async def __call__(self, selector: str, text: str, clear_first: bool = False) -> dict:
        """
        Type text into an input element.

        Args:
            selector: CSS selector for the input element
            text: Text to type
            clear_first: Whether to clear the input before typing

        Returns:
            Setup result dictionary
        """
        logger.info(f"Typing into element: {selector}")

        # Get the page from context
        page = self.context.page
        if not page:
            logger.error("No page available")
            return {
                "status": "error",
                "message": "No browser page available",
            }

        try:
            # Clear the input if requested
            if clear_first:
                await page.fill(selector, "")

            # Type the text (tracking is handled by PlaywrightToolWithMemory)
            await page.type(selector, text)

            logger.info(f"Successfully typed text into: {selector}")
            return {
                "status": "success",
                "message": f"Typed text into element: {selector}",
                "selector": selector,
                "text_length": len(text),
            }
        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return {
                "status": "error",
                "message": f"Failed to type text: {str(e)}",
            }


@setup("wait_for_element", description="Wait for an element to appear")
class WaitForElementSetup:
    """Setup function to wait for an element."""

    def __init__(self, context):
        self.context = context

    async def __call__(self, selector: str, timeout: int = 30000, state: str = "visible") -> dict:
        """
        Wait for an element to reach a certain state.

        Args:
            selector: CSS selector for the element
            timeout: Maximum time to wait (milliseconds)
            state: State to wait for ("visible", "hidden", "attached", "detached")

        Returns:
            Setup result dictionary
        """
        logger.info(f"Waiting for element: {selector} (state: {state})")

        # Get the page from context
        page = self.context.page
        if not page:
            logger.error("No page available")
            return {
                "status": "error",
                "message": "No browser page available",
            }

        try:
            # Wait for the element
            await page.wait_for_selector(selector, timeout=timeout, state=state)

            logger.info(f"Element found: {selector}")
            return {
                "status": "success",
                "message": f"Element is {state}: {selector}",
                "selector": selector,
                "state": state,
            }
        except Exception as e:
            logger.error(f"Failed to wait for element: {e}")
            return {
                "status": "error",
                "message": f"Timeout waiting for element: {str(e)}",
            }
