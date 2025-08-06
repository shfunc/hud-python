"""Interaction setup functions for remote browser environment."""

import logging
from typing import Any
from hud.tools import BaseSetup, SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup("click_element", "Click on an element by selector")
class ClickElementSetup(BaseSetup):
    """Setup function to click on an element."""

    async def __call__(
        self, context: Any, selector: str, timeout: int = 30000, **kwargs
    ) -> SetupResult:
        """Click on an element identified by a CSS selector.

        Args:
            context: Browser context with playwright_tool
            selector: CSS selector for the element to click
            timeout: Maximum time to wait for the element (milliseconds)
            **kwargs: Additional arguments

        Returns:
            Setup result dictionary
        """
        logger.info(f"Clicking element: {selector}")

        # Get the playwright tool from context
        playwright_tool = context.playwright_tool if hasattr(context, "playwright_tool") else None
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


@setup("type_text", "Type text into an input element")
class TypeTextSetup(BaseSetup):
    """Setup function to type text into an element."""

    async def __call__(
        self, context: Any, selector: str, text: str, clear_first: bool = False, **kwargs
    ) -> SetupResult:
        """Type text into an input element.

        Args:
            context: Browser context with playwright_tool
            selector: CSS selector for the input element
            text: Text to type
            clear_first: Whether to clear the input before typing
            **kwargs: Additional arguments

        Returns:
            Setup result dictionary
        """
        logger.info(f"Typing into element: {selector}")

        # Get the playwright tool from context
        if not context or not hasattr(context, "page") or not context.page:
            logger.error("No browser page available")
            return {
                "status": "error",
                "message": "No browser page available",
            }

        page = context.page

        try:
            # Clear the input if requested
            if clear_first:
                await page.fill(selector, "")

            # Type the text (tracking is handled by PlaywrightToolWithMemory)
            await page.type(selector, text)

            # Record this in action history if the playwright tool supports it
            if hasattr(context, "_record_action"):
                context._record_action("type", {"selector": selector, "text": text})

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


@setup("wait_for_element", "Wait for an element to appear")
class WaitForElementSetup(BaseSetup):
    """Setup function to wait for an element."""

    async def __call__(
        self, context: Any, selector: str, timeout: int = 30000, state: str = "visible", **kwargs
    ) -> SetupResult:
        """Wait for an element to reach a certain state.

        Args:
            context: Browser context with playwright_tool
            selector: CSS selector for the element
            timeout: Maximum time to wait (milliseconds)
            state: State to wait for ("visible", "hidden", "attached", "detached")
            **kwargs: Additional arguments

        Returns:
            Setup result dictionary
        """
        logger.info(f"Waiting for element: {selector} (state: {state})")

        # Get the playwright tool from context
        playwright_tool = context.playwright_tool if hasattr(context, "playwright_tool") else None
        if not playwright_tool or not playwright_tool.page:
            logger.error("No browser page available")
            return {
                "status": "error",
                "message": "No browser page available",
            }

        page = playwright_tool.page

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
