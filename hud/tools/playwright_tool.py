"""Playwright web automation tool for HUD."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ImageContent, TextContent
from pydantic import Field

from hud.tools.base import BaseTool, ToolResult, tool_result_to_content_blocks

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

logger = logging.getLogger(__name__)


class PlaywrightTool(BaseTool):
    """Playwright tool for web automation."""

    def __init__(self) -> None:
        super().__init__()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def __call__(
        self,
        action: str = Field(
            ...,
            description="The action to perform (navigate, screenshot, click, type, get_page_info, wait_for_element)",  # noqa: E501
        ),
        url: str | None = Field(None, description="URL to navigate to (for navigate action)"),
        selector: str | None = Field(
            None, description="CSS selector for element (for click, type, wait_for_element actions)"
        ),
        text: str | None = Field(None, description="Text to type (for type action)"),
        path: str | None = Field(
            None, description="File path to save screenshot (for screenshot action)"
        ),
        wait_for_load_state: str | None = Field(
            None,
            description="State to wait for: load, domcontentloaded, networkidle (default: networkidle)",  # noqa: E501
        ),
        timeout: int | None = Field(
            None, description="Timeout in milliseconds for wait_for_element (default: 30000)"
        ),
    ) -> list[ImageContent | TextContent]:
        """
        Execute a Playwright web automation action.

        Returns:
            List of MCP content blocks
        """
        logger.info("PlaywrightTool executing action: %s", action)

        try:
            if action == "navigate":
                if url is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="url parameter is required for navigate"
                        )
                    )
                result = await self.navigate(url, wait_for_load_state or "networkidle")

            elif action == "screenshot":
                result = await self.screenshot(path)

            elif action == "click":
                if selector is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="selector parameter is required for click"
                        )
                    )
                result = await self.click(selector)

            elif action == "type":
                if selector is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="selector parameter is required for type"
                        )
                    )
                if text is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="text parameter is required for type"
                        )
                    )
                result = await self.type_text(selector, text)

            elif action == "get_page_info":
                result = await self.get_page_info()

            elif action == "wait_for_element":
                if selector is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message="selector parameter is required for wait_for_element",
                        )
                    )
                result = await self.wait_for_element(selector, timeout or 30000)

            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown action: {action}"))

            # Convert dict result to ToolResult
            if isinstance(result, dict):
                if result.get("success"):
                    if "screenshot" in result:
                        # Return screenshot as image content
                        tool_result = ToolResult(
                            output=result.get("message", ""), base64_image=result["screenshot"]
                        )
                    else:
                        tool_result = ToolResult(output=result.get("message", ""))
                else:
                    tool_result = ToolResult(error=result.get("error", "Unknown error"))
            else:
                tool_result = result

            # Convert result to content blocks
            return tool_result_to_content_blocks(tool_result)

        except McpError:
            raise
        except Exception as e:
            logger.error("PlaywrightTool error: %s", e)
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Playwright error: {e}")) from e

    async def _ensure_browser(self) -> None:
        """Ensure browser is launched and ready."""
        if self._browser is None or not self._browser.is_connected():
            logger.info("Launching Playwright browser...")

            # Ensure DISPLAY is set
            os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":1")

            if self._playwright is None:
                from playwright.async_api import async_playwright

                self._playwright = await async_playwright().start()

            self._browser = await self._playwright.chromium.launch(
                headless=False,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-blink-features=AutomationControlled",
                    "--window-size=1920,1080",
                    "--window-position=0,0",
                    "--start-maximized",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=TranslateUI",
                    "--disable-ipc-flooding-protection",
                    "--disable-default-apps",
                    "--no-first-run",
                    "--disable-sync",
                    "--no-default-browser-check",
                ],
            )

            self._context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True,
            )

            self._page = await self._context.new_page()
            logger.info("Playwright browser launched successfully")

    async def navigate(self, url: str, wait_for_load_state: str = "networkidle") -> dict[str, Any]:
        """Navigate to a URL.

        Args:
            url: URL to navigate to
            wait_for_load_state: Load state to wait for (load, domcontentloaded, networkidle)

        Returns:
            Dict with navigation result
        """
        await self._ensure_browser()

        logger.info("Navigating to %s", url)
        try:
            await self._page.goto(url, wait_until=wait_for_load_state)
            current_url = self._page.url
            title = await self._page.title()

            return {
                "success": True,
                "url": current_url,
                "title": title,
                "message": f"Successfully navigated to {url}",
            }
        except Exception as e:
            logger.error("Navigation failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to navigate to {url}: {e}",
            }

    async def screenshot(self, path: str | None = None) -> dict[str, Any]:
        """Take a screenshot of the current page.

        Args:
            path: Optional path to save screenshot

        Returns:
            Dict with screenshot result
        """
        await self._ensure_browser()

        try:
            if path:
                await self._page.screenshot(path=path, full_page=True)
                return {"success": True, "path": path, "message": f"Screenshot saved to {path}"}
            else:
                # Return base64 encoded screenshot
                screenshot_bytes = await self._page.screenshot(full_page=True)
                import base64

                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                return {
                    "success": True,
                    "screenshot": screenshot_b64,
                    "message": "Screenshot captured",
                }
        except Exception as e:
            logger.error("Screenshot failed: %s", e)
            return {"success": False, "error": str(e), "message": f"Failed to take screenshot: {e}"}

    async def click(self, selector: str) -> dict[str, Any]:
        """Click an element by selector.

        Args:
            selector: CSS selector for element to click

        Returns:
            Dict with click result
        """
        await self._ensure_browser()

        try:
            await self._page.click(selector)
            return {"success": True, "message": f"Clicked element: {selector}"}
        except Exception as e:
            logger.error("Click failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to click {selector}: {e}",
            }

    async def type_text(self, selector: str, text: str) -> dict[str, Any]:
        """Type text into an element.

        Args:
            selector: CSS selector for input element
            text: Text to type

        Returns:
            Dict with type result
        """
        await self._ensure_browser()

        try:
            await self._page.fill(selector, text)
            return {"success": True, "message": f"Typed '{text}' into {selector}"}
        except Exception as e:
            logger.error("Type failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to type into {selector}: {e}",
            }

    async def get_page_info(self) -> dict[str, Any]:
        """Get current page information.

        Returns:
            Dict with page info
        """
        await self._ensure_browser()

        try:
            url = self._page.url
            title = await self._page.title()
            return {
                "success": True,
                "url": url,
                "title": title,
                "message": f"Current page: {title} ({url})",
            }
        except Exception as e:
            logger.error("Get page info failed: %s", e)
            return {"success": False, "error": str(e), "message": f"Failed to get page info: {e}"}

    async def wait_for_element(self, selector: str, timeout: int = 30000) -> dict[str, Any]:
        """Wait for an element to appear.

        Args:
            selector: CSS selector for element
            timeout: Timeout in milliseconds

        Returns:
            Dict with wait result
        """
        await self._ensure_browser()

        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            return {"success": True, "message": f"Element {selector} appeared"}
        except Exception as e:
            logger.error("Wait for element failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Element {selector} did not appear within {timeout}ms: {e}",
            }

    async def close(self) -> None:
        """Close browser and cleanup."""
        if self._browser:
            try:
                await self._browser.close()
                logger.info("Browser closed")
            except Exception as e:
                logger.error("Error closing browser: %s", e)

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.error("Error stopping playwright: %s", e)

        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None
