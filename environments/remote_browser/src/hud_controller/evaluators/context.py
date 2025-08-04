"""Context for remote browser environment evaluation."""

import logging
from typing import Optional, Any
from datetime import datetime
from hud.tools.playwright_tool import PlaywrightTool
from ..providers.base import BrowserProvider

logger = logging.getLogger(__name__)


class RemoteBrowserContext:
    """Context for remote browser evaluations.

    Provides access to the browser provider and playwright tool
    for evaluators and setup functions.
    """

    def __init__(
        self,
        browser_provider: BrowserProvider,
        playwright_tool=None,
    ):
        """Initialize context with browser provider and tools.

        Args:
            browser_provider: The active browser provider instance
            playwright_tool: PlaywrightToolWithMemory instance
        """
        self.browser_provider = browser_provider
        self.playwright_tool = playwright_tool

    @property
    def page(self):
        """Get the current Playwright page."""
        if self.playwright_tool:
            return self.playwright_tool.page
        return None

    @property
    def context(self):
        """Get the browser context."""
        if self.playwright_tool and self.playwright_tool._context:
            return self.playwright_tool._context
        return None

    def get_provider_info(self) -> dict[str, Any]:
        """Get information about the current browser provider."""
        return {
            "provider": self.browser_provider.__class__.__name__,
            "is_running": self.browser_provider.is_running,
            "cdp_url": self.browser_provider.cdp_url,
        }

    # Delegate history tracking methods to playwright tool
    def get_navigation_count(self) -> int:
        """Get the number of navigations that have occurred."""
        if self.playwright_tool:
            return self.playwright_tool.get_navigation_count()
        return 0

    def get_last_action(self) -> Optional[dict]:
        """Get the last action performed."""
        if self.playwright_tool:
            return self.playwright_tool.get_last_action()
        return None

    def get_selector_at_index(self, index: int) -> Optional[str]:
        """Get selector at specific index in history."""
        if self.playwright_tool:
            return self.playwright_tool.get_selector_at_index(index)
        return None

    async def close(self):
        """Cleanup resources if needed."""
        # Context cleanup if needed
        pass
