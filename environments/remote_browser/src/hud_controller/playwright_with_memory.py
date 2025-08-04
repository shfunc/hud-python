"""PlaywrightTool with memory/history tracking for remote browser environment."""

import logging
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import Field
from hud.tools.playwright_tool import PlaywrightTool
from mcp.types import ImageContent, TextContent

logger = logging.getLogger(__name__)


class PlaywrightToolWithMemory(PlaywrightTool):
    """Extended PlaywrightTool that tracks navigation and action history."""

    def __init__(self, cdp_url: str | None = None) -> None:
        """Initialize with history tracking capabilities."""
        super().__init__(cdp_url)

        # Initialize history tracking
        self.navigation_history: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.selector_history: List[str] = []

    async def _ensure_browser(self) -> None:
        """Ensure browser is launched and setup event listeners."""
        await super()._ensure_browser()

        # Setup event listeners for tracking
        if self._page:
            self._setup_event_listeners()

    def _setup_event_listeners(self) -> None:
        """Setup event listeners to track navigation history."""
        if not self._page:
            return

        try:
            # Track frame navigations (includes main frame)
            def on_frame_navigated(frame):
                if self._page is None:
                    return
                if frame == self._page.main_frame:
                    self.navigation_history.append(
                        {"url": frame.url, "timestamp": datetime.now().isoformat()}
                    )
                    logger.debug(f"Navigation tracked: {frame.url}")

            self._page.on("framenavigated", on_frame_navigated)

            # Track initial page URL
            if self._page.url and self._page.url != "about:blank":
                self.navigation_history.append(
                    {"url": self._page.url, "timestamp": datetime.now().isoformat()}
                )

        except Exception as e:
            logger.warning(f"Failed to setup event listeners: {e}")

    def _track_action(self, action_type: str, details: Dict[str, Any]) -> None:
        """Track an action in history."""
        self.action_history.append(
            {"type": action_type, "details": details, "timestamp": datetime.now().isoformat()}
        )

        # Track selectors if present
        if "selector" in details:
            self.selector_history.append(details["selector"])

        logger.debug(f"Action tracked: {action_type} - {details}")

    async def __call__(
        self,
        action: str = Field(
            ...,
            description="The action to perform (navigate, screenshot, click, type, get_page_info, wait_for_element)",
        ),
        url: str | None = Field(None, description="URL to navigate to (for navigate action)"),
        selector: str | None = Field(
            None, description="CSS selector for element (for click, type, wait_for_element actions)"
        ),
        text: str | None = Field(None, description="Text to type (for type action)"),
        wait_for_load_state: Literal["commit", "domcontentloaded", "load", "networkidle"]
        | None = Field(
            None,
            description="State to wait for: commit, domcontentloaded, load, networkidle (default: networkidle)",
        ),
    ) -> list[ImageContent | TextContent]:
        """Execute a Playwright action with tracking."""

        # Track the action before executing
        action_details = {
            "action": action,
            "url": url,
            "selector": selector,
            "text": text,
            "wait_for_load_state": wait_for_load_state,
        }
        # Remove None values
        action_details = {k: v for k, v in action_details.items() if v is not None}

        # Call parent implementation
        result = await super().__call__(
            action=action,
            url=url,
            selector=selector,
            text=text,
            wait_for_load_state=wait_for_load_state,
        )

        # Track the action after execution
        self._track_action(action, action_details)

        return result

    # History access methods
    def get_navigation_count(self) -> int:
        """Get the number of navigations that have occurred."""
        return len(self.navigation_history)

    def get_last_action(self) -> Optional[Dict[str, Any]]:
        """Get the last action performed."""
        return self.action_history[-1] if self.action_history else None

    def get_selector_at_index(self, index: int) -> Optional[str]:
        """Get selector at specific index in history."""
        if 0 <= index < len(self.selector_history):
            return self.selector_history[index]
        return None

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get full action history."""
        return self.action_history.copy()

    def get_navigation_history(self) -> List[Dict[str, Any]]:
        """Get full navigation history."""
        return self.navigation_history.copy()

    def clear_history(self) -> None:
        """Clear all history tracking."""
        self.navigation_history.clear()
        self.action_history.clear()
        self.selector_history.clear()
        logger.info("Cleared all history tracking")
