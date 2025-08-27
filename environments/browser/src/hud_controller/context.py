"""
Context server for browser environment that persists state across hot-reloads.

Run this as a separate process to maintain browser/app state during development:
    python -m hud_controller.context
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from hud.server.context import run_context_server
from .services import ServiceManager

logger = logging.getLogger(__name__)


class BrowserContext:
    """Simple context that holds browser environment state across reloads."""

    def __init__(self):
        """Initialize the browser context."""
        self.service_manager = ServiceManager()
        self.is_initialized = False
        self._running_apps: List[str] = []
        self._startup_complete = False

        logger.info("[BrowserContext] Created new browser context")

    def startup(self):
        """One-time startup when context server starts."""
        if self._startup_complete:
            logger.info("[BrowserContext] Startup already complete, skipping")
            return

        logger.info("[BrowserContext] Performing one-time startup")
        self._startup_complete = True

    # === Proxy-friendly methods for multiprocessing.Manager ===
    # Note: These are needed because direct attribute access doesn't always
    # work correctly through the multiprocessing proxy, especially for
    # complex operations like list modifications

    def get_service_manager(self) -> ServiceManager:
        """Get the service manager instance."""
        return self.service_manager

    def get_is_initialized(self) -> bool:
        """Check if environment is initialized."""
        return self.is_initialized

    def set_initialized(self, value: bool) -> None:
        """Set initialization status."""
        self.is_initialized = value
        logger.info(f"[BrowserContext] Initialization status: {value}")

    def get_running_apps(self) -> List[str]:
        """Get list of running apps."""
        return self._running_apps.copy()

    def add_running_app(self, app_name: str) -> None:
        """Add app to running list."""
        if app_name not in self._running_apps:
            self._running_apps.append(app_name)
            logger.info(f"[BrowserContext] Added running app: {app_name}")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "is_initialized": self.is_initialized,
            "startup_complete": self._startup_complete,
            "running_apps": self._running_apps.copy(),
        }

    # === Compatibility methods for setup/evaluate functions ===

    async def call_app_api(
        self, app_name: str, endpoint: str, method: str = "GET", json: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Call an app's backend API.

        Args:
            app_name: Name of the app (e.g., '2048', 'todo')
            endpoint: API endpoint (e.g., '/api/game/new')
            method: HTTP method
            json: JSON data for POST/PUT requests

        Returns:
            API response data
        """
        import httpx

        # Get the backend port for the app
        backend_port = self.service_manager.get_app_port(app_name)
        url = f"http://localhost:{backend_port}{endpoint}"

        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, json=json)
            response.raise_for_status()
            return response.json()


if __name__ == "__main__":
    # Run the context server with BrowserContext
    context = BrowserContext()
    context.startup()

    # Log initial state
    logger.info(f"[Context] Starting browser context server")
    logger.info(f"[Context] Initial state: {context.get_state_summary()}")

    # Run the context server
    asyncio.run(run_context_server(context, "/tmp/hud_browser_ctx.sock"))
