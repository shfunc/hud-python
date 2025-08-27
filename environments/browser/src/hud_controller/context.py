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
        self._app_ports: Dict[str, Dict[str, int]] = {}  # Track app ports
        self._startup_complete = False
        self.playwright_tool = None  # Store the playwright tool

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

    def set_app_ports(self, app_name: str, frontend_port: int, backend_port: int) -> None:
        """Set port information for an app."""
        self._app_ports[app_name] = {"frontend": frontend_port, "backend": backend_port}
        logger.info(
            f"[BrowserContext] Set ports for {app_name}: frontend={frontend_port}, backend={backend_port}"
        )

    def get_playwright_tool(self):
        """Get the playwright tool instance."""
        return self.playwright_tool

    def set_playwright_tool(self, tool) -> None:
        """Set the playwright tool instance."""
        self.playwright_tool = tool
        logger.info(f"[BrowserContext] Set playwright tool")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "is_initialized": self.is_initialized,
            "startup_complete": self._startup_complete,
            "running_apps": self._running_apps.copy(),
            "has_playwright_tool": self.playwright_tool is not None,
        }

    # === Compatibility methods for setup/evaluate functions ===

    def get_app_url(self, app_name: str) -> str:
        """Get the frontend URL for a running app.

        Args:
            app_name: Name of the app (e.g., '2048', 'todo')

        Returns:
            Frontend URL for the app
        """
        frontend_port = self.get_app_frontend_port(app_name)
        return f"http://localhost:{frontend_port}"

    def get_app_backend_port(self, app_name: str) -> int:
        """Get the backend port for a running app.

        Args:
            app_name: Name of the app (e.g., '2048', 'todo')

        Returns:
            Backend port number
        """
        # Check if app is tracked
        if app_name not in self._running_apps:
            raise ValueError(f"App '{app_name}' not in running apps: {self._running_apps}")

        # Get tracked port
        if app_name not in self._app_ports:
            raise ValueError(f"Port information not available for app '{app_name}'")

        return self._app_ports[app_name]["backend"]

    def get_app_frontend_port(self, app_name: str) -> int:
        """Get the frontend port for a running app.

        Args:
            app_name: Name of the app (e.g., '2048', 'todo')

        Returns:
            Frontend port number
        """
        # Check if app is tracked
        if app_name not in self._running_apps:
            raise ValueError(f"App '{app_name}' not in running apps: {self._running_apps}")

        # Get tracked port
        if app_name not in self._app_ports:
            raise ValueError(f"Port information not available for app '{app_name}'")

        return self._app_ports[app_name]["frontend"]


if __name__ == "__main__":
    # Run the context server with BrowserContext
    context = BrowserContext()
    context.startup()

    # Log initial state
    logger.info(f"[Context] Starting browser context server")
    logger.info(f"[Context] Initial state: {context.get_state_summary()}")

    # Run the context server
    asyncio.run(run_context_server(context, "/tmp/hud_browser_ctx.sock"))
