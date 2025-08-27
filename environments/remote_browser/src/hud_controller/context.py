"""
Context server for remote browser environment that persists state across hot-reloads.

Run this as a separate process to maintain browser session state during development:
    python -m hud_controller.context
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from hud.server.context import run_context_server

logger = logging.getLogger(__name__)


class RemoteBrowserContext:
    """Context that holds remote browser state across reloads."""
    
    def __init__(self):
        """Initialize the remote browser context."""
        self.browser_provider = None
        self.cdp_url: Optional[str] = None
        self.is_initialized = False
        self.provider_config: Optional[Dict[str, Any]] = None
        self.launch_options: Optional[Dict[str, Any]] = None
        self.provider_name: Optional[str] = None
        self.instance_id: Optional[str] = None
        self._startup_complete = False
        self.playwright_tool = None  # Store the playwright tool
        
        logger.info("[RemoteBrowserContext] Created new remote browser context")
    
    def startup(self):
        """One-time startup when context server starts."""
        if self._startup_complete:
            logger.info("[RemoteBrowserContext] Startup already complete, skipping")
            return
            
        logger.info("[RemoteBrowserContext] Performing one-time startup")
        self._startup_complete = True
    
    # === Proxy-friendly methods for multiprocessing.Manager ===
    # Note: These are needed because direct attribute access doesn't always
    # work correctly through the multiprocessing proxy
    
    def get_browser_provider(self):
        """Get the browser provider instance."""
        return self.browser_provider
    
    def set_browser_provider(self, provider) -> None:
        """Set the browser provider instance."""
        self.browser_provider = provider
        if provider:
            self.provider_name = provider.__class__.__name__.replace("Provider", "").lower()
            logger.info(f"[RemoteBrowserContext] Set browser provider: {self.provider_name}")
    
    def get_cdp_url(self) -> Optional[str]:
        """Get the CDP URL."""
        return self.cdp_url
    
    def set_cdp_url(self, url: str) -> None:
        """Set the CDP URL."""
        self.cdp_url = url
        logger.info(f"[RemoteBrowserContext] Set CDP URL: {url}")
    
    def get_is_initialized(self) -> bool:
        """Check if environment is initialized."""
        return self.is_initialized
    
    def set_initialized(self, value: bool) -> None:
        """Set initialization status."""
        self.is_initialized = value
        logger.info(f"[RemoteBrowserContext] Initialization status: {value}")
    
    def get_provider_config(self) -> Optional[Dict[str, Any]]:
        """Get provider configuration."""
        return self.provider_config
    
    def set_provider_config(self, config: Dict[str, Any]) -> None:
        """Set provider configuration."""
        self.provider_config = config
        logger.info(f"[RemoteBrowserContext] Set provider config")
    
    def get_launch_options(self) -> Optional[Dict[str, Any]]:
        """Get launch options."""
        return self.launch_options
    
    def set_launch_options(self, options: Dict[str, Any]) -> None:
        """Set launch options."""
        self.launch_options = options
        logger.info(f"[RemoteBrowserContext] Set launch options")
    
    def get_playwright_tool(self):
        """Get the playwright tool instance."""
        return self.playwright_tool
    
    def set_playwright_tool(self, tool) -> None:
        """Set the playwright tool instance."""
        self.playwright_tool = tool
        logger.info(f"[RemoteBrowserContext] Set playwright tool")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "is_initialized": self.is_initialized,
            "startup_complete": self._startup_complete,
            "provider_name": self.provider_name,
            "has_cdp_url": self.cdp_url is not None,
            "has_browser_provider": self.browser_provider is not None,
            "has_playwright_tool": self.playwright_tool is not None,
        }
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data from the browser provider."""
        # Return basic telemetry data without async calls
        # The browser provider status check is skipped to avoid async issues
        
        # Get live view URL if available
        live_url = None
        if self.browser_provider and hasattr(self.browser_provider, 'get_live_view_url'):
            try:
                live_url = self.browser_provider.get_live_view_url()
            except Exception as e:
                logger.warning(f"Failed to get live view URL: {e}")
        
        return {
            "provider": self.provider_name or "unknown",
            "status": "running" if self.browser_provider and self.is_initialized else "not_initialized",
            "live_url": live_url,
            "cdp_url": self.cdp_url,
            "instance_id": self.instance_id,
        }


if __name__ == "__main__":
    # Run the context server with RemoteBrowserContext
    context = RemoteBrowserContext()
    context.startup()
    
    # Log initial state
    logger.info(f"[Context] Starting remote browser context server")
    logger.info(f"[Context] Initial state: {context.get_state_summary()}")
    
    # Run the context server
    asyncio.run(run_context_server(context, "/tmp/hud_remote_browser_ctx.sock"))
