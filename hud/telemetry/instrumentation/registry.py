from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class InstrumentorRegistry:
    """Registry for telemetry instrumentors."""
    
    def __init__(self) -> None:
        self._instrumentors: dict[str, Any] = {}
        
    def register(self, name: str, instrumentor: Any) -> None:
        """Register an instrumentor.
        
        Args:
            name: Name of the instrumentor
            instrumentor: The instrumentor instance
        """
        self._instrumentors[name] = instrumentor
        logger.debug("Registered instrumentor: %s", name)
        
    def install_all(self) -> None:
        """Install all registered instrumentors."""
        for name, instrumentor in self._instrumentors.items():
            try:
                instrumentor.install()
                logger.debug("Installed instrumentor: %s", name)
            except Exception as e:
                logger.warning("Failed to install instrumentor %s: %s", name, e)


# Create a singleton registry
registry = InstrumentorRegistry()

# Try to register MCP instrumentor if available
try:
    from .mcp import MCPInstrumentor
    registry.register("mcp", MCPInstrumentor())
    logger.debug("MCP instrumentor registered")
except Exception as e:
    logger.debug("Could not register MCP instrumentor: %s", e)
