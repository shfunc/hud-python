from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class InstrumentorRegistry:
    """Registry for telemetry instrumentors."""
    
    def __init__(self):
        self._instrumentors: Dict[str, Any] = {}
        
    def register(self, name: str, instrumentor: Any) -> None:
        """Register an instrumentor.
        
        Args:
            name: Name of the instrumentor
            instrumentor: The instrumentor instance
        """
        self._instrumentors[name] = instrumentor
        logger.debug(f"Registered instrumentor: {name}")
        
    def install_all(self) -> None:
        """Install all registered instrumentors."""
        for name, instrumentor in self._instrumentors.items():
            try:
                instrumentor.install()
                logger.debug(f"Installed instrumentor: {name}")
            except Exception as e:
                logger.warning(f"Failed to install instrumentor {name}: {e}")


# Create a singleton registry
registry = InstrumentorRegistry()

# Try to register MCP instrumentor if available
try:
    from .mcp import MCPInstrumentor
    registry.register("mcp", MCPInstrumentor())
    logger.debug("MCP instrumentor registered")
except Exception as e:
    logger.debug(f"Could not register MCP instrumentor: {e}") 