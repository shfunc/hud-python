"""Base context class for MCP environments.

Provides a standard interface for managing environment state
across setup, evaluation, and tool execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class EnvironmentContext(ABC):
    """Base class for environment-specific context.
    
    This class manages the state of an environment (browser, game, etc.)
    and provides it to setup functions, evaluators, and tools.
    """
    
    def __init__(self):
        """Initialize the context."""
        self._initialized = False
    
    @abstractmethod
    async def initialize(self, **kwargs) -> None:
        """Initialize the environment.
        
        Args:
            **kwargs: Environment-specific initialization parameters
        """
        self._initialized = True
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        self._initialized = False
    
    @property
    def initialized(self) -> bool:
        """Check if the context is initialized."""
        return self._initialized
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class SimpleContext(EnvironmentContext):
    """Simple context implementation for basic environments."""
    
    def __init__(self, state: Optional[Any] = None):
        super().__init__()
        self.state = state
    
    async def initialize(self, **kwargs) -> None:
        """Initialize with optional state."""
        await super().initialize(**kwargs)
        if "state" in kwargs:
            self.state = kwargs["state"]
        logger.info("SimpleContext initialized")
    
    async def close(self) -> None:
        """Clean up."""
        await super().close()
        self.state = None
        logger.info("SimpleContext closed")
    
    def set_state(self, state: Any) -> None:
        """Update the state."""
        self.state = state
    
    def get_state(self) -> Any:
        """Get the current state."""
        return self.state