"""Registry pattern for MCP environments.

This module provides base classes and utilities for creating
registries of setup functions and evaluators in MCP environments.
"""

from typing import Dict, Type, Any, Callable, Optional
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseSetup(ABC):
    """Base class for environment setup functions."""
    
    @abstractmethod
    async def __call__(self, context: Any, **kwargs) -> dict:
        """Execute the setup.
        
        Args:
            context: Environment-specific context object
            **kwargs: Setup-specific arguments
            
        Returns:
            Dict with at least:
                - status: "success" or "error"
                - message: Human-readable message
        """
        pass


class BaseEvaluator(ABC):
    """Base class for environment evaluators."""
    
    @abstractmethod
    async def __call__(self, context: Any, **kwargs) -> dict:
        """Execute the evaluation.
        
        Args:
            context: Environment-specific context object
            **kwargs: Evaluator-specific arguments
            
        Returns:
            Dict with:
                - reward: float between 0 and 1
                - done: bool indicating if task is complete
                - info: dict with additional information
        """
        pass


class Registry:
    """Generic registry for MCP environment components."""
    
    def __init__(self, name: str = "registry"):
        self.name = name
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: str, description: str = "", app: str = ""):
        """Decorator to register a class.
        
        Args:
            name: The function name used in configurations
            description: Optional description
            app: Optional app this is specific to
        """
        def decorator(cls: Type) -> Type:
            # Store metadata on the class
            setattr(cls, f"_{self.name}_name", name)
            setattr(cls, f"_{self.name}_description", description)
            setattr(cls, f"_{self.name}_app", app)
            
            self._registry[name] = cls
            logger.info(f"Registered {self.name}: {name} -> {cls.__name__}")
            return cls
        
        return decorator
    
    def get(self, name: str) -> Optional[Type]:
        """Get a registered class by name.
        
        Args:
            name: The registered name
            
        Returns:
            The registered class or None
        """
        return self._registry.get(name)
    
    def create_instance(self, name: str) -> Any:
        """Create an instance of a registered class.
        
        Args:
            name: The registered name
            
        Returns:
            An instance of the registered class
        """
        if name not in self._registry:
            raise ValueError(f"Unknown {self.name}: {name}")
        
        cls = self._registry[name]
        return cls()
    
    def to_json(self) -> str:
        """Export registry as JSON for MCP resources."""
        items = {}
        for name, cls in self._registry.items():
            items[name] = {
                "description": getattr(cls, f"_{self.name}_description", ""),
                "app": getattr(cls, f"_{self.name}_app", "")
            }
        
        return json.dumps({self.name: items}, indent=2)
    
    def list_items(self) -> list:
        """List all registered items."""
        return list(self._registry.keys())


# Pre-configured registries for common use cases
SetupRegistry = Registry("setup")
EvaluatorRegistry = Registry("evaluator")


# Decorator shortcuts
def setup(name: str, description: str = "", app: str = ""):
    """Decorator to register a setup class."""
    return SetupRegistry.register(name, description, app)


def evaluator(name: str, description: str = "", app: str = ""):
    """Decorator to register an evaluator class."""
    return EvaluatorRegistry.register(name, description, app)