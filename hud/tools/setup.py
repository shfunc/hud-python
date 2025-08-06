"""Setup tool with built-in registry for MCP environments."""

import json
import logging
from typing import Dict, Type, Any, Optional, TypedDict, Literal
from abc import ABC, abstractmethod
from mcp.types import TextContent
from pydantic import Field

logger = logging.getLogger(__name__)


class SetupResult(TypedDict):
    """Standard setup result format."""
    status: Literal["success", "error"]
    message: str


class BaseSetup(ABC):
    """Base class for environment setup functions."""
    
    @abstractmethod
    async def __call__(self, context: Any, **kwargs) -> SetupResult:
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


class SetupTool:
    """Tool that manages and executes setup functions with built-in registry."""
    
    def __init__(self, context: Any = None, name: str = "setup", description: str | None = None):
        """Initialize the setup tool.
        
        Args:
            context: Environment context to pass to setup functions
            name: Tool name for MCP registration
            description: Tool description
        """
        self.context = context
        self.name = name
        self.description = description or "Setup/configure the environment"
        self._registry: Dict[str, Type[BaseSetup]] = {}
        
    def register(self, name: str, description: str = "", app: str = "default"):
        """Decorator to register a setup class.
        
        Args:
            name: Name of the setup configuration
            description: Description of what this setup does
            app: Application/category name
            
        Example:
            @setup_tool.register("game_2048", "Initialize 2048 game")
            class Game2048Setup(BaseSetup):
                async def __call__(self, context, **kwargs):
                    ...
        """
        def decorator(cls: Type[BaseSetup]) -> Type[BaseSetup]:
            self._registry[name] = cls
            # Store metadata on the class
            setattr(cls, "_name", name)
            setattr(cls, "_description", description)
            setattr(cls, "_app", app)
            logger.info(f"Registered setup: {name} -> {cls.__name__}")
            return cls
        return decorator
    
    async def __call__(
        self,
        function: str = Field(default=None, description="Name of the setup function to execute"),
        args: dict | None = Field(default=None, description="Arguments to pass to the setup function")
    ) -> list[TextContent]:
        """Execute a setup function from the registry.
        
        This method is designed to be called as an MCP tool.
        
        Args:
            function: Name of the setup function to execute
            args: Arguments to pass to the setup function
            
        Returns:
            List of TextContent with the setup result
        """
        args = args or {}
        
        # Default to first registered function if not specified
        if not function:
            if not self._registry:
                return [TextContent(
                    text="❌ No setup functions registered",
                    type="text"
                )]
            function = next(iter(self._registry.keys()))
        
        # Check if function exists
        if function not in self._registry:
            available = ", ".join(self._registry.keys())
            return [TextContent(
                text=f"❌ Unknown setup: {function}\nAvailable: {available}",
                type="text"
            )]
        
        try:
            # Create instance and execute
            setup_class = self._registry[function]
            setup_instance = setup_class()
            result = await setup_instance(self.context, **args)
            
            # Format response
            status = result.get("status", "unknown")
            message = result.get("message", "Setup completed")
            
            if status == "success":
                text = f"✅ {message}"
                # Add any additional info from result
                for key, value in result.items():
                    if key not in ["status", "message"]:
                        text += f"\n{key}: {value}"
            else:
                text = f"❌ Setup failed: {message}"
                if "error" in result:
                    text += f"\nError: {result['error']}"
            
            return [TextContent(text=text, type="text")]
            
        except Exception as e:
            logger.error(f"Setup execution error: {e}", exc_info=True)
            return [TextContent(
                text=f"❌ Setup error: {str(e)}",
                type="text"
            )]
    
    def get_registry_json(self) -> str:
        """Get the registry as JSON for MCP resources."""
        items = {}
        for name, cls in self._registry.items():
            items[name] = {
                "name": getattr(cls, "_name", name),
                "description": getattr(cls, "_description", cls.__doc__ or ""),
                "app": getattr(cls, "_app", "default")
            }
        return json.dumps({"setup_functions": items}, indent=2)
    
    def list_functions(self) -> list[str]:
        """List all registered setup functions."""
        return list(self._registry.keys())


