"""Setup tool with built-in registry for MCP environments."""
# flake8: noqa: B008

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from mcp.types import ContentBlock, TextContent
from pydantic import Field

from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class SetupResult(TypedDict, total=False):
    """Standard setup result format.

    Required fields: status, message
    Additional fields are allowed.
    """

    status: Literal["success", "error"]
    message: str


class BaseSetup(ABC):
    """Base class for environment setup functions."""

    _name: str
    _description: str
    _app: str

    @abstractmethod
    async def __call__(self, context: Any, **kwargs: Any) -> SetupResult:
        """Execute the setup.

        Args:
            context: Environment-specific context object
            **kwargs: Setup-specific arguments

        Returns:
            Dict with at least:
                - status: "success" or "error"
                - message: Human-readable message
        """


class SetupTool(BaseTool):
    """Tool that manages and executes setup functions with built-in registry."""

    def __init__(
        self,
        context: Any = None,
        name: str = "setup",
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize the setup tool.

        Args:
            context: Environment context to pass to setup functions
            name: Tool name for MCP registration
            title: Human-readable title for the tool
            description: Tool description
        """
        super().__init__(
            context=context,
            name=name or "setup",
            title=title or "Environment Setup",
            description=description or "Setup/configure the environment",
        )
        self._registry: dict[str, type[BaseSetup]] = {}

    def register(
        self, name: str, description: str = "", app: str = "default"
    ) -> Callable[[type[BaseSetup]], type[BaseSetup]]:
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

        def decorator(cls: type[BaseSetup]) -> type[BaseSetup]:
            self._registry[name] = cls
            # Store metadata on the class
            cls._name = name
            cls._description = description
            cls._app = app
            logger.info("Registered setup: %s -> %s", name, cls.__name__)
            return cls

        return decorator

    async def __call__(
        self,
        function: str = Field(default=None, description="Name of the setup function to execute"),
        args: dict | None = Field(
            default=None, description="Arguments to pass to the setup function"
        ),
    ) -> list[ContentBlock]:
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
                return [TextContent(text="❌ No setup functions registered", type="text")]
            function = next(iter(self._registry.keys()))

        # Check if function exists
        if function not in self._registry:
            available = ", ".join(self._registry.keys())
            return [
                TextContent(
                    text=f"❌ Unknown setup: {function}\nAvailable: {available}", type="text"
                )
            ]

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
            logger.exception("Setup execution error: %s", e)
            return [TextContent(text=f"❌ Setup error: {e!s}", type="text")]

    def get_registry_json(self) -> str:
        """Get the registry as JSON for MCP resources."""
        items = {}
        for name, cls in self._registry.items():
            items[name] = {
                "name": getattr(cls, "_name", name),
                "description": getattr(cls, "_description", cls.__doc__ or ""),
                "app": getattr(cls, "_app", "default"),
            }
        return json.dumps({"setup_functions": items}, indent=2)

    def list_functions(self) -> list[str]:
        """List all registered setup functions."""
        return list(self._registry.keys())
