"""Setup tools registry for browser environment."""

import json
import logging
from typing import Dict, Any, Type, Callable, List
from ..evaluators.context import BrowserEnvironmentContext

logger = logging.getLogger(__name__)

# Global setup registry
SETUP_REGISTRY: Dict[str, Type] = {}


def setup(name: str, app: str = "", description: str = ""):
    """Decorator to register setup tools (mirror of @evaluator pattern).

    Args:
        name: Setup tool name
        app: Application this setup is for (optional)
        description: Human-readable description

    Usage:
        @setup("todo_seed", app="todo", description="Seed test data")
        class TodoSeedSetup:
            async def __call__(self, context, **kwargs):
                # Setup implementation
                pass
    """

    def decorator(cls: Type) -> Type:
        # Store metadata on the class
        cls._setup_name = name
        cls._setup_app = app
        cls._setup_description = description

        # Register in global registry
        SETUP_REGISTRY[name] = cls
        logger.debug(f"Registered setup tool: {name} -> {cls.__name__}")

        return cls

    return decorator


class SetupRegistry:
    """Registry for setup tools with factory and introspection methods."""

    @classmethod
    def create_setup(
        cls, setup_spec: Dict[str, Any], context: BrowserEnvironmentContext
    ) -> Callable:
        """Create a setup function from specification.

        Args:
            setup_spec: {"function": "setup_name", "args": {...}}
            context: Environment context

        Returns:
            Callable that executes the setup when called
        """
        function_name = setup_spec.get("function")
        args = setup_spec.get("args", {})

        if function_name not in SETUP_REGISTRY:
            raise ValueError(f"Unknown setup tool: {function_name}")

        setup_class = SETUP_REGISTRY[function_name]
        setup_instance = setup_class()

        async def setup_executor():
            return await setup_instance(context, **args)

        return setup_executor

    @classmethod
    def get_all_setup_tools(cls) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered setup tools."""
        result = {}
        for name, setup_class in SETUP_REGISTRY.items():
            result[name] = {
                "name": name,
                "class": setup_class.__name__,
                "app": getattr(setup_class, "_setup_app", ""),
                "description": getattr(setup_class, "_setup_description", ""),
                "module": setup_class.__module__,
            }
        return result

    @classmethod
    def get_setup_tools_by_app(cls, app: str) -> Dict[str, Dict[str, Any]]:
        """Get setup tools for a specific app."""
        all_tools = cls.get_all_setup_tools()
        return {name: info for name, info in all_tools.items() if info.get("app") == app}

    @classmethod
    def get_setup_schema(cls, setup_name: str) -> Dict[str, Any]:
        """Get detailed schema for a specific setup tool."""
        if setup_name not in SETUP_REGISTRY:
            return {"error": f"Setup tool '{setup_name}' not found"}

        setup_class = SETUP_REGISTRY[setup_name]
        return {
            "name": setup_name,
            "class": setup_class.__name__,
            "app": getattr(setup_class, "_setup_app", ""),
            "description": getattr(setup_class, "_setup_description", ""),
            "module": setup_class.__module__,
            "docstring": setup_class.__doc__,
            "call_method_doc": getattr(setup_class.__call__, "__doc__", None)
            if hasattr(setup_class, "__call__")
            else None,
        }

    @classmethod
    def to_json(cls) -> str:
        """Export all setup tools as JSON."""
        return json.dumps(
            {
                "setup_tools": cls.get_all_setup_tools(),
                "total_count": len(SETUP_REGISTRY),
                "registry_type": "setup_tools",
            },
            indent=2,
        )
