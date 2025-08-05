"""Registry system for setup functions."""

from typing import Dict, Type, Any, List
import json
import logging

logger = logging.getLogger(__name__)

# Global registry for setup classes
SETUP_REGISTRY: Dict[str, Type] = {}


def setup(name: str, app: str | None = None, description: str | None = None):
    """Decorator to register a setup class.

    Args:
        name: The function name used in task configurations
        app: Optional app this setup is specific to
        description: Optional description for the setup

    Example:
        @setup("navigate_to_url", description="Navigate browser to a specific URL")
        class NavigateSetup:
            def __call__(self, context, url: str) -> dict:
                return {"status": "success", "message": f"Navigated to {url}"}
    """

    def decorator(cls):
        # Store metadata on the class
        cls._setup_name = name
        cls._setup_app = app
        cls._setup_description = description

        SETUP_REGISTRY[name] = cls
        logger.info(f"Registered setup: {name} -> {cls.__name__} (app: {app})")
        return cls

    return decorator


class SetupRegistry:
    """Registry for setup functions."""

    @staticmethod
    def create_setup(spec: dict, context=None):
        """Create a setup function from a specification.

        Args:
            spec: Configuration dict with 'function' and 'args' keys
            context: Optional context to pass to setup

        Returns:
            Callable setup instance
        """
        function_name = spec.get("function")
        args = spec.get("args", {})

        if function_name not in SETUP_REGISTRY:
            available = list(SETUP_REGISTRY.keys())
            raise ValueError(f"Unknown setup function: {function_name}. Available: {available}")

        setup_class = SETUP_REGISTRY[function_name]
        instance = setup_class(context)

        # Return a callable that applies the args
        async def _setup():
            # Pass the args to the setup method
            if isinstance(args, dict):
                return await instance(**args)
            else:
                # For backwards compatibility, pass non-dict args directly
                return await instance(args=args)

        return _setup

    @staticmethod
    def to_json() -> str:
        """Convert registry to JSON for MCP resource serving."""
        setups = []
        for name, cls in SETUP_REGISTRY.items():
            setups.append(
                {
                    "function": name,
                    "class": cls.__name__,
                    "app": getattr(cls, "_setup_app", None),
                    "description": getattr(cls, "_setup_description", None),
                }
            )
        return json.dumps(setups, indent=2)

    @staticmethod
    def list_setups() -> List[str]:
        """Get list of available setup function names."""
        return list(SETUP_REGISTRY.keys())

    @staticmethod
    def get_setup_info(name: str) -> dict:
        """Get information about a specific setup."""
        if name not in SETUP_REGISTRY:
            raise ValueError(f"Unknown setup: {name}")

        cls = SETUP_REGISTRY[name]
        return {
            "function": name,
            "class": cls.__name__,
            "app": getattr(cls, "_setup_app", None),
            "description": getattr(cls, "_setup_description", None),
        }
