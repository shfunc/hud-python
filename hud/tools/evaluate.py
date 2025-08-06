"""Evaluate tool with built-in registry for MCP environments."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict

from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class EvaluationResult(TypedDict):
    """Standard evaluation result format."""

    reward: float  # Value between 0.0 and 1.0
    done: bool  # Whether the task/episode is complete
    info: dict[str, Any]  # Additional information


class BaseEvaluator(ABC):
    """Base class for environment evaluators."""

    _name: str
    _description: str
    _app: str

    @abstractmethod
    async def __call__(self, context: Any, **kwargs: Any) -> EvaluationResult:
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


class EvaluateTool(BaseTool):
    """Tool that manages and executes evaluator functions with built-in registry."""

    def __init__(
        self,
        context: Any = None,
        name: str = "evaluate",
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize the evaluate tool.

        Args:
            context: Environment context to pass to evaluator functions
            name: Tool name for MCP registration
            title: Human-readable title for the tool
            description: Tool description
        """
        super().__init__(
            context=context,
            name=name or "evaluate",
            title=title or "State Evaluator",
            description=description or "Evaluate the current environment state",
        )
        self._registry: dict[str, type[BaseEvaluator]] = {}

    def register(
        self, name: str, description: str = "", app: str = "default"
    ) -> Callable[[type[BaseEvaluator]], type[BaseEvaluator]]:
        """Decorator to register an evaluator class.

        Args:
            name: Name of the evaluator
            description: Description of what this evaluator measures
            app: Application/category name

        Example:
            @evaluate_tool.register("max_tile", "Evaluate highest tile")
            class MaxTileEvaluator(BaseEvaluator):
                async def __call__(self, context, **kwargs):
                    ...
        """

        def decorator(cls: type[BaseEvaluator]) -> type[BaseEvaluator]:
            self._registry[name] = cls
            # Store metadata on the class
            cls._name = name
            cls._description = description
            cls._app = app
            logger.info("Registered evaluator: %s -> %s", name, cls.__name__)
            return cls

        return decorator

    async def __call__(self, function: str, args: dict | None = None) -> EvaluationResult:
        """Execute an evaluator function from the registry.

        This method is designed to be called as an MCP tool.

        Args:
            function: Name of the evaluator function to execute
            args: Arguments to pass to the evaluator

        Returns:
            List of TextContent with the evaluation result
        """
        args = args or {}

        # Default to first registered function if not specified
        if not function:
            if not self._registry:
                return EvaluationResult(
                    reward=0.0,
                    done=True,
                    info={"message": "❌ No evaluator functions registered"},
                )
            function = next(iter(self._registry.keys()))

        # Check if function exists
        if function not in self._registry:
            available = ", ".join(self._registry.keys())
            return EvaluationResult(
                reward=0.0,
                done=True,
                info={"message": f"❌ Unknown evaluator: {function}\nAvailable: {available}"},
            )

        try:
            # Create instance and execute
            evaluator_class = self._registry[function]
            evaluator_instance = evaluator_class()
            result = await evaluator_instance(self.context, **args)

            # Format response
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})

            text = f"📊 Evaluation ({function}):\n"
            text += f"Reward: {reward:.3f}\n"
            text += f"Complete: {done}"

            # Add info details if present
            if info:
                if "message" in info:
                    text += f"\n{info['message']}"
                # Add other info fields
                for key, value in info.items():
                    if key not in ["message", "success"]:
                        text += f"\n{key}: {value}"

            return EvaluationResult(
                reward=reward,
                done=done,
                info=info,
            )

        except Exception as e:
            logger.exception("Evaluation execution error: %s", e)
            return EvaluationResult(
                reward=0.0,
                done=True,
                info={"message": f"❌ Evaluation error: {e!s}"},
            )

    def get_registry_json(self) -> str:
        """Get the registry as JSON for MCP resources."""
        items = {}
        for name, cls in self._registry.items():
            items[name] = {
                "name": getattr(cls, "_name", name),
                "description": getattr(cls, "_description", cls.__doc__ or ""),
                "app": getattr(cls, "_app", "default"),
            }
        return json.dumps({"evaluator_functions": items}, indent=2)

    def list_functions(self) -> list[str]:
        """List all registered evaluator functions."""
        return list(self._registry.keys())
