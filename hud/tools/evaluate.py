"""Evaluate tool with built-in registry for MCP environments."""

import json
import logging
from typing import Dict, Type, Any, Optional, TypedDict
from abc import ABC, abstractmethod
from mcp.types import ContentBlock, TextContent

from .base import BaseTool

logger = logging.getLogger(__name__)


class EvaluationResult(TypedDict):
    """Standard evaluation result format."""
    reward: float  # Value between 0.0 and 1.0
    done: bool  # Whether the task/episode is complete
    info: dict[str, Any]  # Additional information


class BaseEvaluator(ABC):
    """Base class for environment evaluators."""
    
    @abstractmethod
    async def __call__(self, context: Any, **kwargs) -> EvaluationResult:
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


class EvaluateTool(BaseTool):
    """Tool that manages and executes evaluator functions with built-in registry."""
    
    def __init__(self, context: Any = None, name: str = "evaluate", description: str = None):
        """Initialize the evaluate tool.
        
        Args:
            context: Environment context to pass to evaluator functions
            name: Tool name for MCP registration
            description: Tool description
        """
        super().__init__(
            context=context,
            name=name or "evaluate",
            description=description or "Evaluate the current environment state"
        )
        self._registry: Dict[str, Type[BaseEvaluator]] = {}
        
    def register(self, name: str, description: str = "", app: str = "default"):
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
        def decorator(cls: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
            self._registry[name] = cls
            # Store metadata on the class
            setattr(cls, "_name", name)
            setattr(cls, "_description", description)
            setattr(cls, "_app", app)
            logger.info(f"Registered evaluator: {name} -> {cls.__name__}")
            return cls
        return decorator
    
    async def __call__(
        self,
        function: str = None,
        args: Optional[dict] = None
    ) -> list[ContentBlock]:
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
                return [TextContent(
                    text="❌ No evaluator functions registered",
                    type="text"
                )]
            function = next(iter(self._registry.keys()))
        
        # Check if function exists
        if function not in self._registry:
            available = ", ".join(self._registry.keys())
            return [TextContent(
                text=f"❌ Unknown evaluator: {function}\nAvailable: {available}",
                type="text"
            )]
        
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
            
            return [TextContent(text=text, type="text")]
            
        except Exception as e:
            logger.error(f"Evaluation execution error: {e}", exc_info=True)
            return [TextContent(
                text=f"❌ Evaluation error: {str(e)}",
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
        return json.dumps({"evaluator_functions": items}, indent=2)
    
    def list_functions(self) -> list[str]:
        """List all registered evaluator functions."""
        return list(self._registry.keys())


