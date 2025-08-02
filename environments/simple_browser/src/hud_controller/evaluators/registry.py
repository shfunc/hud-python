"""Registry system for evaluators with MCP resource support."""

from typing import Dict, Type, Any, List
import json
import logging

logger = logging.getLogger(__name__)

# Global registry for evaluator classes
EVALUATOR_REGISTRY: Dict[str, Type] = {}


def evaluator(name: str, app: str = None, description: str = None):
    """Decorator to register an evaluator class.

    Args:
        name: The function name used in task configurations
        app: Optional app this evaluator is specific to (e.g., 'todo', 'google')
        description: Optional description for the evaluator

    Example:
        @evaluator("todo_completed", app="todo", description="Check completed todo count")
        class TodoCompletedEvaluator:
            def __call__(self, context, expected_count: int) -> dict:
                return {"reward": 1.0, "done": True}
    """

    def decorator(cls):
        # Store metadata on the class
        cls._evaluator_name = name
        cls._evaluator_app = app
        cls._evaluator_description = description

        EVALUATOR_REGISTRY[name] = cls
        logger.info(f"Registered evaluator: {name} -> {cls.__name__} (app: {app})")
        return cls

    return decorator


class EvaluatorRegistry:
    """Registry that can serve evaluator information as MCP resources."""

    @staticmethod
    def create_evaluator(spec: dict, context=None):
        """Create an evaluator from a function/args specification.

        Args:
            spec: Configuration dict with 'function' and 'args' keys
            context: Optional context to pass to evaluator

        Returns:
            Callable evaluator instance
        """
        function_name = spec.get("function")
        args = spec.get("args", {})

        if function_name not in EVALUATOR_REGISTRY:
            available = list(EVALUATOR_REGISTRY.keys())
            raise ValueError(
                f"Unknown evaluator function: '{function_name}'. Available evaluators: {available}"
            )

        evaluator_cls = EVALUATOR_REGISTRY[function_name]
        evaluator_instance = evaluator_cls()

        # Return a callable that includes the context and args
        async def evaluate():
            return await evaluator_instance(context, **args)

        # Attach metadata for debugging
        evaluate.function_name = function_name
        evaluate.args = args
        evaluate.evaluator_class = evaluator_cls

        return evaluate

    @staticmethod
    def get_all_evaluators() -> List[Dict[str, Any]]:
        """Get all evaluators with metadata for MCP resource."""
        evaluators = []

        for name, cls in EVALUATOR_REGISTRY.items():
            evaluator_info = {
                "name": name,
                "class_name": cls.__name__,
                "app": getattr(cls, "_evaluator_app", None),
                "description": getattr(cls, "_evaluator_description", None),
                "docstring": cls.__doc__ or "No documentation available",
            }

            # Try to get __call__ method info
            if hasattr(cls, "__call__"):
                call_method = getattr(cls, "__call__")
                evaluator_info["call_docstring"] = (
                    call_method.__doc__ or "No documentation available"
                )
                evaluator_info["call_annotations"] = str(
                    getattr(call_method, "__annotations__", {})
                )

            evaluators.append(evaluator_info)

        return evaluators

    @staticmethod
    def get_evaluators_by_app(app: str) -> List[Dict[str, Any]]:
        """Get evaluators filtered by app."""
        all_evaluators = EvaluatorRegistry.get_all_evaluators()
        return [e for e in all_evaluators if e.get("app") == app]

    @staticmethod
    def get_evaluator_schema(function_name: str) -> Dict[str, Any]:
        """Get detailed schema for a specific evaluator."""
        if function_name not in EVALUATOR_REGISTRY:
            return {"error": f"Evaluator '{function_name}' not found"}

        cls = EVALUATOR_REGISTRY[function_name]

        schema = {
            "function": function_name,
            "class_name": cls.__name__,
            "app": getattr(cls, "_evaluator_app", None),
            "description": getattr(cls, "_evaluator_description", None),
            "docstring": cls.__doc__ or "No documentation available",
            "usage_example": {
                "function": function_name,
                "args": {},  # Could be enhanced to parse method signature
            },
        }

        # Try to extract call method signature
        if hasattr(cls, "__call__"):
            call_method = getattr(cls, "__call__")
            schema["call_signature"] = str(getattr(call_method, "__annotations__", {}))
            schema["call_docstring"] = call_method.__doc__ or "No documentation available"

        return schema

    @staticmethod
    def to_json() -> str:
        """Export all evaluators as JSON for MCP resource."""
        data = {
            "evaluators": EvaluatorRegistry.get_all_evaluators(),
            "apps": list(
                set(e.get("app") for e in EvaluatorRegistry.get_all_evaluators() if e.get("app"))
            ),
            "count": len(EVALUATOR_REGISTRY),
        }
        return json.dumps(data, indent=2)
