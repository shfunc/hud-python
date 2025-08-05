"""Registry system for evaluators with MCP resource support."""

from typing import Dict, Type, Any, List
import json
import logging
import io
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Setup evaluator logger for rich logging
evaluator_logger = logging.getLogger("hud_controller.evaluator_logger")
evaluator_logger.propagate = True
evaluator_logger.setLevel(logging.INFO)

# Global registry for evaluator classes
EVALUATOR_REGISTRY: Dict[str, Type] = {}


@contextmanager
def capture_user_logs():
    """
    Context manager to capture log messages during evaluation.

    IMPORTANT: Does NOT redirect stdout to avoid interfering with MCP protocol.
    Only captures log messages that should be included in evaluation results.
    """
    log_capture = io.StringIO()
    capture_handler = logging.StreamHandler(log_capture)
    capture_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # Only capture from the evaluator logger, not root logger
    evaluator_logger.addHandler(capture_handler)

    try:
        yield log_capture
    finally:
        evaluator_logger.removeHandler(capture_handler)


def evaluator(name: str, app: str | None = None, description: str | None = None):
    """Decorator to register an evaluator class.

    Args:
        name: The function name used in task configurations
        app: Optional app this evaluator is specific to
        description: Optional description for the evaluator

    Example:
        @evaluator("url_match", description="Check if URL matches expected pattern")
        class URLMatchEvaluator:
            def __call__(self, context, expected_url: str) -> dict:
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
            raise ValueError(f"Unknown evaluator function: {function_name}. Available: {available}")

        evaluator_class = EVALUATOR_REGISTRY[function_name]
        instance = evaluator_class(context)

        # Return a callable that applies the args and captures logs
        async def _evaluator():
            # Capture logs during evaluation
            with capture_user_logs() as log_capture:
                try:
                    evaluator_logger.info(f"Starting evaluation with evaluator: {function_name}")
                    evaluator_logger.info(f"Evaluator args: {args}")

                    # Call the evaluator
                    result = await instance(**args)

                    evaluator_logger.info(f"Evaluation completed. Result: {result}")
                    captured_logs = log_capture.getvalue()

                    # Add logs to the result if it's a dict
                    if isinstance(result, dict):
                        result["logs"] = captured_logs

                    return result

                except Exception as e:
                    error_msg = f"Error in evaluator {function_name}: {str(e)}"
                    evaluator_logger.error(error_msg)
                    captured_logs = log_capture.getvalue()
                    return {
                        "reward": 0.0,
                        "done": True,
                        "info": {"error": str(e), "logs": captured_logs},
                    }

        return _evaluator

    @staticmethod
    def to_json() -> str:
        """Convert registry to JSON for MCP resource serving."""
        evaluators = []
        for name, cls in EVALUATOR_REGISTRY.items():
            evaluators.append(
                {
                    "function": name,
                    "class": cls.__name__,
                    "app": getattr(cls, "_evaluator_app", None),
                    "description": getattr(cls, "_evaluator_description", None),
                }
            )
        return json.dumps(evaluators, indent=2)

    @staticmethod
    def list_evaluators() -> List[str]:
        """Get list of available evaluator function names."""
        return list(EVALUATOR_REGISTRY.keys())

    @staticmethod
    def get_evaluator_info(name: str) -> dict:
        """Get information about a specific evaluator."""
        if name not in EVALUATOR_REGISTRY:
            raise ValueError(f"Unknown evaluator: {name}")

        cls = EVALUATOR_REGISTRY[name]
        return {
            "function": name,
            "class": cls.__name__,
            "app": getattr(cls, "_evaluator_app", None),
            "description": getattr(cls, "_evaluator_description", None),
        }
