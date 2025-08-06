"""Selector history evaluator for remote browser environment."""

import logging
from typing import Any
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("selector_history", "Check if selector at index matches expected")
class SelectorHistoryEvaluator(BaseEvaluator):
    """Evaluator that checks if a selector at a specific index in history matches expected."""

    async def __call__(
        self, context: Any, index: int, expected_selector: str, **kwargs
    ) -> EvaluationResult:
        """Check if the selector at the given index matches the expected selector.

        Args:
            context: Browser context with playwright_tool
            index: Index in selector history (0-based)
            expected_selector: Expected selector string
            **kwargs: Additional arguments

        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating selector_history: index={index}, expected={expected_selector}")

        # Context IS the playwright tool
        if not context:
            logger.error("No playwright tool available")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": "No playwright tool available"},
            }

        # Get selector history
        selector_history = context.selector_history if hasattr(context, "selector_history") else []

        # Check if index is valid
        if index < 0 or index >= len(selector_history):
            logger.info(f"No selector found at index {index}")
            return {
                "reward": 0.0,
                "done": False,
                "info": {
                    "success": False,
                    "message": f"No selector found at index {index}",
                    "expected_selector": expected_selector,
                    "selector_history_length": len(selector_history),
                },
            }

        # Get selector at index
        actual_selector = selector_history[index]

        # Check if selector matches
        success = actual_selector == expected_selector

        info = {
            "success": success,
            "actual_selector": actual_selector,
            "expected_selector": expected_selector,
            "index": index,
            "selector_history_length": len(selector_history),
        }

        if success:
            info["message"] = f"Selector at index {index} matches: {expected_selector}"
        else:
            info["message"] = (
                f"Selector at index {index} '{actual_selector}' does not match expected '{expected_selector}'"
            )

        logger.info(f"Selector history evaluation: {info['message']}")

        return {
            "reward": 1.0 if success else 0.0,
            "done": success,
            "info": info,
        }
