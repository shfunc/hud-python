"""Selector history evaluator for remote browser environment."""

import logging
from ..evaluators import evaluator

logger = logging.getLogger(__name__)


@evaluator("selector_history", description="Check if selector at index matches expected")
class SelectorHistoryEvaluator:
    """Evaluator that checks if a selector at a specific index in history matches expected."""

    def __init__(self, context):
        self.context = context

    async def __call__(self, index: int, expected_selector: str) -> dict:
        """
        Check if the selector at the given index matches the expected selector.

        Args:
            index: Index in selector history (0-based)
            expected_selector: Expected selector string

        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating selector_history: index={index}, expected={expected_selector}")

        # Get selector at index from context
        actual_selector = self.context.get_selector_at_index(index)

        if actual_selector is None:
            logger.info(f"No selector found at index {index}")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "message": f"No selector found at index {index}",
                    "expected_selector": expected_selector,
                    "selector_history_length": len(self.context.selector_history),
                },
            }

        # Check if selector matches
        success = actual_selector == expected_selector

        info = {
            "success": success,
            "actual_selector": actual_selector,
            "expected_selector": expected_selector,
            "index": index,
            "selector_history_length": len(self.context.selector_history),
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
            "done": True,
            "info": info,
        }
