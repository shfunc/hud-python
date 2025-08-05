"""History length evaluator for remote browser environment."""

import logging
from ..evaluators import evaluator

logger = logging.getLogger(__name__)


@evaluator("history_length", description="Check if navigation history has expected length")
class HistoryLengthEvaluator:
    """Evaluator that checks navigation history length."""

    def __init__(self, context):
        self.context = context

    async def __call__(self, expected_length: int, min_length: int | None = None) -> dict:
        """
        Check if the navigation history has the expected length.

        Args:
            expected_length: Expected history length
            min_length: If provided, check if history length >= min_length

        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating history_length: expected={expected_length}, min={min_length}")

        # Get the page from context
        page = self.context.page
        if not page:
            logger.error("No page available in context")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "message": "No browser page available",
                },
            }

        # Get navigation history from context
        try:
            history_length = self.context.get_navigation_count()
            current_url = page.url

            if min_length is not None:
                success = history_length >= min_length
                message = f"History length {history_length} {'meets' if success else 'does not meet'} minimum {min_length}"
            else:
                success = history_length == expected_length
                message = f"History length {history_length} {'matches' if success else 'does not match'} expected {expected_length}"

            logger.info(f"History length check: {message}")

            return {
                "reward": 1.0 if success else 0.0,
                "done": True,
                "info": {
                    "success": success,
                    "message": message,
                    "history_length": history_length,
                    "expected_length": expected_length,
                    "min_length": min_length,
                    "current_url": current_url,
                },
            }

        except Exception as e:
            logger.error(f"Failed to check history length: {e}")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "message": f"Failed to check history length: {str(e)}",
                },
            }
