"""Raw last action evaluator for remote browser environment."""

import logging
from ..evaluators import evaluator

logger = logging.getLogger(__name__)


@evaluator("raw_last_action_is", description="Check if the last action matches expected")
class RawLastActionIsEvaluator:
    """Evaluator that checks if the last action matches expected type and details."""

    def __init__(self, context):
        self.context = context

    async def __call__(self, expected_action: str, expected_details: dict | None = None) -> dict:
        """
        Check if the last action matches the expected action.

        Args:
            expected_action: Expected action type (e.g., "click", "type", "navigate")
            expected_details: Optional expected details of the action

        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating raw_last_action_is: expected={expected_action}")

        # Get last action from context
        last_action = self.context.get_last_action()

        if not last_action:
            logger.info("No actions have been performed yet")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "message": "No actions have been performed",
                    "expected_action": expected_action,
                },
            }

        # Check if action type matches
        action_matches = last_action["type"] == expected_action
        details_match = True

        # Check details if provided
        if expected_details and action_matches:
            actual_details = last_action.get("details", {})
            for key, expected_value in expected_details.items():
                if actual_details.get(key) != expected_value:
                    details_match = False
                    break

        success = action_matches and details_match

        info = {
            "success": success,
            "last_action": last_action,
            "expected_action": expected_action,
            "expected_details": expected_details,
        }

        if success:
            info["message"] = f"Last action matches: {expected_action}"
        else:
            if not action_matches:
                info["message"] = (
                    f"Last action '{last_action['type']}' does not match expected '{expected_action}'"
                )
            else:
                info["message"] = f"Action matches but details do not match"

        logger.info(f"Last action evaluation: {info['message']}")

        return {
            "reward": 1.0 if success else 0.0,
            "done": True,
            "info": info,
        }
