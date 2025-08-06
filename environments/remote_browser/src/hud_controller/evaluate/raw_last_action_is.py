"""Raw last action evaluator for remote browser environment."""

import logging
from typing import Any, Optional, Dict
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("raw_last_action_is", "Check if the last action matches expected")
class RawLastActionIsEvaluator(BaseEvaluator):
    """Evaluator that checks if the last action matches expected type and details."""

    async def __call__(
        self, 
        context: Any, 
        expected_action: str, 
        expected_details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EvaluationResult:
        """Check if the last action matches the expected action.
        
        Args:
            context: Browser context with playwright_tool
            expected_action: Expected action type (e.g., "click", "type", "navigate")
            expected_details: Optional expected details of the action
            **kwargs: Additional arguments
            
        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating raw_last_action_is: expected={expected_action}")
        
        # Context IS the playwright tool
        if not context:
            logger.error("No playwright tool available")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": "No playwright tool available"},
            }
        
        # Get action history
        action_history = context.action_history if hasattr(context, 'action_history') else []
        
        if not action_history:
            logger.info("No actions have been performed yet")
            return {
                "reward": 0.0,
                "done": False,
                "info": {
                    "success": False,
                    "message": "No actions have been performed",
                    "expected_action": expected_action,
                },
            }
        
        # Get last action
        last_action = action_history[-1]
        
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
            "done": success,
            "info": info,
        }