"""Evaluator to verify a click-then-type action sequence."""

import logging
from typing import Any, Optional
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("verify_type_action", "Check for a sequence: first click on element, then type text into it")
class VerifyTypeActionEvaluator(BaseEvaluator):
    """Evaluator that checks for a sequence: first click on element, then type text into it."""

    async def __call__(
        self,
        context: Any,
        expected_text: str,
        selector: Optional[str] = None,
        partial_rewarding: bool = False,
        **kwargs
    ) -> EvaluationResult:
        """Check for a sequence: first click on element, then type text into it.
        
        Args:
            context: Browser context with playwright_tool
            expected_text: The expected text that should have been typed
            selector: Optional selector to check (if not provided, checks last type action)
            partial_rewarding: Whether to give partial rewards
            **kwargs: Additional arguments
            
        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info("Starting verify_type_action evaluation")
        
        expected_value = expected_text
        
        if not expected_value:
            logger.error("No expected text provided")
            return {
                "reward": 0.0,
                "done": False,
                "info": {
                    "success": False,
                    "message": "No expected text provided",
                },
            }
        
        logger.info(f"Looking for type action with text: {expected_value}")
        if selector:
            logger.info(f"Checking for specific selector: {selector}")
        
        # Get action history from context
        if not context or not hasattr(context, 'action_history') or not context.action_history:
            logger.error("No playwright tool available")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": "No playwright tool available"},
            }
        
        action_history = context.action_history
        
        logger.info(f"Total actions in history: {len(action_history)}")
        
        if len(action_history) == 0:
            logger.info("No actions in history")
            return {
                "reward": 0.0,
                "done": False,
                "info": {
                    "success": False,
                    "message": "No actions in history",
                    "action_count": 0,
                },
            }
        
        # Look for the most recent type action
        for i in range(len(action_history) - 1, -1, -1):
            action = action_history[i]
            
            if action.get("type") == "type":
                action_details = action.get("details", {})
                typed_text = action_details.get("text", "")
                action_selector = action_details.get("selector", "")
                
                # Check if selector matches (if specified)
                if selector and action_selector != selector:
                    continue
                
                # Check if typed text matches
                if str(typed_text) == str(expected_value):
                    logger.info(f"✓ Found matching type action at index {i}")
                    logger.info(f"  Selector: {action_selector}")
                    logger.info(f"  Text: '{typed_text}'")
                    
                    return {
                        "reward": 1.0,
                        "done": True,
                        "info": {
                            "success": True,
                            "message": "Found matching type action",
                            "typed_text": typed_text,
                            "selector": action_selector,
                            "action_index": i,
                        },
                    }
                elif not selector:
                    # If no specific selector required, any mismatch is a failure
                    logger.info(f"✗ Found type action but text mismatch")
                    logger.info(f"  Expected: '{expected_value}'")
                    logger.info(f"  Got: '{typed_text}'")
                    
                    if partial_rewarding:
                        return {
                            "reward": 0.5,
                            "done": False,
                            "info": {
                                "success": False,
                                "message": "Text mismatch",
                                "expected": expected_value,
                                "actual": typed_text,
                                "selector": action_selector,
                            },
                        }
        
        logger.info("No matching type action found")
        return {
            "reward": 0.0,
            "done": False,
            "info": {
                "success": False,
                "message": "No matching type action found",
                "expected_text": expected_value,
                "required_selector": selector,
            },
        }