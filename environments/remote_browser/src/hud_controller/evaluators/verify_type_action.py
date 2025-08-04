"""Evaluator to verify a click-then-type action sequence."""

import logging
from typing import Dict, Any, List, Optional
from .registry import evaluator
from .context import RemoteBrowserContext
from . import evaluator_logger

logger = logging.getLogger(__name__)


@evaluator("verify_type_action")
class VerifyTypeActionEvaluator:
    """Evaluator that checks for a sequence: first click on element, then type text into it."""

    name = "verify_type_action"

    async def evaluate(
        self,
        args: Any,
        context: RemoteBrowserContext,
        partial_rewarding: bool = False,
    ) -> float:
        """
        Check for a sequence: first click on element, then type text into it.

        Args:
            args: Can be:
                - List with [selector, value]
                - List with [{"selector": "...", "value": "..."}]
                - Dict with {"selector": "...", "value": "..."}
            context: The remote browser context
            partial_rewarding: Whether to give partial rewards

        Returns:
            1.0 if sequence found, 0.0 otherwise
        """
        evaluator_logger.info("Starting verify_type_action evaluation")

        # Parse args to get selector and expected value
        selector = None
        expected_value = None

        if isinstance(args, list) and len(args) >= 2:
            if isinstance(args[0], dict):
                # Format: [{"selector": "...", "value": "..."}]
                selector = args[0].get("selector")
                expected_value = args[0].get("value")
            else:
                # Format: [selector, value]
                selector = args[0]
                expected_value = args[1]
        elif isinstance(args, dict):
            # Format: {"selector": "...", "value": "..."}
            selector = args.get("selector")
            expected_value = args.get("value")

        if not selector or expected_value is None:
            evaluator_logger.error(f"Invalid args format: {args}")
            return 0.0

        evaluator_logger.info(f"Looking for click-type sequence on selector: {selector}")
        evaluator_logger.info(f"Expected typed value: {expected_value}")

        # Get action history from context
        action_history = []
        if context.playwright_tool:
            action_history = context.playwright_tool.get_action_history()

        if len(action_history) < 2:
            evaluator_logger.info("Not enough actions in history")
            return 0.0

        # Look for the pattern in reverse order (most recent first)
        for i in range(len(action_history) - 1, 0, -1):
            current_action = action_history[i]
            previous_action = action_history[i - 1]

            # Check if current is type and previous is click
            if current_action.get("type") == "type" and previous_action.get("type") == "click":
                # Check if selectors match
                type_selector = current_action.get("details", {}).get("selector")
                click_selector = previous_action.get("details", {}).get("selector")

                if type_selector == selector and click_selector == selector:
                    # Check if typed value matches
                    typed_text = current_action.get("details", {}).get("text", "")

                    if str(typed_text) == str(expected_value):
                        evaluator_logger.info(f"✓ Found matching click-type sequence!")
                        evaluator_logger.info(f"  Click at index {i - 1}: {click_selector}")
                        evaluator_logger.info(
                            f"  Type at index {i}: {type_selector} = '{typed_text}'"
                        )
                        return 1.0
                    else:
                        evaluator_logger.info(f"✗ Found sequence but value mismatch")
                        evaluator_logger.info(f"  Expected: '{expected_value}'")
                        evaluator_logger.info(f"  Got: '{typed_text}'")

        evaluator_logger.info("No matching click-type sequence found")
        return 0.0
