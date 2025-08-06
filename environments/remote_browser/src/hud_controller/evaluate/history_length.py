"""History length evaluator for remote browser environment."""

import logging
from typing import Any, Optional
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("history_length", "Check if action history has specific length")
class HistoryLengthEvaluator(BaseEvaluator):
    """Evaluator to check action history length."""
    
    async def __call__(
        self, 
        context: Any,
        min_length: Optional[int] = None, 
        max_length: Optional[int] = None,
        **kwargs
    ) -> EvaluationResult:
        """Check if action history length is within bounds.
        
        Args:
            context: Browser context with playwright_tool
            min_length: Minimum required length
            max_length: Maximum allowed length
            **kwargs: Additional arguments
            
        Returns:
            Evaluation result
        """
        logger.info(f"Evaluating history length - min: {min_length}, max: {max_length}")
        
        # Context IS the playwright tool
        if not context:
            logger.error("No playwright tool available")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": "No playwright tool available"},
            }
        
        # Get action history from PlaywrightToolWithMemory
        history_length = len(context.action_history) if hasattr(context, 'action_history') else 0
        logger.info(f"Current history length: {history_length}")
        
        in_range = True
        if min_length is not None and history_length < min_length:
            in_range = False
            logger.info(f"❌ History too short: {history_length} < {min_length}")
        if max_length is not None and history_length > max_length:
            in_range = False
            logger.info(f"❌ History too long: {history_length} > {max_length}")
        
        if in_range:
            logger.info(f"✅ History length in range: {history_length}")
        
        # Calculate reward based on how close we are to the target
        if min_length is not None and max_length is not None:
            target = (min_length + max_length) / 2
            reward = max(0, 1 - abs(history_length - target) / target)
        else:
            reward = 1.0 if in_range else 0.0
        
        return {
            "reward": float(reward),
            "done": in_range,
            "info": {
                "history_length": history_length,
                "min_length": min_length,
                "max_length": max_length,
                "in_range": in_range,
            },
        }