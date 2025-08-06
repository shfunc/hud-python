"""Element exists evaluator for remote browser environment."""

import logging
from typing import Any
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("element_exists", "Check if an element exists on the page")
class ElementExistsEvaluator(BaseEvaluator):
    """Evaluator to check if an element exists."""
    
    async def __call__(self, context: Any, selector: str, **kwargs) -> EvaluationResult:
        """Check if an element exists on the page.
        
        Args:
            context: Browser context with playwright_tool
            selector: CSS selector for the element
            **kwargs: Additional arguments
            
        Returns:
            Evaluation result
        """
        logger.info(f"Checking if element exists: {selector}")
        
        # Context IS the playwright tool
        if not context or not hasattr(context, 'page') or not context.page:
            logger.error("No browser page available")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": "No browser page available"},
            }
        
        try:
            element = await context.page.query_selector(selector)
            exists = element is not None
            
            if exists:
                logger.info(f"✅ Element found: {selector}")
            else:
                logger.info(f"❌ Element not found: {selector}")
            
            return {
                "reward": 1.0 if exists else 0.0,
                "done": exists,
                "info": {
                    "selector": selector,
                    "exists": exists,
                },
            }
        except Exception as e:
            logger.error(f"Error checking element: {e}")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": str(e)},
            }