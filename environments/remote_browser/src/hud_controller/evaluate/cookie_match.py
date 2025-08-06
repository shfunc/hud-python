"""Cookie match evaluator for remote browser environment."""

import logging
from typing import Any, Dict
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("cookie_match", "Check if cookies match expected values")
class CookieMatchEvaluator(BaseEvaluator):
    """Evaluator that checks if cookies match expected values."""

    async def __call__(
        self,
        context: Any,
        expected_cookies: Dict[str, str],
        **kwargs
    ) -> EvaluationResult:
        """Check if cookies match expected values.
        
        Args:
            context: Browser context with playwright_tool
            expected_cookies: Dictionary of cookie name to expected value
            **kwargs: Additional arguments
            
        Returns:
            Standard evaluation result
        """
        logger.info(f"Evaluating cookie_match for: {list(expected_cookies.keys())}")
        
        # Context IS the playwright tool
        if not context or not hasattr(context, 'page') or not context.page:
            logger.error("No browser page available")
            return {
                "reward": 0.0,
                "done": False,
                "info": {
                    "success": False,
                    "message": "No browser page available",
                },
            }
        
        # Get all cookies
        try:
            cookies = await context.page.context.cookies()
            cookie_dict = {cookie["name"]: cookie["value"] for cookie in cookies}
            logger.info(f"Found {len(cookies)} cookies in browser")
        except Exception as e:
            logger.error(f"Failed to get cookies: {e}")
            return {
                "reward": 0.0,
                "done": False,
                "info": {
                    "success": False,
                    "message": f"Failed to get cookies: {str(e)}",
                },
            }
        
        # Check cookie values
        matches = []
        mismatches = []
        missing = []
        
        for name, expected_value in expected_cookies.items():
            if name not in cookie_dict:
                missing.append(name)
                logger.info(f"❌ Cookie missing: '{name}'")
            elif cookie_dict[name] == expected_value:
                matches.append(name)
                logger.info(f"✅ Cookie matches: '{name}'")
            else:
                mismatches.append({
                    "name": name,
                    "expected": expected_value,
                    "actual": cookie_dict[name]
                })
                logger.info(f"❌ Cookie mismatch: '{name}' - expected '{expected_value}', got '{cookie_dict[name]}'")
        
        # Calculate reward
        total = len(expected_cookies)
        if total > 0:
            reward = len(matches) / total
        else:
            reward = 1.0
        
        # Build info
        info = {
            "success": reward == 1.0,
            "matches": matches,
            "mismatches": mismatches,
            "missing": missing,
            "total_expected": total,
        }
        
        if reward == 1.0:
            info["message"] = "All cookies match expected values"
        elif reward > 0:
            info["message"] = f"{len(matches)} of {total} cookies match"
        else:
            info["message"] = "No cookies match expected values"
        
        logger.info(f"Cookie match evaluation complete. Reward: {reward}")
        
        return {
            "reward": float(reward),
            "done": reward == 1.0,
            "info": info,
        }