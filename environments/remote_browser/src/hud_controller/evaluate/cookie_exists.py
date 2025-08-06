"""Cookie exists evaluator for remote browser environment."""

import logging
from typing import Any, Union, List
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("cookie_exists", "Check if specific cookies exist")
class CookieExistsEvaluator(BaseEvaluator):
    """Evaluator that checks if specific cookies exist."""

    async def __call__(
        self,
        context: Any,
        cookie_names: Union[str, List[str]],
        partial_rewarding: bool = False,
        **kwargs,
    ) -> EvaluationResult:
        """Check if the specified cookies exist.

        Args:
            context: Browser context with playwright_tool
            cookie_names: Cookie name(s) to check for (string or list of strings)
            partial_rewarding: If True, give partial credit for finding some cookies
            **kwargs: Additional arguments

        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating cookie_exists for: {cookie_names}")

        # Context IS the playwright tool
        if not context or not hasattr(context, "page") or not context.page:
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
            cookie_dict = {cookie["name"]: cookie for cookie in cookies}
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

        # Normalize cookie names to list
        if isinstance(cookie_names, str):
            names = [cookie_names]
        else:
            names = cookie_names

        # Check for cookies
        found_cookies = []
        not_found_cookies = []

        for name in names:
            if name in cookie_dict:
                found_cookies.append(name)
                logger.info(f"✅ Found cookie: '{name}'")
            else:
                not_found_cookies.append(name)
                logger.info(f"❌ Cookie not found: '{name}'")

        # Calculate reward
        if partial_rewarding and names:
            reward = len(found_cookies) / len(names)
        else:
            reward = 1.0 if len(not_found_cookies) == 0 else 0.0

        # Build info
        info = {
            "success": reward > 0,
            "found_cookies": found_cookies,
            "not_found_cookies": not_found_cookies,
            "total_cookies_checked": len(names),
            "total_cookies_in_browser": len(cookies),
            "partial_rewarding": partial_rewarding,
        }

        if reward == 1.0:
            info["message"] = "All cookies found"
        elif reward > 0:
            info["message"] = f"Found {len(found_cookies)} of {len(names)} cookies"
        else:
            info["message"] = "No cookies found"

        logger.info(f"Cookie exists evaluation complete. Reward: {reward}")

        return {
            "reward": float(reward),
            "done": reward == 1.0,
            "info": info,
        }
