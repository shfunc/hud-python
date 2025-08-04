"""Cookie exists evaluator for remote browser environment."""

import logging
from typing import Union, List
from ..evaluators import evaluator

logger = logging.getLogger(__name__)


@evaluator("cookie_exists", description="Check if specific cookies exist")
class CookieExistsEvaluator:
    """Evaluator that checks if specific cookies exist."""

    def __init__(self, context):
        self.context = context

    async def __call__(
        self, cookie_names: Union[str, List[str]], partial_rewarding: bool = False
    ) -> dict:
        """
        Check if the specified cookies exist.

        Args:
            cookie_names: Cookie name(s) to check for (string or list of strings)
            partial_rewarding: If True, give partial credit for finding some cookies

        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating cookie_exists for: {cookie_names}")

        # Get the browser context
        browser_context = self.context.context
        if not browser_context:
            logger.error("No browser context available")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "message": "No browser context available",
                },
            }

        # Get all cookies
        try:
            cookies = await browser_context.cookies()
            cookie_dict = {cookie["name"]: cookie for cookie in cookies}
            logger.info(f"Found {len(cookies)} cookies in browser")
        except Exception as e:
            logger.error(f"Failed to get cookies: {e}")
            return {
                "reward": 0.0,
                "done": True,
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
            "done": True,
            "info": info,
        }
