"""Cookie match evaluator for remote browser environment."""

import logging
from typing import Dict, Any
from ..evaluators import evaluator

logger = logging.getLogger(__name__)


@evaluator("cookie_match", description="Check if cookies match expected values")
class CookieMatchEvaluator:
    """Evaluator that checks if cookies match expected values."""

    def __init__(self, context):
        self.context = context

    async def __call__(
        self, expected_cookies: Dict[str, Any], partial_rewarding: bool = False
    ) -> dict:
        """
        Check if cookies match expected values.

        Args:
            expected_cookies: Dictionary of cookie name to expected value/properties
            partial_rewarding: If True, give partial credit for matching some cookies

        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating cookie_match for: {list(expected_cookies.keys())}")

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

        # Check cookie matches
        matched_cookies = []
        mismatched_cookies = []
        not_found_cookies = []

        for name, expected_value in expected_cookies.items():
            if name not in cookie_dict:
                not_found_cookies.append(name)
                logger.info(f"❌ Cookie not found: '{name}'")
                continue

            actual_cookie = cookie_dict[name]

            # Handle different types of expected values
            if isinstance(expected_value, dict):
                # Check multiple properties
                all_match = True
                for prop, expected_prop_value in expected_value.items():
                    if actual_cookie.get(prop) != expected_prop_value:
                        all_match = False
                        break

                if all_match:
                    matched_cookies.append(name)
                    logger.info(f"✅ Cookie matches: '{name}'")
                else:
                    mismatched_cookies.append(
                        {
                            "name": name,
                            "expected": expected_value,
                            "actual": {k: actual_cookie.get(k) for k in expected_value.keys()},
                        }
                    )
                    logger.info(f"❌ Cookie mismatch: '{name}'")
            else:
                # Simple value check
                if actual_cookie.get("value") == str(expected_value):
                    matched_cookies.append(name)
                    logger.info(f"✅ Cookie matches: '{name}' = '{expected_value}'")
                else:
                    mismatched_cookies.append(
                        {
                            "name": name,
                            "expected": expected_value,
                            "actual": actual_cookie.get("value"),
                        }
                    )
                    logger.info(
                        f"❌ Cookie mismatch: '{name}' (expected: '{expected_value}', actual: '{actual_cookie.get('value')}')"
                    )

        # Calculate reward
        total_checks = len(expected_cookies)
        if partial_rewarding and total_checks > 0:
            reward = len(matched_cookies) / total_checks
        else:
            reward = 1.0 if (len(mismatched_cookies) == 0 and len(not_found_cookies) == 0) else 0.0

        # Build info
        info = {
            "success": reward > 0,
            "matched_cookies": matched_cookies,
            "mismatched_cookies": mismatched_cookies,
            "not_found_cookies": not_found_cookies,
            "total_cookies_checked": total_checks,
            "partial_rewarding": partial_rewarding,
        }

        if reward == 1.0:
            info["message"] = "All cookies match expected values"
        elif reward > 0:
            info["message"] = f"Matched {len(matched_cookies)} of {total_checks} cookies"
        else:
            info["message"] = "No cookies match expected values"

        logger.info(f"Cookie match evaluation complete. Reward: {reward}")

        return {
            "reward": float(reward),
            "done": True,
            "info": info,
        }
