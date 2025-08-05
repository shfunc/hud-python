"""URL match evaluator for remote browser environment."""

import logging
from typing import Any, Dict, Optional
from ..evaluators import evaluator

logger = logging.getLogger(__name__)


@evaluator("url_match", description="Check if the current URL contains a target string")
class UrlMatchEvaluator:
    """Evaluator that checks if the current URL contains a target string."""

    def __init__(self, context):
        self.context = context

    async def __call__(self, target_url: str) -> dict:
        """
        Check if the current URL contains the target string.

        Args:
            target_url: The target URL string to look for

        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating URL match for target: '{target_url}'")

        # Get the current page from context
        page = self.context.page
        if not page:
            logger.error("No page available in context")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "message": "No browser page available",
                },
            }

        # Get the current URL
        current_url = page.url
        logger.info(f"Current page URL: '{current_url}'")

        # Check if target URL is in current URL
        if target_url in current_url:
            logger.info(f"✅ URL match successful: '{target_url}' found in '{current_url}'")
            return {
                "reward": 1.0,
                "done": True,
                "info": {
                    "success": True,
                    "message": f"URL match successful",
                    "current_url": current_url,
                    "target_url": target_url,
                },
            }
        else:
            logger.info(f"❌ URL match failed: '{target_url}' not found in '{current_url}'")

            # Provide helpful debugging information
            info = {
                "success": False,
                "message": f"URL match failed",
                "current_url": current_url,
                "target_url": target_url,
            }

            # Check for partial matches
            if target_url.lower() in current_url.lower():
                info["note"] = "Case-insensitive match found"

            # Check for protocol differences
            if current_url.startswith("https://") and not target_url.startswith("https://"):
                alt_target = "https://" + target_url
                if alt_target in current_url:
                    info["note"] = "Match found with https:// prefix"

            return {
                "reward": 0.0,
                "done": True,
                "info": info,
            }
