"""Page contains evaluator for remote browser environment."""

import logging
from typing import Union, List
from ..evaluators import evaluator

logger = logging.getLogger(__name__)


@evaluator("page_contains", description="Check if the page contains specific text")
class PageContainsEvaluator:
    """Evaluator that checks if the page contains specific text."""

    def __init__(self, context):
        self.context = context

    async def __call__(
        self, search_terms: Union[str, List[str]], partial_rewarding: bool = False
    ) -> dict:
        """
        Check if the page contains the specified text terms.

        Args:
            search_terms: Text to search for (string or list of strings)
            partial_rewarding: If True, give partial credit for finding some terms

        Returns:
            Standard evaluation result with reward between 0.0 and 1.0
        """
        logger.info(f"Evaluating page_contains for terms: {search_terms}")

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

        # Get page content
        try:
            content = await page.content()
            logger.info(f"Page content retrieved, length: {len(content)}")
        except Exception as e:
            logger.error(f"Failed to get page content: {e}")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "message": f"Failed to get page content: {str(e)}",
                },
            }

        # Normalize search terms to list
        if isinstance(search_terms, str):
            terms = [search_terms]
        else:
            terms = search_terms

        # Search for terms
        found_terms = []
        not_found_terms = []

        for term in terms:
            if term in content:
                found_terms.append(term)
                logger.info(f"✅ Found term: '{term}'")
            else:
                not_found_terms.append(term)
                logger.info(f"❌ Term not found: '{term}'")

        # Calculate reward
        if partial_rewarding and terms:
            reward = len(found_terms) / len(terms)
        else:
            reward = 1.0 if len(not_found_terms) == 0 else 0.0

        # Build info
        info = {
            "success": reward > 0,
            "found_terms": found_terms,
            "not_found_terms": not_found_terms,
            "total_terms": len(terms),
            "partial_rewarding": partial_rewarding,
        }

        if reward == 1.0:
            info["message"] = "All terms found on page"
        elif reward > 0:
            info["message"] = f"Found {len(found_terms)} of {len(terms)} terms"
        else:
            info["message"] = "No terms found on page"

        logger.info(f"Page contains evaluation complete. Reward: {reward}")

        return {
            "reward": float(reward),
            "done": True,
            "info": info,
        }
