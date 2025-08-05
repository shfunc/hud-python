"""Evaluator to check if a Google Sheet contains specific text."""

import logging
from typing import Dict, Any, List
from .registry import evaluator
from .context import RemoteBrowserContext
from . import evaluator_logger

logger = logging.getLogger(__name__)


@evaluator("sheet_contains")
class SheetContainsEvaluator:
    """Evaluator that checks if a Google Sheet contains specific text by copying content to clipboard."""

    name = "sheet_contains"

    async def evaluate(
        self,
        args: str | List[str],
        context: RemoteBrowserContext,
        partial_rewarding: bool = False,
    ) -> float:
        """
        Check if the sheet contains the specified text by copying all content to clipboard.

        Args:
            args: Search terms as string or list of strings
            context: The remote browser context
            partial_rewarding: Whether to give partial rewards

        Returns:
            1.0 if all terms found (or partial score if partial_rewarding), 0.0 otherwise
        """
        evaluator_logger.info("Starting sheet_contains evaluation")

        # Get page from context
        page = context.page
        if not page:
            evaluator_logger.error("No page available in context")
            return 0.0

        # Verify we're on a Google Sheets page
        current_url = page.url
        evaluator_logger.info(f"Current page URL: {current_url}")

        if "docs.google.com/spreadsheets" not in current_url:
            evaluator_logger.error(f"Not on a Google Sheets page! URL: {current_url}")
            return 0.0

        evaluator_logger.info("Confirmed on Google Sheets page")

        # Process search terms
        search_terms = []
        if isinstance(args, str):
            search_terms = [args]
        elif isinstance(args, list):
            search_terms = args
        else:
            evaluator_logger.error(
                f"Invalid args format: {args}. Expected string or list of strings."
            )
            return 0.0

        if not search_terms:
            evaluator_logger.error("No search terms provided")
            return 0.0

        evaluator_logger.info(f"Search terms to find: {search_terms}")

        try:
            # Select all cells using Ctrl+A
            evaluator_logger.info("Selecting all cells with Ctrl+A")
            await page.keyboard.press("Control+A")
            await page.wait_for_timeout(500)

            # Copy to clipboard with Ctrl+C
            evaluator_logger.info("Copying content to clipboard with Ctrl+C")
            await page.keyboard.press("Control+C")
            await page.wait_for_timeout(1000)

            # Get clipboard content
            evaluator_logger.info("Getting clipboard content")
            clipboard_content = await page.evaluate("() => navigator.clipboard.readText()")

            if not clipboard_content:
                evaluator_logger.warning("Clipboard content is empty")
                return 0.0

            evaluator_logger.info(f"Clipboard content length: {len(clipboard_content)} characters")
            evaluator_logger.debug(f"First 200 chars: {clipboard_content[:200]}...")

            # Check for search terms
            found_terms = []
            missing_terms = []

            for term in search_terms:
                if term.lower() in clipboard_content.lower():
                    found_terms.append(term)
                    evaluator_logger.info(f"✓ Found term: '{term}'")
                else:
                    missing_terms.append(term)
                    evaluator_logger.info(f"✗ Missing term: '{term}'")

            # Calculate reward
            if partial_rewarding and len(search_terms) > 0:
                reward = float(len(found_terms)) / len(search_terms)
                evaluator_logger.info(
                    f"Partial rewarding: {len(found_terms)}/{len(search_terms)} = {reward}"
                )
            elif not missing_terms:
                reward = 1.0
                evaluator_logger.info("All terms found!")
            else:
                reward = 0.0
                evaluator_logger.info(f"Missing terms: {missing_terms}")

        except Exception as e:
            evaluator_logger.error(f"Exception during sheet_contains evaluation: {str(e)}")
            reward = 0.0

        evaluator_logger.info(f"Final reward: {reward}")
        return reward
