"""Evaluator to check if a Google Sheet contains specific text."""

import logging
from typing import Any, Union, List
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("sheet_contains", "Check if a Google Sheet contains specific text by copying content to clipboard")
class SheetContainsEvaluator(BaseEvaluator):
    """Evaluator that checks if a Google Sheet contains specific text by copying content to clipboard."""

    async def __call__(
        self,
        context: Any,
        args: Union[str, List[str]],
        partial_rewarding: bool = False,
        **kwargs
    ) -> EvaluationResult:
        """Check if the sheet contains the specified text by copying all content to clipboard.
        
        Args:
            context: Browser context with playwright_tool
            args: Search terms as string or list of strings
            partial_rewarding: Whether to give partial rewards
            **kwargs: Additional arguments
            
        Returns:
            Evaluation result
        """
        logger.info("Starting sheet_contains evaluation")
        
        # Get playwright tool from context
        if not context or not hasattr(context, 'page') or not context.page:
            logger.error("No browser page available")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": "No browser page available"}
            }
        
        page = context.page
        
        # Verify we're on a Google Sheets page
        current_url = page.url
        logger.info(f"Current page URL: {current_url}")
        
        if "docs.google.com/spreadsheets" not in current_url:
            logger.error(f"Not on a Google Sheets page! URL: {current_url}")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": f"Not on a Google Sheets page! URL: {current_url}"}
            }
        
        logger.info("Confirmed on Google Sheets page")
        
        # Process search terms
        search_terms = []
        if isinstance(args, str):
            search_terms = [args]
        elif isinstance(args, list):
            search_terms = args
        else:
            logger.error(f"Invalid args format: {args}. Expected string or list of strings.")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": f"Invalid args format. Expected string or list, got {type(args)}"}
            }
        
        if not search_terms:
            logger.error("No search terms provided")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": "No search terms provided"}
            }
        
        logger.info(f"Search terms to find: {search_terms}")
        
        try:
            # Select all cells using Ctrl+A
            logger.info("Selecting all cells with Ctrl+A")
            await page.keyboard.press("Control+A")
            await page.wait_for_timeout(500)
            
            # Copy to clipboard with Ctrl+C
            logger.info("Copying content to clipboard with Ctrl+C")
            await page.keyboard.press("Control+C")
            await page.wait_for_timeout(1000)
            
            # Get clipboard content
            logger.info("Getting clipboard content")
            clipboard_content = await page.evaluate("() => navigator.clipboard.readText()")
            
            if not clipboard_content:
                logger.warning("Clipboard content is empty")
                return {
                    "reward": 0.0,
                    "done": False,
                    "info": {"error": "Clipboard content is empty"}
                }
            
            logger.info(f"Clipboard content length: {len(clipboard_content)} characters")
            logger.debug(f"First 200 chars: {clipboard_content[:200]}...")
            
            # Check for search terms
            found_terms = []
            missing_terms = []
            
            for term in search_terms:
                if term.lower() in clipboard_content.lower():
                    found_terms.append(term)
                    logger.info(f"✓ Found term: '{term}'")
                else:
                    missing_terms.append(term)
                    logger.info(f"✗ Missing term: '{term}'")
            
            # Calculate reward
            if partial_rewarding and len(search_terms) > 0:
                reward = float(len(found_terms)) / len(search_terms)
                logger.info(f"Partial rewarding: {len(found_terms)}/{len(search_terms)} = {reward}")
            elif not missing_terms:
                reward = 1.0
                logger.info("All terms found!")
            else:
                reward = 0.0
                logger.info(f"Missing terms: {missing_terms}")
            
            success = not missing_terms
            
            return {
                "reward": float(reward),
                "done": success,
                "info": {
                    "success": success,
                    "found_terms": found_terms,
                    "missing_terms": missing_terms,
                    "total_terms": len(search_terms),
                    "clipboard_length": len(clipboard_content),
                }
            }
            
        except Exception as e:
            logger.error(f"Exception during sheet_contains evaluation: {str(e)}")
            return {
                "reward": 0.0,
                "done": False,
                "info": {"error": f"Failed to evaluate: {str(e)}"}
            }