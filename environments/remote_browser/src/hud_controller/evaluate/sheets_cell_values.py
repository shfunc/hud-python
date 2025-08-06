"""Evaluator to check if specific cells in a Google Sheet have expected values."""

import logging
from typing import Any, Dict
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("sheets_cell_values", "Check if specific cells in a Google Sheet have expected values")
class SheetsCellValuesEvaluator(BaseEvaluator):
    """Evaluator that checks if specific cells in a Google Sheet have expected values."""

    async def __call__(self, context: Any, **kwargs) -> EvaluationResult:
        """Check if specific cells in a Google Sheet have expected values.
        
        Accepts either:
        - Direct cell mappings: __call__(A1="value", B2="value")
        - args parameter with list: __call__(args=[{"A1": "value", "B2": "value"}])
        - args parameter with dict: __call__(args={"A1": "value", "B2": "value"})
        
        Args:
            context: Browser context with playwright_tool
            **kwargs: Cell values and optional partial_rewarding flag
            
        Returns:
            Evaluation result dict with reward, done, and info
        """
        logger.info("Starting sheets_cell_values evaluation")
        logger.info(f"Received kwargs: {kwargs}")
        
        # Extract cell values from kwargs
        if "args" in kwargs:
            args_value = kwargs["args"]
            if isinstance(args_value, list) and len(args_value) > 0:
                # Handle args as list: args=[{"A1": "value"}]
                cell_values = args_value[0] if isinstance(args_value[0], dict) else {}
            elif isinstance(args_value, dict):
                # Handle args as dict: args={"A1": "value"}
                cell_values = args_value
            else:
                cell_values = {}
        else:
            # Direct kwargs: A1="value", B2="value"
            cell_values = {k: v for k, v in kwargs.items() if k != "partial_rewarding"}
        
        partial_rewarding = kwargs.get("partial_rewarding", False)
        
        logger.info(f"Cell values to check: {cell_values}")
        
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
                "info": {"error": f"Not on a Google Sheets page! URL: {current_url}"},
            }
        
        logger.info("Confirmed on Google Sheets page")
        
        # Validate cell_values
        if not isinstance(cell_values, dict):
            logger.error(
                f"Invalid cell values format: {cell_values}. Expected dictionary of cell references to values."
            )
            return {
                "reward": 0.0,
                "done": False,
                "info": {
                    "error": f"Invalid cell values format. Expected dict, got {type(cell_values)}"
                },
            }
        
        if not cell_values:
            logger.warning("No cell values to check")
            return {
                "reward": 1.0, 
                "done": True, 
                "info": {"message": "No cell values to check"}
            }
        
        # Get all the cell values from the sheet
        try:
            logger.info("Attempting to extract cell values from the page...")
            
            # JavaScript to get all cell values
            js_code = """
            () => {
                const cells = {};
                const elements = document.querySelectorAll('[id^="waffle-grid-container"] [dir="ltr"]');
                
                elements.forEach(el => {
                    const id = el.id;
                    if (id && id.includes('-')) {
                        const parts = id.split('-');
                        const cellRef = parts[parts.length - 1];
                        if (cellRef && /^[A-Z]+[0-9]+$/.test(cellRef)) {
                            const spans = el.querySelectorAll('span');
                            let value = '';
                            if (spans.length > 0) {
                                value = Array.from(spans).map(s => s.textContent || '').join('');
                            }
                            cells[cellRef] = value;
                        }
                    }
                });
                
                return cells;
            }
            """
            sheet_cells = await page.evaluate(js_code)
            logger.info(f"Found {len(sheet_cells)} cells in the sheet")
            logger.info(f"Sheet cells: {sheet_cells}")
            
            # Check each expected cell value
            total_cells = len(cell_values)
            matching_cells = 0
            mismatches = []
            
            for cell_ref, expected_value in cell_values.items():
                actual_value = sheet_cells.get(cell_ref, "")
                logger.info(
                    f"Checking {cell_ref}: expected='{expected_value}', actual='{actual_value}'"
                )
                
                if str(actual_value).strip() == str(expected_value).strip():
                    matching_cells += 1
                    logger.info(f"✓ {cell_ref} matches")
                else:
                    mismatches.append(
                        {
                            "cell": cell_ref,
                            "expected": expected_value,
                            "actual": actual_value,
                        }
                    )
                    logger.warning(
                        f"✗ {cell_ref} mismatch: expected '{expected_value}', got '{actual_value}'"
                    )
            
            # Calculate reward
            if partial_rewarding:
                reward = matching_cells / total_cells if total_cells > 0 else 0.0
            else:
                reward = 1.0 if matching_cells == total_cells else 0.0
            
            success = matching_cells == total_cells
            logger.info(
                f"Evaluation result: {matching_cells}/{total_cells} cells match, reward={reward}"
            )
            
            return {
                "reward": float(reward),
                "done": success,
                "info": {
                    "success": success,
                    "matching_cells": matching_cells,
                    "total_cells": total_cells,
                    "mismatches": mismatches,
                },
            }
            
        except Exception as e:
            logger.error(f"Error evaluating sheet cells: {str(e)}", exc_info=True)
            return {
                "reward": 0.0, 
                "done": False, 
                "info": {"error": f"Failed to evaluate: {str(e)}"}
            }