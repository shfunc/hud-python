"""Evaluators package for remote browser environment."""

from .registry import EvaluatorRegistry, evaluator, evaluator_logger
from .context import RemoteBrowserContext

# Import evaluators to trigger registration
from .url_match import UrlMatchEvaluator
from .page_contains import PageContainsEvaluator
from .cookie_exists import CookieExistsEvaluator
from .cookie_match import CookieMatchEvaluator
from .history_length import HistoryLengthEvaluator
from .raw_last_action_is import RawLastActionIsEvaluator
from .selector_history import SelectorHistoryEvaluator
from .sheet_contains import SheetContainsEvaluator
from .sheets_cell_values import SheetsCellValuesEvaluator
from .verify_type_action import VerifyTypeActionEvaluator

__all__ = [
    "EvaluatorRegistry",
    "evaluator",
    "RemoteBrowserContext",
]
