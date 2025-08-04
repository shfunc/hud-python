"""Setup functions for remote browser environment."""

from .registry import SetupRegistry, setup

# Import setup functions to trigger registration
from .navigate import NavigateSetup
from .cookies import SetCookiesSetup, ClearCookiesSetup
from .interact import ClickElementSetup, TypeTextSetup, WaitForElementSetup
from .sheets import SheetsFromXlsxSetup, SheetsFromBytesSetup
from .load_html import LoadHtmlContentSetup

__all__ = [
    "SetupRegistry",
    "setup",
]
