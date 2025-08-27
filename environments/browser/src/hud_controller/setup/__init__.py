"""Setup module for browser environment."""

from hud.tools.base import BaseHub

setup = BaseHub("setup")

# Import all setup tools to register them
from . import game_2048, todo

__all__ = ["setup"]
