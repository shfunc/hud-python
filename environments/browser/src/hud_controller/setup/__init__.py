"""Setup module for browser environment."""

from hud.tools.base import BaseHub

setup = BaseHub(
    name="setup",
    title="Browser Setup",
    description="Initialize or configure the browser environment",
)

# Import all setup tools to register them
from . import game_2048, todo

__all__ = ["setup"]
