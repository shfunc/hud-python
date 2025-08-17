"""Setup module for browser environment."""

from hud.tools.base import BaseHub

setup = BaseHub(
    name="setup",
    title="Browser Setup",
    description="Initialize or configure the browser environment",
)

from . import todo, apps  # noqa: E402

__all__ = ["setup"]
