"""Evaluators module for browser environment."""

from hud.tools.base import BaseHub

evaluate = BaseHub(
    name="evaluate",
    title="Browser Evaluators",
    description="Evaluate the current browser state",
)

from . import todo  # noqa: E402

__all__ = ["evaluate"]
