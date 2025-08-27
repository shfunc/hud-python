"""Evaluators module for browser environment."""

from hud.tools.base import BaseHub

evaluate = BaseHub("evaluate")

from . import todo, game_2048  # noqa: E402

__all__ = ["evaluate"]
