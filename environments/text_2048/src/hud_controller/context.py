"""
Context server for text-2048 that persists game state across hot-reloads.

Run this as a separate process to maintain game state during development.
"""

import asyncio
from hud.server.context import run_context_server
from .game import Game2048
import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Run the context server with Game2048 directly as the context
    game = Game2048()

    # Add a startup message
    logger.info(f"[Context] Starting with {game.size}x{game.size} game")
    asyncio.run(run_context_server(game, "/tmp/hud_ctx.sock"))
