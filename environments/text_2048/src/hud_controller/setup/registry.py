"""Setup functions for 2048 environment."""

import logging
from hud.tools import BaseSetup, SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup("board", "Initialize 2048 game with configurable board size")
class Game2048Setup(BaseSetup):
    """Flexible 2048 game setup that supports different board sizes."""

    async def __call__(self, context, board_size: int = 4, **kwargs) -> SetupResult:
        """Initialize a new game with the specified board size.

        Args:
            context: The current game instance
            board_size: Size of the game board (default 4 for 4x4)
            **kwargs: Additional arguments

        Returns:
            Setup result with status and message
        """
        # Reinitialize the game with new size
        context.reset(size=board_size)

        logger.info(f"Reset game to {board_size}x{board_size}")

        return {
            "status": "success",
            "message": f"{board_size}x{board_size} game initialized",
            "board_size": board_size,
        }
