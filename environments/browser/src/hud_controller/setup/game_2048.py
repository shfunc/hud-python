"""2048 game setup tools."""

import logging
from typing import List, Optional
from fastmcp import Context
from hud.tools.types import SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup.tool("game_2048_board")
async def game_2048_board(ctx: Context, board_size: int = 4, target_tile: int = 2048):
    """Initialize new game with specified board size and target tile.

    Args:
        board_size: Size of the game board (3-6)
        target_tile: Target tile value (64, 128, 256, 512, 1024, 2048, etc.)

    Returns:
        SetupResult with game initialization status
    """
    try:
        env = setup.env  # Get BrowserEnvironmentContext from hub
        result = await env.call_app_api(
            "2048",
            "/api/game/new",
            method="POST",
            json={"board_size": board_size, "target_tile": target_tile},
        )

        highest_tile = result.get("highest_tile", 0)

        return SetupResult(
            content=f"{board_size}x{board_size} game initialized with target {target_tile}",
            info={
                "board_size": board_size,
                "target_tile": target_tile,
                "highest_tile": highest_tile,
            },
        )
    except Exception as e:
        logger.error(f"game_2048_board failed: {e}")
        return SetupResult(content=f"Failed to initialize game: {str(e)}", isError=True)


@setup.tool("game_2048_set_board")
async def game_2048_set_board(ctx: Context, board: List[List[int]], score: int = 0, moves: int = 0):
    """Set specific board configuration for testing.

    Args:
        board: 2D list representing the board state
        score: Initial score
        moves: Initial move count

    Returns:
        SetupResult with board configuration status
    """
    try:
        env = setup.env  # Get BrowserEnvironmentContext from hub
        result = await env.call_app_api(
            "2048",
            "/api/eval/set_board",
            method="POST",
            json={"board": board, "score": score, "moves": moves},
        )

        highest_tile = result.get("highest_tile", 0)

        return SetupResult(
            content=f"Board set successfully (highest tile: {highest_tile})",
            info={"board": board, "score": score, "moves": moves, "highest_tile": highest_tile},
        )
    except Exception as e:
        logger.error(f"game_2048_set_board failed: {e}")
        return SetupResult(content=f"Failed to set board: {str(e)}", isError=True)


@setup.tool("game_2048_near_win")
async def game_2048_near_win(ctx: Context, target_tile: int = 2048):
    """Set board close to winning (with tiles near target).

    Args:
        target_tile: The target tile for the game

    Returns:
        SetupResult with near-win board status
    """
    try:
        # Create a board that's one move away from target
        if target_tile == 2048:
            board = [[1024, 1024, 256, 128], [512, 256, 64, 32], [128, 64, 16, 8], [32, 16, 4, 2]]
        elif target_tile == 1024:
            board = [[512, 512, 128, 64], [256, 128, 32, 16], [64, 32, 8, 4], [16, 8, 2, 0]]
        elif target_tile == 512:
            board = [[256, 256, 64, 32], [128, 64, 16, 8], [32, 16, 4, 2], [8, 4, 2, 0]]
        else:
            # Generic near-win board
            half_target = target_tile // 2
            quarter_target = target_tile // 4
            board = [
                [half_target, half_target, quarter_target, quarter_target // 2],
                [quarter_target, quarter_target // 2, 16, 8],
                [16, 8, 4, 2],
                [4, 2, 0, 0],
            ]

        # Set a high score and move count
        score = sum(sum(row) for row in board) * 2
        moves = 150

        env = setup.env  # Get BrowserEnvironmentContext from hub
        result = await env.call_app_api(
            "2048",
            "/api/eval/set_board",
            method="POST",
            json={"board": board, "score": score, "moves": moves},
        )

        return SetupResult(
            content=f"Board set near winning state for target {target_tile}",
            info={
                "target_tile": target_tile,
                "highest_tile": result.get("highest_tile", 0),
                "score": score,
                "moves": moves,
            },
        )
    except Exception as e:
        logger.error(f"game_2048_near_win failed: {e}")
        return SetupResult(content=f"Failed to set near-win board: {str(e)}", isError=True)


@setup.tool("game_2048_navigate")
async def game_2048_navigate(ctx: Context, url: str = None):
    """Navigate to 2048 game.

    Args:
        url: Optional custom URL to navigate to

    Returns:
        SetupResult with navigation status
    """
    try:
        env = setup.env  # Get BrowserEnvironmentContext from hub
        # Get the default URL if not provided
        if not url:
            url = env.get_app_url("2048")

        # Use Playwright to navigate
        if env.playwright:
            nav_result = await env.playwright.navigate(url)

            return SetupResult(
                content=f"Navigated to 2048 game at {url}",
                info={
                    "url": url,
                },
            )
        else:
            return SetupResult(content="Playwright tool not available for navigation", isError=True)
    except Exception as e:
        logger.error(f"game_2048_navigate failed: {e}")
        return SetupResult(content=f"Failed to navigate to 2048 game: {str(e)}", isError=True)


@setup.tool("game_2048_reset")
async def game_2048_reset(ctx: Context):
    """Reset game to initial state.

    Returns:
        SetupResult with reset status
    """
    try:
        env = setup.env  # Get BrowserEnvironmentContext from hub
        result = await env.call_app_api("2048", "/api/eval/reset", method="POST")

        return SetupResult(
            content="Game reset to initial state",
            info={
                "board_size": result.get("board_size", 4),
                "target_tile": result.get("target_tile", 2048),
            },
        )
    except Exception as e:
        logger.error(f"game_2048_reset failed: {e}")
        return SetupResult(content=f"Failed to reset game: {str(e)}", isError=True)


@setup.tool("game_2048_test_seed")
async def game_2048_test_seed(ctx: Context):
    """Seed the board with a test configuration.

    Returns:
        SetupResult with test seed status
    """
    try:
        env = setup.env  # Get BrowserEnvironmentContext from hub
        result = await env.call_app_api("2048", "/api/eval/seed", method="POST")

        return SetupResult(
            content="Test board seeded successfully",
            info={
                "highest_tile": result.get("highest_tile", 0),
                "message": result.get("message", ""),
            },
        )
    except Exception as e:
        logger.error(f"game_2048_test_seed failed: {e}")
        return SetupResult(content=f"Failed to seed test board: {str(e)}", isError=True)
