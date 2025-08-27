"""2048 game setup tools."""

import logging
from typing import List, Optional
import httpx
from mcp.types import TextContent
from . import setup

logger = logging.getLogger(__name__)


@setup.tool("game_2048_board")
async def game_2048_board(board_size: int = 4, target_tile: int = 2048):
    """Initialize new game with specified board size and target tile.

    Args:
        board_size: Size of the game board (3-6)
        target_tile: Target tile value (64, 128, 256, 512, 1024, 2048, etc.)

    Returns:
        SetupResult with game initialization status
    """
    try:
        # Get the backend port from persistent context
        persistent_ctx = setup.env
        backend_port = persistent_ctx.get_app_backend_port("2048")

        # Initialize new game
        url = f"http://localhost:{backend_port}/api/game/new"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, json={"board_size": board_size, "target_tile": target_tile}
            )
            response.raise_for_status()

        return TextContent(
            text=f"{board_size}x{board_size} game initialized with target {target_tile}",
            type="text",
        )
    except Exception as e:
        logger.error(f"game_2048_board failed: {e}")
        return TextContent(text=f"Failed to initialize game: {str(e)}", type="text")


@setup.tool("game_2048_set_board")
async def game_2048_set_board(board: List[List[int]], score: int = 0, moves: int = 0):
    """Set specific board configuration for testing.

    Args:
        board: 2D list representing the board state
        score: Initial score
        moves: Initial move count

    Returns:
        SetupResult with board configuration status
    """
    try:
        # Get the backend port from persistent context
        persistent_ctx = setup.env
        backend_port = persistent_ctx.get_app_backend_port("2048")

        # Set the board configuration
        url = f"http://localhost:{backend_port}/api/eval/set_board"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"board": board, "score": score, "moves": moves})
            response.raise_for_status()
            result = response.json()

        highest_tile = result.get("highest_tile", 0)

        return TextContent(
            text=f"Board set successfully (highest tile: {highest_tile})",
            type="text",
        )
    except Exception as e:
        logger.error(f"game_2048_set_board failed: {e}")
        return TextContent(text=f"Failed to set board: {str(e)}", type="text")


@setup.tool("game_2048_near_win")
async def game_2048_near_win(target_tile: int = 2048):
    """Set board close to winning (with tiles near target).

    Args:
        target_tile: The target tile for the game

    Returns:
        SetupResult with near-win board status
    """
    try:
        # Get the persistent context (just like remote browser does)
        persistent_ctx = setup.env  # Get BrowserContext from hub

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

        # Get the backend port for the 2048 app directly from service manager
        service_manager = persistent_ctx.get_service_manager()
        backend_port = service_manager.get_app_port("2048")
        url = f"http://localhost:{backend_port}/api/eval/set_board"

        # Make the API call directly
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"board": board, "score": score, "moves": moves})
            response.raise_for_status()
            result = response.json()

        return TextContent(
            text=f"Board set near winning state for target {target_tile}",
            type="text",
        )
    except Exception as e:
        logger.error(f"game_2048_near_win failed: {e}")
        return TextContent(text=f"Failed to set near-win board: {str(e)}", type="text")


@setup.tool("game_2048_navigate")
async def game_2048_navigate(url: Optional[str] = None):
    """Navigate to 2048 game.

    Args:
        url: Optional custom URL to navigate to

    Returns:
        SetupResult with navigation status
    """
    logger.info(f"Navigating to 2048 game")

    # Get the playwright tool from the environment context (remote browser pattern)
    persistent_ctx = setup.env
    playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No playwright tool available")
        return TextContent(text="No browser available for navigation", type="text")

    # Default to localhost:3000 if no URL provided
    if not url:
        url = "http://localhost:3000"

    # Navigate using the playwright tool
    result = await playwright_tool.navigate(url)

    if result.get("success"):
        logger.info(f"Successfully navigated to {url}")
        return TextContent(
            text=f"Navigated to 2048 game at {url} - Title: {result.get('title', 'Unknown')}",
            type="text",
        )
    else:
        logger.error(f"Failed to navigate: {result.get('error')}")
        return TextContent(text=f"Navigation failed: {result.get('error')}", type="text")


@setup.tool("game_2048_reset")
async def game_2048_reset():
    """Reset game to initial state.

    Returns:
        SetupResult with reset status
    """
    logger.info("Resetting 2048 game")

    # Get the playwright tool from the environment context (remote browser pattern)
    persistent_ctx = setup.env
    playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No playwright tool available")
        return TextContent(text="No browser available", type="text")

    try:
        # Navigate to ensure we're on the game page
        await playwright_tool.navigate("http://localhost:3000")

        # Click "New Game" button to reset
        await playwright_tool.page.click('button:has-text("New Game")')

        # Wait a moment for the game to reset
        await playwright_tool.page.wait_for_timeout(500)

        return TextContent(
            text="Game reset to initial state",
            type="text",
        )
    except Exception as e:
        logger.error(f"game_2048_reset failed: {e}")
        return TextContent(text=f"Failed to reset game: {str(e)}", type="text")


@setup.tool("game_2048_test_seed")
async def game_2048_test_seed():
    """Seed the board with a test configuration.

    Returns:
        SetupResult with test seed status
    """
    try:
        # Get the backend port from persistent context
        persistent_ctx = setup.env
        backend_port = persistent_ctx.get_app_backend_port("2048")
        url = f"http://localhost:{backend_port}/api/eval/seed"

        # Make the API call directly
        async with httpx.AsyncClient() as client:
            response = await client.post(url)
            response.raise_for_status()
            result = response.json()

        return TextContent(
            text="Test board seeded successfully",
            type="text",
        )
    except Exception as e:
        logger.error(f"game_2048_test_seed failed: {e}")
        return TextContent(text=f"Failed to seed test board: {str(e)}", type="text")
