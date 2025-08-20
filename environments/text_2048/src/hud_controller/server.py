"""
MCP server for text-based 2048 game environment using BaseHub pattern.
"""

import sys
import logging

from hud.server import MCPServer

from .game import Game2048
from .tools import MoveTool

# Configure logging to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress MCP server logs
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

# Global game instance (initialized during startup)
game = None

# Import setup/evaluate layers
from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub

# Create main server first
mcp = MCPServer(name="text-2048")


@mcp.initialize
async def initialize_environment(session=None, progress_token=None):
    """Initialize the 2048 environment with progress notifications."""
    global game

    async def send_progress(progress: int, message: str):
        if session and progress_token:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=progress,
                total=100,
                message=message,
            )

    logger.info("Initializing 2048 environment...")
    await send_progress(0, "Starting 2048 game environment...")

    # Create and initialize the game
    await send_progress(30, "Creating game instance...")

    game = Game2048()

    await send_progress(50, "Setting up game board...")

    # Initialize game state
    game.reset()

    # Set up the game instance on hubs and tools
    await send_progress(70, "Configuring tools and hubs...")

    setup_hub.env = game
    evaluate_hub.env = game

    # Mount hubs
    logger.info(f"Mounting hubs: {setup_hub} and {evaluate_hub}")

    mcp.mount(setup_hub)
    mcp.mount(evaluate_hub)

    # Create and register move tool
    mcp.add_tool(MoveTool(env=game))

    await send_progress(100, "2048 environment ready")


if __name__ == "__main__":
    mcp.run()
