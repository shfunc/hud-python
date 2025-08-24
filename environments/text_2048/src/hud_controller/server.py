"""
MCP server for text-based 2048 game environment using BaseHub pattern.
"""

import sys
import logging

from hud.server import MCPServer
from hud.server.context import attach_context

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
async def initialize_environment(ctx):
    """Initialize the 2048 environment."""
    global game

    logger.info("Initializing 2048 environment...")

    # Connect to context server (must be running)
    game = attach_context("/tmp/hud_ctx.sock")
    logger.info("Connected to socket-based game context")

    # Log whether we're resuming or starting fresh
    if game.get_moves_made() > 0:
        logger.info(f"Resuming game - Score: {game.get_score()}, Moves: {game.get_moves_made()}")
    else:
        logger.info("Starting fresh game")

    # Set up the game instance on hubs and tools
    setup_hub.env = game
    evaluate_hub.env = game

    # Mount hubs
    logger.info(f"Mounting hubs: {setup_hub} and {evaluate_hub}")

    mcp.mount(setup_hub)
    mcp.mount(evaluate_hub)

    # Create and register move tool
    mcp.add_tool(MoveTool(env=game))

    logger.info("2048 environment ready")


if __name__ == "__main__":
    mcp.run()
