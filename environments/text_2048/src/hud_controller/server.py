"""
MCP server for text-based 2048 game environment.
"""

import sys
import logging
from mcp.server.fastmcp import FastMCP

from hud.tools.helper import mcp_intialize_wrapper, register_instance_tool

from .game import Game2048
from .tools import MoveTool
from .setup import setup_tool
from .evaluate import evaluate_tool

# Configure logging to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global game instance
game: Game2048 | None = None

# Initialize MCP server
mcp = FastMCP(name="text-2048")


# Register resources for discovering available functions
@mcp.resource("setup://registry")
async def get_setup_registry() -> str:
    """Get available setup functions."""
    return setup_tool.get_registry_json()


@mcp.resource("evaluators://registry")
async def get_evaluator_registry() -> str:
    """Get available evaluator functions."""
    return evaluate_tool.get_registry_json()


@mcp_intialize_wrapper()
async def initialize_environment(session=None, progress_token=None):
    """Initialize the 2048 environment."""
    global game
    logger.info("Initializing 2048 environment...")

    # Create default game
    game = Game2048()

    # Create the move tool with the game as context
    move_tool = MoveTool(context=game)

    # Set contexts for setup and evaluate tools - the game IS the context
    setup_tool.context = game
    evaluate_tool.context = game

    # Register all tools with MCP
    register_instance_tool(mcp, move_tool)
    register_instance_tool(mcp, setup_tool)
    register_instance_tool(mcp, evaluate_tool)

    logger.info("2048 environment ready with all tools registered")


if __name__ == "__main__":
    mcp.run()
