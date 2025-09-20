"""Minimal MCP server for HUD."""

from __future__ import annotations

import logging
import sys

from hud.server import MCPServer
from hud.server.context import attach_context
from hud.tools.types import EvaluationResult

# Configure logging to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)

mcp = MCPServer(name="test_test")
env = None


@mcp.initialize
async def init(ctx):
    global env
    env = attach_context("/tmp/hud_ctx.sock")
    logging.info("Connected to context server")


@mcp.shutdown
async def cleanup():
    global env
    env = None


@mcp.tool()
async def act() -> str:
    """Perform an action that changes the environment state."""
    if env is None:
        raise RuntimeError("Context not initialized")
    count = env.act()
    return f"Action #{count} performed. Current count: {count}"


@mcp.tool()
async def setup() -> str:
    """Reset the environment to initial state."""
    if env is None:
        raise RuntimeError("Context not initialized")
    env.reset()
    return "Counter reset to 0"


@mcp.tool()
async def evaluate(target: int = 10) -> EvaluationResult:
    """Check if the counter reached the target value."""
    if env is None:
        raise RuntimeError("Context not initialized")
    current_count = env.get_count()

    # Calculate reward as progress towards target
    reward = min(current_count / target, 1.0) if target > 0 else 0.0
    done = current_count >= target

    return EvaluationResult(
        reward=reward, done=done, content=f"Counter at {current_count}/{target}"
    )


if __name__ == "__main__":
    mcp.run()
