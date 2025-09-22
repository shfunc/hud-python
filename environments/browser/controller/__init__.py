"""Browser controller - MCP tools for browser automation."""

import httpx
import logging
import sys
from hud.server import MCPServer

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)

# Explicit MCP server (frontend)
mcp = MCPServer()

# Configuration
ENV_SERVER_URL = "http://localhost:8000"

# Shared HTTP client to talk to the environment (backend)
http_client = httpx.AsyncClient(
    base_url=ENV_SERVER_URL,
    timeout=30.0,
    headers={"User-Agent": "HUD-Browser-Controller/1.0"}
)

# Import submodules to register decorators
from . import tools, resources, setup, evaluate, hooks

__all__ = ["mcp", "http_client", "ENV_SERVER_URL"]