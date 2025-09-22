"""DeepResearch controller - MCP tools that call the environment HTTP API."""

import logging
import os
import sys
import httpx
from hud.server import MCPServer

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)

# MCP server (frontend/controller)
mcp = MCPServer(name="deepresearch")

# Environment server URL (backend)
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

# Shared HTTP client to talk to the environment
http_client = httpx.AsyncClient(
    base_url=ENV_SERVER_URL,
    timeout=30.0,
    headers={"User-Agent": "HUD-DeepResearch-Controller/1.0"},
)

# Register server (contains hooks and tools)
from . import server

__all__ = ["mcp", "http_client", "ENV_SERVER_URL"]
