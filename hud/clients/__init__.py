"""HUD MCP client implementations."""

from __future__ import annotations

from .base import AgentMCPClient, BaseHUDClient
from .fastmcp import FastMCPHUDClient
from .mcp_use import MCPUseHUDClient

# Default to MCP-use for new features
MCPClient = MCPUseHUDClient

__all__ = [
    "AgentMCPClient",
    "BaseHUDClient",
    "FastMCPHUDClient",
    "MCPClient",
]
