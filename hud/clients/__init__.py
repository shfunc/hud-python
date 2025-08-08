"""HUD MCP client implementations."""

from .base import AgentMCPClient, BaseHUDClient
from .fastmcp import FastMCPHUDClient
from .mcp_use import MCPUseHUDClient

# Default to MCP-use for backward compatibility
MCPClient = FastMCPHUDClient

__all__ = [
    "AgentMCPClient",
    "BaseHUDClient",
    "FastMCPHUDClient",
    "MCPUseHUDClient",
    "MCPClient",
]
