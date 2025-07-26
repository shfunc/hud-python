"""MCP server with HUD tools."""

from __future__ import annotations

import asyncio
import sqlite3
import httpx
from typing import Dict, List, Any, Optional
from pathlib import Path

from mcp.types import ImageContent, TextContent
from hud.tools.base import ToolResult
from hud.tools import AnthropicComputerTool, HudComputerTool, OpenAIComputerTool
from hud.tools.helper import register_instance_tool
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("HUD Controller", port=8040, host="0.0.0.0")

# Register computer tool
register_instance_tool(mcp, "computer", HudComputerTool())
register_instance_tool(mcp, "computer_anthropic", AnthropicComputerTool())
register_instance_tool(mcp, "computer_openai", OpenAIComputerTool())

# Register custom tools
@mcp.tool()
async def api_request(
    method: str, # Changed from Literal
    path: str,
    base_url: str = "http://localhost:5000",
    json_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make an HTTP request to the backend API."""
    url = f"{base_url}{path}"
    print(f"Making API request: {method} {url}")
    async with httpx.AsyncClient() as client:
        if method == "GET":
            response = await client.get(url)
        elif method == "POST":
            response = await client.post(url, json=json_data)
        elif method == "PUT":
            response = await client.put(url, json=json_data)
        elif method == "DELETE":
            response = await client.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()

@mcp.tool()
def query_database(sql: str) -> List[Dict[str, Any]]:
    """Execute a read-only SQL query against the backend's SQLite database."""
    db_path = "/app/backend/app.db" # Assuming the database is here
    if not Path(db_path).exists():
        return {"error": f"Database not found at {db_path}"}

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row # Return rows as dicts
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        return {"error": f"Database query failed: {e}"}
    finally:
        if conn:
            conn.close()


def main(transport: str = "stdio"):
    # Run with specified transport
    if transport == "streamable-http":
        mcp.run(transport=transport)
    else:
        mcp.run(transport=transport)