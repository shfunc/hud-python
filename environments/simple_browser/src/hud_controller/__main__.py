#!/usr/bin/env python3
"""Main entry point for HUD Controller."""

import sys
import os
from pathlib import Path
import typer

# Add apps directory to Python path
apps_dir = Path("/app/apps")
if apps_dir.exists():
    sys.path.insert(0, str(apps_dir))

app = typer.Typer()


@app.command()
def mcp(
    transport: str = typer.Option(
        "streamable-http",
        "--transport", 
        "-t", 
        help="Transport mode for MCP server"
    ),
):
    """Run the MCP server."""
    from . import mcp_server
    mcp_server.main(transport=transport)


@app.command()
def browser():
    """Launch the browser."""
    import asyncio
    from .browser import launch_browser
    asyncio.run(launch_browser())


if __name__ == "__main__":
    app() 