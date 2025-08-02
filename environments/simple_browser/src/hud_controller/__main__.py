#!/usr/bin/env python3
"""HUD Controller MCP Server."""

from .server import mcp


def main():
    """Main entry point for the package."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
