#!/usr/bin/env python3
"""
Simple Browser Environment Example

This example demonstrates how to use the Simple Browser MCP environment
with computer control, API requests, and database queries.

For Docker with stdio (recommended):
  # Build the Docker image first:
  cd environments/simple_browser
  docker build -t hud-browser .

  # Run this script:
  python examples/environments/simple_browser_example.py --stdio

For accessing VNC viewer:
  Open http://localhost:8080/vnc.html in your browser
"""

import asyncio
import sys
import json
import logging
from hud.mcp import ClaudeMCPAgent
from mcp_use import MCPClient

# Simple logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main(use_http=False):
    if not use_http:
        # Launch Docker container and connect via stdio
        config = {
            "mcp_config": {
                "browser": {
                    "command": "docker",
                    "args": [
                        "run",
                        "--rm",
                        "-i",
                        "-p",
                        "6080:6080",  # Port for VNC viewer
                        # "-e",
                        # "LAUNCH_APPS=todo",  # Launch todo app
                        # "-e",
                        # "BROWSER_URL=http://localhost:3000",  # Navigate to todo app
                        # "-e",
                        # "ID=example-send-email",
                        "gmail-clone",  # Docker image name
                    ],
                }
            }
        }
        print("🚀 Launching Docker container...")
        print("🖥️  Access VNC at: http://localhost:8080/vnc.html")
        print("📱 Todo app at: http://localhost:3000")
    else:
        # For Docker/HTTP transport (requires docker-compose)
        config = {"mcp_config": {"browser": {"url": "http://localhost:8041/mcp"}}}
        print("🚀 Using HTTP transport mode")

    # Create MCP client and session
    print("📡 Connecting to browser environment...")
    try:
        client = MCPClient.from_dict(config)
        session = await client.create_session("browser")
        print("✅ Connected successfully!")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return

    # Create agent with available tools
    agent = ClaudeMCPAgent(
        mcp_client=client,
        model="claude-sonnet-4-20250514",
        allowed_tools=["computer", "playwright", "launch_app", "api_request", "query_database"],
    )

    # Demo: Take a screenshot
    print("\n📸 Taking initial screenshot...")
    try:
        result = await agent.run("Take a screenshot of the current browser view")
        print("Screenshot taken successfully!")
    except Exception as e:
        print(f"❌ Failed to take screenshot: {e}")

    # Interactive mode
    print("\n💬 Interactive mode - Enter commands to control the browser")
    print("Examples:")
    print("  - Take a screenshot")
    print("  - Click the first todo item checkbox")
    print("  - Add a new todo item")

    try:
        while True:
            try:
                query = input("\n> ")
                if query.lower() in ["quit", "exit", "q"]:
                    break

                result = await agent.run(query)
                print(f"Result: {result}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        await client.close_all_sessions()
        print("✅ Done!")


if __name__ == "__main__":
    # Check command line flags
    use_http = "--http" in sys.argv

    if not use_http:
        print("🐳 Simple Browser MCP Environment Example")
        print("=" * 50)
        print("\nMake sure you have built the Docker image:")
        print("  cd environments/simple_browser")
        print("  docker build -t hud-browser .")
        print("\nPress Enter to continue or Ctrl+C to cancel...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        asyncio.run(main(use_http))
    else:
        print("🚀 Using HTTP transport (Docker Compose mode)")
        print("Make sure Docker Compose is running:")
        print("  cd environments/simple_browser && docker-compose up -d")
        asyncio.run(main(use_http))
