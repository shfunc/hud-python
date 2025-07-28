#!/usr/bin/env python3
"""Claude MCP agent example for HUD tools via HTTP."""

import asyncio
import os
from dotenv import load_dotenv
import hud
from mcp_use import MCPClient
from hud.mcp_agent import ClaudeMCPAgent

load_dotenv()

# To control your own computer: python -m hud.tools.helper.mcp_server http --port 8041
# This will start the computer use MCP server on your machine.
# BASE_URL = "http://localhost:8041/mcp"

# To run inside a docker container, see environments/simple_browser/README.md

# To run on the cloud:
BASE_URL = "https://orchestrator-v3.up.railway.app"

HUD_API_KEY = os.getenv("HUD_API_KEY")


async def main():
    with hud.trace() as run_id:  # This will show you telemetry for any MCP using agent
        """Run Claude MCP agent with HUD tools."""

        # Configure MCP client to connect to the router
        config = {
            "mcpServers": {
                "hud": {
                    "url": f"{BASE_URL}/v3/mcp",
                    "headers": {  # This is how the cloud server is configured to work
                        "Authorization": f"Bearer {HUD_API_KEY}",
                        "Mcp-Image": "hudpython/hud-browser:latest",
                        "Run-Id": run_id,
                    },
                }
            }
        }

        # Create client
        client = MCPClient.from_dict(config)

        # Create Claude agent
        agent = ClaudeMCPAgent(
            client=client,
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            # initial_screenshot=True,
            display_width_px=1400,
            display_height_px=850,
            # append_tool_system_prompt=True,
            # custom_system_prompt="You are a helpful assistant that can control the computer to help users with their tasks.",
            allowed_tools=["anthropic_computer"],  # Only allow the Anthropic computer tool
        )

        try:
            query = input("Enter a query: ")
            result = await agent.run(query, max_steps=15)

            print(f"\n✅ Result: {result}")

        finally:
            await client.close_all_sessions()


if __name__ == "__main__":
    print(f"🚀 Connecting to MCP router at {BASE_URL}")
    asyncio.run(main())
