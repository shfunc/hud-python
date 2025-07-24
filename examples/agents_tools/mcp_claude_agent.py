#!/usr/bin/env python3
"""Claude MCP agent example for HUD tools via HTTP."""

import asyncio
from dotenv import load_dotenv
import hud
from mcp_use import MCPClient
from hud.mcp_agent import ClaudeMCPAgent

load_dotenv()

# To run this locally: python -m hud.tools.helper.mcp_server http --port 8039
# This will start the computer use MCP server on your machine.
BASE_URL = "http://localhost:8039/mcp"


async def main():
    """Run Claude MCP agent with HUD tools."""

    # Configure MCP client to connect to the router
    config = {"mcpServers": {"hud": {"url": BASE_URL}}}

    # Create client
    client = MCPClient.from_dict(config)

    # Create Claude agent
    agent = ClaudeMCPAgent(
        client=client,
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        # initial_screenshot=True,
        display_width_px=1024,
        display_height_px=768,
        # append_tool_system_prompt=True,
        # custom_system_prompt="You are a helpful assistant that can control the computer to help users with their tasks.",
        allowed_tools=["computer_anthropic"],  # Only allow the Anthropic computer tool
    )

    try:
        # Run the agent
        query = "Click on the chat in the bottom right corner, and type 'Hello, how are you?'"
        print(f"\n🤖 Running: {query}\n")

        # Use trace_debug to see MCP calls in real-time
        with hud.trace_sync():
            result = await agent.run(query, max_iterations=15)

        print(f"\n✅ Result: {result}")

    finally:
        await client.close_all_sessions()


if __name__ == "__main__":
    print(f"🚀 Connecting to MCP router at {BASE_URL}")
    asyncio.run(main())
