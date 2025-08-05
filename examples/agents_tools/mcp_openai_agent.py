#!/usr/bin/env python3
"""OpenAI MCP agent example for HUD tools via HTTP."""

import asyncio
from dotenv import load_dotenv
import hud
from mcp_use import MCPClient
from hud.mcp import OpenAIMCPAgent

load_dotenv()

# To run this locally: python -m hud.tools.helper.mcp_server http --port 8041
# This will start the computer use MCP server on your machine.
BASE_URL = "http://localhost:8041/mcp"


async def main():
    """Run OpenAI MCP agent with HUD tools."""

    # Configure MCP client to connect to the router
    config = {"mcp_config": {"hud": {"url": BASE_URL}}}

    # Create client
    client = MCPClient.from_dict(config)

    # Create OpenAI agent
    agent = OpenAIMCPAgent(
        mcp_client=client,
        model="computer-use-preview",  # OpenAI's computer use model
        environment="browser",  # Can be "windows", "mac", "linux", or "browser"
        # initial_screenshot=True,
        display_width=1024,
        display_height=768,
        # append_tool_system_prompt=True,
        # custom_system_prompt="You are an autonomous agent that completes tasks without asking for confirmation. When asked to click on something or type a message, DO IT immediately without asking permission. The user has already authorized you by running this script.",
        allowed_tools=["openai_computer"],  # Only allow the OpenAI computer tool
    )

    try:
        # Run the agent
        # query = "Find the hud-sdk repo on github and click on it"
        # print(f"\n🤖 Running: {query}\n")

        # Ask user for query in terminal
        query = input("Enter a query: ")

        # Use trace to see MCP calls in real-time
        with hud.trace():
            result = await agent.run(query, max_steps=15)

        print(f"\n✅ Result: {result}")

    finally:
        await client.close_all_sessions()


if __name__ == "__main__":
    print(f"🚀 Connecting to MCP router at {BASE_URL}")
    asyncio.run(main())
