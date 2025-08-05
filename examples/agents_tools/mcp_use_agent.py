#!/usr/bin/env python3
"""MCP-Use agent example for HUD tools via HTTP."""

import asyncio
from dotenv import load_dotenv
import hud
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from mcp_use import MCPAgent, MCPClient

load_dotenv()

# Configuration
BASE_URL = "http://localhost:8039/mcp"


async def main():
    """Run MCP-Use agent with HUD tools."""

    # Configure MCP client to connect to the router
    config = {"mcp_config": {"hud": {"url": BASE_URL}}}

    # Create client and agent
    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(
        llm=llm,
        mcp_client=client,
        max_steps=30,
        verbose=True,
        disallowed_tools=["edit", "bash", "computer_anthropic", "computer_openai"],
    )

    try:
        # Run the agent
        query = "Click on the chat in the bottom right corner, and type 'Hello, how are you?'"
        print(f"\n🤖 Running: {query}\n")

        # Use trace_debug to see MCP calls in real-time
        with hud.trace():
            result = await agent.run(query)
        print(f"\n✅ Result: {result}")

    finally:
        await client.close_all_sessions()


if __name__ == "__main__":
    print(f"🚀 Connecting to MCP router at {BASE_URL}")
    asyncio.run(main())
