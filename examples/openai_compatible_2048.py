#!/usr/bin/env python3
"""
OpenAI Chat Agent playing Text 2048

This example demonstrates using the OpenAIChatAgent with the text-2048 environment.
It shows how to:
- Initialize an OpenAI client with the openai_chat agent
- Configure the text-2048 environment
- Run the agent to play the game

Requirements:
- pip install openai
- export OPENAI_API_KEY="your-api-key"  # Or set OPENAI_BASE_URL for custom endpoints

Environment Variables:
- OPENAI_BASE_URL: Custom OpenAI-compatible API endpoint
- OPENAI_API_KEY: API key for authentication
"""

import asyncio
import os
from openai import AsyncOpenAI
import hud
from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
from hud.clients import MCPClient
from hud.datasets import Task


async def main():
    # Initialize OpenAI client with environment variables
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    openai_client = AsyncOpenAI(
        base_url=base_url if base_url else None,  # None will use default OpenAI endpoint
        api_key=api_key,
    )

    mcp_config = {
        "local": {
            "command": "docker",
            "args": ["run", "--rm", "-i", "hudevals/hud-text-2048:latest"],
        }
    }

    system_prompt = """You are an expert 2048 game player. Your goal is to reach the tile specified by the user.

HOW 2048 WORKS:
- 4x4 grid with numbered tiles (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048...)
- When you move, all tiles slide in that direction
- When two tiles with SAME number touch, they merge into one (2+2=4, 4+4=8, etc.)
- After each move, a new tile (2 or 4) appears randomly
- Game ends when grid is full and no merges possible

CRITICAL RULES:
- ALWAYS analyze the board before moving
- ALWAYS make a tool call for your move
- Use the 'move' tool with these choices: "up", "down", "left", or "right"
- Remember: ALL strings in JSON must have quotes!
- Make exactly ONE move per turn
- NEVER ask for permission - just keep playing until the game ends
- Don't ask "Should I continue?" - just make your next move

Example tool call: {"name": "move", "arguments": {"direction": "right"}}"""

    # Define the task with game setup and evaluation
    task = Task(
        prompt="""Aim for the 128 tile (atleast a score of 800!)""",
        mcp_config=mcp_config,
        setup_tool={
            "name": "setup",
            "arguments": {"name": "board", "arguments": {"board_size": 4}},
        },  # type: ignore
        evaluate_tool={
            "name": "evaluate",
            "arguments": {"name": "max_number", "arguments": {"target": 128}},
        },  # type: ignore
    )

    # Initialize MCP client
    client = MCPClient(mcp_config=task.mcp_config)

    model_name = "gpt-5-mini"  # Replace with your model name

    # Create OpenAI agent with the text-2048 game tools
    agent = GenericOpenAIChatAgent(
        mcp_client=client,
        openai_client=openai_client,
        model_name=model_name,
        allowed_tools=["move"],
        parallel_tool_calls=False,
        system_prompt=system_prompt,
    )

    agent.metadata = {}

    with hud.job("OpenAI 2048 Game", metadata={"model": model_name}) as job:
        try:
            print("🎮 Starting 2048 game with OpenAI agent...")
            print(f"🤖 Model: {agent.model_name}")
            print("=" * 50)

            with hud.trace("Game Execution", job_id=job.id):
                result = await agent.run(task, max_steps=100)

            # Display results
            print("=" * 50)
            print(f"✅ Game completed!")
            print(f"🏆 Final Score/Max Tile: {result.reward}")
            if result.info:
                print(f"📊 Game Stats: {result.info}")

        except Exception as e:
            print(f"❌ Error during game: {e}")
        finally:
            await client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
