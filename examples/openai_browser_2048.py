#!/usr/bin/env python3
"""
OpenAI Chat Agent playing Browser 2048

This example demonstrates using the OpenAIChatAgent with the browser-based 2048 game.
It shows how to:
- Initialize an OpenAI client with browser automation capabilities
- Configure the browser-2048 environment with Docker
- Use computer vision and interaction tools to play the game

Requirements:
- pip install openai
- export OPENAI_API_KEY="your-api-key"  # Or set OPENAI_BASE_URL for custom endpoints
- Docker installed and running

Environment Variables:
- OPENAI_BASE_URL: Custom OpenAI-compatible API endpoint (optional)
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
        base_url=base_url if base_url else None,
        api_key=api_key,
    )

    # Configure the browser-2048 environment
    mcp_config = {
        "local": {
            "command": "docker",
            "args": ["run", "--rm", "-i", "-p", "8080:8080", "hudevals/hud-browser:0.1.3"],
        }
    }

    system_prompt = """You are an expert 2048 game player using a browser interface. Your goal is to reach the tile specified by the user.

HOW 2048 WORKS:
- 4x4 grid with numbered tiles (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048...)
- When you move, all tiles slide in that direction
- When two tiles with SAME number touch, they merge into one (2+2=4, 4+4=8, etc.)
- After each move, a new tile (2 or 4) appears randomly
- Game ends when grid is full and no merges possible

BROWSER INTERACTION USING THE COMPUTER TOOL:
1. FIRST TURN ONLY - TAKE SCREENSHOT:
   Use: computer(action="screenshot")
   This captures the initial game state. Only needed for your first turn.
   After that, the environment will automatically return an image with each successful move.

2. MAKE MOVES - Use arrow keys by calling the computer tool with action="press":
   - Move UP: computer(action="press", keys=["up"])
   - Move DOWN: computer(action="press", keys=["down"]) 
   - Move LEFT: computer(action="press", keys=["left"])
   - Move RIGHT: computer(action="press", keys=["right"])

CRITICAL RULES:
- Make exactly ONE move per turn using the press action with arrow keys
- Continue playing until you reach the target or the game ends, no need to ask the user for confirmation.

Strategy tips:
- Keep your highest tiles in a corner
- Build tiles in descending order from the corner
- Avoid random moves - be strategic
- Try to keep the board organized"""

    # Define the task with browser game setup and evaluation
    task = Task(
        prompt="""Play the browser-based 2048 game and try to reach the 128 tile.

        Start by taking a screenshot to see the initial game board, then make strategic moves using arrow keys.
        After your first screenshot, the game board will be automatically shown after each successful move.""",
        mcp_config=mcp_config,
        setup_tool={"name": "launch_app", "arguments": {"app_name": "2048"}},  # type: ignore
        evaluate_tool={
            "name": "evaluate",
            "arguments": {"name": "game_2048_max_number", "arguments": {"target": 128}},
        },  # type: ignore
    )

    # Initialize MCP client
    client = MCPClient(mcp_config=task.mcp_config)

    model_name = "z-ai/glm-4.5v"  # "z-ai/glm-4.5v", "Qwen/Qwen2.5-VL-7B-Instruct" etc...

    # Create OpenAI agent with browser automation tools
    agent = GenericOpenAIChatAgent(
        mcp_client=client,
        openai_client=openai_client,
        model_name=model_name,
        allowed_tools=["computer"],
        parallel_tool_calls=False,
        system_prompt=system_prompt,
    )

    agent.metadata = {}

    # Run the game with tracing
    with hud.trace("OpenAI Browser 2048 Game"):
        try:
            print("🎮 Starting browser-based 2048 game with OpenAI agent...")
            print(f"🤖 Model: {agent.model_name}")
            print(f"🌐 Browser environment running on localhost:8080")
            print("=" * 50)

            result = await agent.run(task, max_steps=100)

            # Display results
            print("=" * 50)
            print(f"✅ Game completed!")
            print(f"🏆 Final Score/Max Tile: {result.reward}")
            if result.info:
                print(f"📊 Game Stats: {result.info}")

            print("\n📝 Full interaction trace:")
            for i, msg in enumerate(agent.conversation_history):
                print(f"  {i + 1} : {msg}")
                print("-" * 30)

        except Exception as e:
            print(f"❌ Error during game: {e}")
        finally:
            await client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
