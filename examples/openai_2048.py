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

from hud.agents.misc import ResponseAgent


async def main():
    # Initialize OpenAI client with environment variables
    base_url = os.getenv("OPENAI_BASE_URL")  # Optional custom endpoint
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")  # Default to "EMPTY" for local servers
    
    openai_client = AsyncOpenAI(
        base_url=base_url,  # None will use default OpenAI endpoint
        api_key=api_key,
    )
    
    # Configure the text-2048 environment
    mcp_config = {
        "local": {
            "command": "docker",
            "args": ["run", "--rm", "-i", "hudevals/hud-text-2048:latest"]
        }
    }
    
    # Define the task with game setup and evaluation
    task = Task(
        prompt="""Play the 2048 game strategically. 
        
        Tips for high scores:
        - Keep your highest tile in a corner (preferably bottom-right)
        - Build tiles in descending order from that corner
        - Avoid moving up unless absolutely necessary
        - Try to keep tiles of similar values adjacent
        
        Use the 'move' tool with directions: up, down, left, or right.
        Aim for the highest possible score!""",
        mcp_config=mcp_config,
        setup_tool={"name": "setup","arguments": {"name": "board", "arguments": {"board_size": 4}},}, # type: ignore
        evaluate_tool={"name": "evaluate", "arguments": {"name": "max_number", "arguments": {}}}, # type: ignore
    )

    # Initialize MCP client
    client = MCPClient(mcp_config=task.mcp_config)
    
    # Create OpenAI agent with the text-2048 game tools
    agent = GenericOpenAIChatAgent(
        mcp_client=client,
        openai_client=openai_client,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        allowed_tools=["move"],
        parallel_tool_calls=False,
        response_agent=ResponseAgent(),
        system_prompt="""You are an expert 2048 game player. 
        Make strategic moves to achieve the highest score possible.
        Always analyze the board state before making a move.""",
    )

    agent.metadata = {}

    # Run the game with tracing
    with hud.trace("OpenAI 2048 Game"):
        try:
            print("🎮 Starting 2048 game with OpenAI agent...")
            print(f"🤖 Model: {agent.model_name}")
            print("="*50)
            
            # Run the task with unlimited steps (game ends when no moves available)
            result = await agent.run(task, max_steps=-1)
            
            # Display results
            print("="*50)
            print(f"✅ Game completed!")
            print(f"🏆 Final Score/Max Tile: {result.reward}")
            if result.info:
                print(f"📊 Game Stats: {result.info}")

            # Display conversation history
            print("🗣️ Conversation History:")
            for i, msg in enumerate(agent.conversation_history):
                print(f"  {i+1} : {msg}")
                print("-"*30)

        except Exception as e:
            print(f"❌ Error during game: {e}")
        finally:
            await client.shutdown()


if __name__ == "__main__":
    # Check for API configuration
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        print("⚠️  Please configure OpenAI API access:")
        print("   For OpenAI API: export OPENAI_API_KEY='your-api-key'")
        print("   For local/custom endpoints: export OPENAI_BASE_URL='your-custom-endpoint'")
        exit(1)
    
    # Display configuration
    if os.getenv("OPENAI_BASE_URL"):
        print(f"🔗 Using endpoint: {os.getenv('OPENAI_BASE_URL')}")
    else:
        print("🔗 Using default OpenAI API endpoint")
    
    asyncio.run(main())