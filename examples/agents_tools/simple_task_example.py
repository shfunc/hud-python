#!/usr/bin/env python3
"""
Example: Simplified Agent Task Interface

This example demonstrates the new agent.run(task) interface that can handle:
1. Simple string queries
2. Full Task objects with setup/evaluate lifecycle

Usage:
    # First, build and start the simple_browser environment:
    cd environments/simple_browser
    docker build -t hud-browser .

    # Then run this example:
    python examples/agents_tools/simple_task_example.py
"""

import asyncio
import logging
from hud.mcp import ClaudeMCPAgent
from hud.datasets import TaskConfig
from mcp.types import CallToolRequestParams as MCPToolCall
from mcp_use import MCPClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main():
    print("🚀 Simple Task Interface Example")
    print("=" * 50)

    # Configure MCP client to connect to simple_browser environment
    config = {
        "mcp_config": {
            "browser": {
                "command": "docker",
                "args": [
                    "run",
                    "--rm",
                    "-i",
                    "-p",
                    "8080:8080",  # VNC port
                    "-e",
                    "LAUNCH_APPS=todo",  # Launch todo app
                    "-e",
                    "BROWSER_URL=http://localhost:3000",  # Navigate to todo app
                    "hud-browser",
                ],
            }
        }
    }

    # Create MCP client and agent
    print("📡 Connecting to browser environment...")
    client = MCPClient.from_dict(config)

    agent = ClaudeMCPAgent(
        mcp_client=client,
        model="claude-sonnet-4-20250514",
        allowed_tools=["anthropic_computer", "api_request"],
        initial_screenshot=True,
    )

    try:
        print("✅ Agent created! Testing both query and task modes...\n")

        # Example 1: Simple Query (string)
        print("🔍 Example 1: Simple Query")
        print("-" * 30)

        simple_result = await agent.run("Take a screenshot and describe what you see on the page")
        print(f"📝 Query Result: {simple_result}\n")

        # Example 2: Full Task with Setup and Evaluate
        print("🎯 Example 2: Full Task Lifecycle")
        print("-" * 30)

        task = TaskConfig(
            prompt="Add a new todo item called 'Test automated task' and mark it as completed",
            setup_tool=MCPToolCall(name="todo_seed", arguments={"num_items": 2}),
            evaluate_tool=MCPToolCall(name="todo_completed", arguments={"expected_count": 1}),
        )

        print(f"📋 Task: {task.prompt}")
        print(f"⚙️  Setup: {task.setup_tool}")
        print(f"📊 Evaluate: {task.evaluate_tool}")

        eval_result = await agent.run(task)
        print(f"🎉 Task Result: {eval_result}")

        # Show formatted results
        reward = eval_result.get("reward", 0.0)
        success = reward > 0.5
        info = eval_result.get("info", {})

        print(f"\n📈 Task Performance:")
        print(f"   ✅ Success: {success}")
        print(f"   🏆 Reward: {reward}")
        print(f"   📝 Info: {info}")

        # Example 3: Task without evaluation (setup only)
        print("\n🔧 Example 3: Setup-Only Task")
        print("-" * 30)

        setup_task = TaskConfig(
            prompt="Take a screenshot and count how many todo items are visible",
            setup_tool=MCPToolCall(name="todo_seed", arguments={"num_items": 5}),
            # No evaluate - will return success automatically
        )

        setup_result = await agent.run(setup_task)
        print(f"🔧 Setup-only Result: {setup_result}")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Error: {e}")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await client.close_all_sessions()
        print("✅ Done!")


if __name__ == "__main__":
    print("🐳 Make sure you have built the browser environment:")
    print("   cd environments/simple_browser")
    print("   docker build -t hud-browser .")
    print("\nPress Enter to continue or Ctrl+C to cancel...")

    try:
        input()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
