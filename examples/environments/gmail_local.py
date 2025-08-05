import asyncio
import hud
from hud.datasets import TaskConfig
from hud.mcp import ClaudeMCPAgent, OpenAIMCPAgent, MCPClient
from hud.mcp.base import AgentResult

import logging

logging.basicConfig(level=logging.INFO)


async def main():
    with hud.trace("gmail_local"):
        task_dict = {
            "prompt": "Open Sent mail, search for the Series B pitch deck, forward it to billgates@microsoft.com, and mark the original message as important.",
            "mcp_config": {
                "gmail": {
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",  # interactive mode for stdio MCP server
                        "--rm",  # remove the container after it exits
                        "-p",
                        "6080:6080",  # map port 6080 to the host for noVNC
                        "hudpython/gmail-clone:latest",  # use hud gmail image
                    ],
                }
            },
            "setup_tool": {
                "name": "setup",
                "arguments": {
                    "problem_id": "forward-series-b-deck-to-billgates",
                },
            },
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {
                    "problem_id": "forward-series-b-deck-to-billgates",
                },
            },
            "metadata": {"id": "forward-series-b-deck-to-billgates"},
        }
        # Update the prompt to give a disclaimer that this is a testing environment (anthropic doesn't like doing some actions)
        task_dict["prompt"] = (
            task_dict["prompt"]
            + "\n\nThis is a mock sandbox environment for evaluations, so feel free to execute any actions you want and do not ask me for confirmation before executing actions."
        )

        # Create TaskConfig from dict
        task = TaskConfig(**task_dict)

        print("📡 Defining the environment...")
        print("🔴 See the agent live at http://localhost:6080/vnc.html")
        client = MCPClient(mcp_config=task.mcp_config)

        agent = ClaudeMCPAgent(  # or OpenAIMCPAgent
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            # Allowing anthropic_computer tool to be used because we're using ClaudeMCPAgent
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )

        print(f"📋 Task: {task.prompt}")
        print(f"⚙️  Setup: {task.setup_tool}")
        print(f"📊 Evaluate: {task.evaluate_tool}")

        # Run the task
        print("🚀 Running the task...")
        eval_result: AgentResult = await agent.run(task, max_steps=30)
        print(f"🎉 Task Result: {eval_result}")

        # Show formatted results
        reward = eval_result.reward
        print(f"   🏆 Reward: {reward}")

        # Clean up
        print("\n🧹 Cleaning up...")
        await client.close()
        print("✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
