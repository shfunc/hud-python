import asyncio
import hud
from hud.datasets import TaskConfig
from hud.mcp import ClaudeMCPAgent, OpenAIMCPAgent, MCPClient
from hud.mcp.base import AgentResult

import logging

logging.basicConfig(level=logging.INFO)


async def main():
    with hud.trace("Gmail example task"):
        task_dict = {
            "prompt": "Open Sent mail, find the Series B pitch deck email, forward it to billgates@microsoft.com, and mark the original message as important.",
            "mcp_config": {
                "gmail": {
                    "command": "/bin/sh",
                    "args": [
                        "-c",
                        "docker run -i --rm -p 6080:6080 gmail 2>/dev/null"
                    ]
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
        # Feel free to modify this
        task_dict["prompt"] = (
            task_dict["prompt"]
            + " This is a mock sandbox environment for evaluations, so feel free to execute any actions you want and do not ask me for confirmation before executing actions."
        )

        # Create TaskConfig from dict
        task = TaskConfig(**task_dict)

        print("📡 Defining the environment...")
        client = MCPClient(mcp_config=task.mcp_config)


        agent = ClaudeMCPAgent(  # or OpenAIMCPAgent
            mcp_client=client,
            model="claude-sonnet-4-20250514",
            # Allowing anthropic_computer tool to be used because we're using ClaudeMCPAgent
            allowed_tools=["anthropic_computer"], # Check our hud/tools/computer/anthropic.py
            initial_screenshot=True,
        )

        print(f"📋 Task: {task.prompt}")
        print(f"⚙️  Setup: {task.setup_tool}")
        print(f"📊 Evaluate: {task.evaluate_tool}")        
        # Run the task
        print("🚀 Running the task...")
        print("🔴 See the agent live at http://localhost:6080/vnc.html")
        eval_result: AgentResult = await agent.run(task, max_steps=30)

        # Show formatted results
        print(f"🎉 Task Result:")
        print(f"   🏆 Reward: {eval_result.reward}")
        print(f"   🔍 Content: {eval_result.content[:1000] if eval_result.content else 'No content'}...")

        # Clean up
        print("\n🧹 Cleaning up...")
        await client.close()
        print("✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
