import asyncio
import hud
from hud.mcp import ClaudeMCPAgent, OpenAIMCPAgent
from hud.datasets import TaskConfig
from mcp_use import MCPClient


async def main():
    with hud.trace("gmail_remote"):
        # Define task configuration as dict with environment variable templates
        task_dict = {
            "prompt": "Open Sent mail, search for the Series B pitch deck, forward it to billgates@microsoft.com, and mark the original message as important.",
            "mcp_config": {
                "hud": {
                    "url": "${HUD_MCP_URL}",
                    "headers": {
                        "Authorization": "Bearer ${HUD_API_KEY}",
                        "Mcp-Image": "hudpython/gmail-clone:latest",
                        "Run-Id": "${RUN_ID}",  # Automatically filled from trace context
                    },
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

        # Create TaskConfig from dict - env vars are automatically substituted
        task = TaskConfig(**task_dict)

        print("📡 Defining the environment...")
        print("🔴 See the agent live at http://localhost:6080/vnc.html")

        # Create MCP client from resolved servers
        client = MCPClient.from_dict({"mcp_config": task.mcp_config})

        agent = ClaudeMCPAgent(  # or OpenAIMCPAgent
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["computer"],
            initial_screenshot=True,
        )

        print(f"📋 Task: {task.prompt}")
        print(f"⚙️  Setup: {task.setup_tool}")
        print(f"📊 Evaluate: {task.evaluate_tool}")

        # Run the task
        print("🚀 Running the task...")
        eval_result = await agent.run(task, max_steps=10)
        print(f"🎉 Task Result: {eval_result}")

        # Show formatted results
        reward = eval_result.get("reward", 0.0)
        print(f"   🏆 Reward: {reward}")

        # Clean up
        print("\n🧹 Cleaning up...")
        await client.close_all_sessions()
        print("✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
