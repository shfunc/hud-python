import asyncio
import hud
from hud.mcp import ClaudeMCPAgent, OpenAIMCPAgent, MCPClient
from hud.datasets import TaskConfig
from hud.mcp.base import AgentResult


async def main():
    with hud.trace("gmail_remote"):
        # Define task configuration as dict with environment variable templates
        task_dict = {
            "prompt": "Open Sent mail, find the Series B pitch deck email, forward it to billgates@microsoft.com, and mark the original message as important.",
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
        print(f"📊 Evaluate: {task.evaluate_tool}")        # Run the task
        print("🚀 Running the task...")
        eval_result: AgentResult = await agent.run(task, max_steps=10)
        print(f"🎉 Task Result: {eval_result}")

        # Show formatted results
        reward = eval_result.reward
        print(f"   🏆 Reward: {reward}")
        print(f"   🔍 Content: {eval_result.content[:1000] if eval_result.content else 'No content'}...")
        print(f"   🔍 Messages: {eval_result.messages}")

        # Clean up
        print("\n🧹 Cleaning up...")
        await client.close()
        print("✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
