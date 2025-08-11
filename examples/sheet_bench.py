#!/usr/bin/env python3
"""
SheetBench Agent Example

Prerequisites:
- uv add hud-python
- Set HUD_API_KEY environment variable
"""

import asyncio
import hud
from hud.agents import ClaudeMCPAgent
from hud.clients import MCPClient
from datasets import load_dataset
from hud.datasets import to_taskconfigs


async def main():
    # Load the dataset
    print("📊 Loading SheetBench dataset...")
    dataset = load_dataset("hud-evals/SheetBench-50", split="train")

    with hud.trace("SheetBench Agent"):
        task = to_taskconfigs(dataset)[0]

        # Create client and agent
        client = MCPClient(mcp_config=task.mcp_config)
        agent = ClaudeMCPAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )

        try:
            print(task.prompt)
            result = await agent.run(task, max_steps=15)
            print(result.reward)

        finally:
            print("\n🔚 Closing client...")
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
