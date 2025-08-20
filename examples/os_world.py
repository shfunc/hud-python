#!/usr/bin/env python3
"""
OSWorld Agent Example

Prerequisites:
- uv add hud-python
- Set HUD_API_KEY environment variable

Usage:
- Run single task: python sheet_bench.py
- Run entire dataset: python sheet_bench.py dataset
"""

import asyncio
import hud
from hud.agents import OperatorAgent
from hud.clients import MCPClient
from datasets import load_dataset
from hud.datasets import run_dataset, Task

import logging

logging.basicConfig(level=logging.INFO)


async def run_single_task():
    """Run a single task from OSWorld dataset."""
    # Load the dataset
    print("📊 Loading OSWorld dataset...")
    dataset = load_dataset("hud-evals/OSWorld-Gold-Beta", split="train")
    # dataset = load_dataset("hud-evals/OSWorld-Verified-XLang", split="train")

    with hud.trace("OSWorld Agent"):
        task = Task(**dataset[0])

        # Create client and agent
        client = MCPClient(mcp_config=task.mcp_config)
        agent = OperatorAgent(
            mcp_client=client,
            allowed_tools=["openai_computer"],
        )

        try:
            print(task.prompt)
            result = await agent.run(task, max_steps=40)
            print(result.reward)

        finally:
            print("\n🔚 Closing client...")
            await client.close()


async def run_osworld_dataset():
    """Run the entire OSWorld dataset using run_dataset."""
    # Load the dataset
    print("📊 Loading OSWorld dataset...")
    dataset = load_dataset("hud-evals/OSWorld-Gold-Beta", split="train")

    # Define agent configuration
    agent_config = {
        "allowed_tools": ["openai_computer"],
    }

    # Run the dataset
    print("🚀 Running OSWorld dataset evaluation...")
    results = await run_dataset(
        name="OSWorld-Gold-Beta Evaluation",
        dataset=dataset,
        agent_class=OperatorAgent,
        agent_config=agent_config,
        max_concurrent=50,
        metadata={
            "dataset": "OSWorld-Gold-Beta",
            "split": "train",
        },
        max_steps=150,
    )


async def main():
    """Main entry point - choose which function to run."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dataset":
        # Run the entire dataset
        await run_osworld_dataset()
    else:
        # Run a single task (default)
        await run_single_task()


if __name__ == "__main__":
    asyncio.run(main())
