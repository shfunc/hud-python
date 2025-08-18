#!/usr/bin/env python3
"""
SheetBench Agent Example

Prerequisites:
- uv add hud-python
- Set HUD_API_KEY environment variable

Usage:
- Run single task: python sheet_bench.py
- Run entire dataset: python sheet_bench.py dataset
"""

import asyncio
import argparse
import hud
from hud.agents import ClaudeAgent, OperatorAgent
from hud.agents.misc import ResponseAgent
from hud.clients import MCPClient
from openai import AsyncOpenAI
from datasets import load_dataset
from hud.datasets import run_dataset, TaskConfig
from typing import Any

import logging
logging.basicConfig(level=logging.INFO)


async def run_single_task(agent_type: str = "claude", model: str | None = None, allowed_tools: list[str] | None = None) -> None:
    """Run a single task from the SheetBench dataset with the chosen agent."""
    # Load the dataset
    print("📊 Loading SheetBench dataset...")
    dataset = load_dataset("hud-evals/SheetBench-50", split="train")

    with hud.trace("SheetBench Agent"):
        task = TaskConfig(**dataset[0])

        # Create client and agent
        client = MCPClient(mcp_config=task.mcp_config)

        if agent_type == "openai":
            allowed_tools = allowed_tools or ["openai_computer"]
            openai_client = AsyncOpenAI()
            agent = OperatorAgent(
                mcp_client=client,
                allowed_tools=allowed_tools,
                response_agent=ResponseAgent(),
            )
        else:
            allowed_tools = allowed_tools or ["anthropic_computer"]
            agent = ClaudeAgent(
                mcp_client=client,
                model=model or "claude-sonnet-4-20250514",
                allowed_tools=allowed_tools,
            )

        try:
            print(task.prompt)
            result = await agent.run(task, max_steps=40)
            print(result.reward)
        finally:
            print("\n🔚 Closing client...")
            await client.close()


async def run_sheetbench_dataset(agent_type: str = "claude", model: str | None = None, allowed_tools: list[str] | None = None, max_concurrent: int = 50) -> list[Any]:
    """Run the entire SheetBench dataset using run_dataset with the chosen agent."""
    print("📊 Loading SheetBench dataset...")
    dataset = load_dataset("hud-evals/SheetBench-50", split="train")

    if agent_type == "openai":
        agent_class = OperatorAgent
        agent_config = {
            "allowed_tools": allowed_tools or ["openai_computer"],
        }
    else:
        agent_class = ClaudeAgent
        agent_config = {
            "model": model or "claude-sonnet-4-20250514",
            "allowed_tools": allowed_tools or ["anthropic_computer"],
        }

    print("🚀 Running SheetBench dataset evaluation...")
    results = await run_dataset(
        name="SheetBench-50 Evaluation",
        dataset=dataset,
        agent_class=agent_class,
        agent_config=agent_config,
        max_concurrent=max_concurrent,
        metadata={
            "dataset": "SheetBench-50",
            "split": "train",
        },
        max_steps=150,
        auto_respond=True,
    )

    # Process results (same as before)
    print("\n📈 Results Summary:")
    rewards = [r.reward for r in results if r is not None]
    if rewards:
        print(f"  Total tasks: {len(results)}")
        print(f"  Successful tasks: {len(rewards)}")
        print(f"  Average reward: {sum(rewards) / len(rewards):.2f}")
        print(f"  Min reward: {min(rewards):.2f}")
        print(f"  Max reward: {max(rewards):.2f}")
    else:
        print("  No successful tasks completed.")

    return results


async def main() -> None:
    """Parse CLI arguments and run the desired evaluation mode."""
    parser = argparse.ArgumentParser(description="SheetBench Agent Runner")
    parser.add_argument("mode", choices=["single", "dataset"], nargs="?", default="single", help="Run a single task or the entire dataset")
    parser.add_argument("--agent", choices=["claude", "openai"], default="claude", help="Agent backend to use")
    parser.add_argument("--model", dest="model", default=None, help="Model name to use for the chosen agent")
    parser.add_argument("--allowed-tools", dest="allowed_tools", default=None, help="Comma-separated list of allowed tools")
    parser.add_argument("--max-concurrent", dest="max_concurrent", type=int, default=50, help="Maximum concurrency for dataset mode")
    args = parser.parse_args()

    allowed_tools = [t.strip() for t in args.allowed_tools.split(",") if t.strip()] if args.allowed_tools else None

    if args.mode == "dataset":
        await run_sheetbench_dataset(
            agent_type=args.agent,
            model=args.model,
            allowed_tools=allowed_tools,
            max_concurrent=args.max_concurrent,
        )
    else:
        await run_single_task(agent_type=args.agent, model=args.model, allowed_tools=allowed_tools)


if __name__ == "__main__":
    asyncio.run(main())
