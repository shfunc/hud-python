#!/usr/bin/env python3
"""Generic HuggingFace dataset evaluation runner.

This script lets you evaluate any HUD-compatible Task dataset with either
Claude or OpenAI (Operator) agents.

Prerequisites:
• `uv add hud-python`
• Set `HUD_API_KEY` in your env.
• Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your env.

Usage examples
──────────────
# Evaluate the FULL SheetBench dataset with Claude
python examples/run_evaluation.py hud-evals/SheetBench-50 --full --agent claude

# Run a single OSWorld task with OpenAI Operator agent (default single-task mode)
python examples/run_evaluation.py hud-evals/OSWorld-Gold-Beta --agent openai
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Literal

import hud
from datasets import load_dataset
from hud.agents import ClaudeAgent, OperatorAgent
from hud.clients import MCPClient
from hud.datasets import Task, run_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent factory helpers
# ---------------------------------------------------------------------------

def _build_agent(
    agent_type: Literal["claude", "openai"],
    mcp_client: MCPClient,
    *,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
) -> ClaudeAgent | OperatorAgent:
    """Create and return the requested agent type."""

    if agent_type == "openai":
        allowed_tools = allowed_tools or ["openai_computer"]

        return OperatorAgent(
            mcp_client=mcp_client,
            allowed_tools=allowed_tools,
        )

    # Fallback Claude agent (Anthropic)
    model = model or "claude-sonnet-4-20250514"
    allowed_tools = allowed_tools or ["anthropic_computer"]

    return ClaudeAgent(
        mcp_client=mcp_client,
        model=model,
        allowed_tools=allowed_tools,
    )


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

async def run_single_task(
    dataset_name: str,
    *,
    agent_type: Literal["claude", "openai"] = "claude",
    model: str | None = None,
    allowed_tools: list[str] | None = None,
) -> None:
    """Load *one* task from *dataset_name* and execute it."""

    print("📊 Loading dataset…")
    dataset = load_dataset(dataset_name, split="train")
    with hud.trace(name=dataset[0].get("prompt", f"Task {dataset[0]['id']}")):
        task = Task(**dataset[0])

        client = MCPClient(mcp_config=task.mcp_config)
        agent = _build_agent(
            agent_type,
            client,
            model=model,
            allowed_tools=allowed_tools,
        )
        try:
            print(task.prompt)
            result = await agent.run(task, max_steps=40)
            print("✅ Reward:", result.reward)
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Full-dataset runner
# ---------------------------------------------------------------------------

async def run_full_dataset(
    dataset_name: str,
    *,
    agent_type: Literal["claude", "openai"] = "claude",
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_concurrent: int = 50,
    max_steps: int = 150,
) -> list[Any]:
    """Run evaluation across the entire dataset using hud.datasets.run_dataset."""

    # Build agent class + config for run_dataset – we pass the *class* and a minimal
    # config dict, run_dataset will create a fresh agent per task.
    if agent_type == "openai":
        agent_class = OperatorAgent
        agent_config: dict[str, Any] = {
            "allowed_tools": allowed_tools or ["openai_computer"],
        }
    else:
        agent_class = ClaudeAgent
        agent_config = {
            "model": model or "claude-sonnet-4-20250514",
            "allowed_tools": allowed_tools or ["anthropic_computer"],
        }

    print("🚀 Running evaluation…")
    return await run_dataset(
        name=f"Evaluation {dataset_name.split('/')[-1]}",
        dataset=dataset_name,
        agent_class=agent_class,
        agent_config=agent_config,
        max_concurrent=max_concurrent,
        metadata={"dataset": dataset_name},
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # type: ignore[valid-type]
    parser = argparse.ArgumentParser(description="Generic HUD dataset evaluation runner")
    parser.add_argument(
        "dataset",
        help="HuggingFace dataset identifier, e.g. 'hud-evals/SheetBench-50'",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the entire dataset (omit for single-task debug mode)",
    )
    parser.add_argument(
        "--agent",
        choices=["claude", "openai"],
        default="claude",
        help="Agent backend to use",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Model name for the chosen agent",
    )
    parser.add_argument(
        "--allowed-tools",
        dest="allowed_tools",
        default=None,
        help="Comma-separated list of allowed tools (overrides defaults)",
    )
    parser.add_argument(
        "--max-concurrent",
        dest="max_concurrent",
        type=int,
        default=50,
        help="Concurrency level for full-dataset mode",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    allowed_tools = (
        [t.strip() for t in args.allowed_tools.split(",") if t.strip()]
        if args.allowed_tools
        else None
    )

    if args.full:
        await run_full_dataset(
            args.dataset,
            agent_type=args.agent,
            model=args.model,
            allowed_tools=allowed_tools,
            max_concurrent=args.max_concurrent,
        )
    else:
        await run_single_task(
            args.dataset,
            agent_type=args.agent,
            model=args.model,
            allowed_tools=allowed_tools,
        )


if __name__ == "__main__":
    asyncio.run(main())
