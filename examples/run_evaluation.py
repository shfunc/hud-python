#!/usr/bin/env python3
"""Generic HuggingFace dataset evaluation runner with parallel execution support.

This script lets you evaluate any HUD-compatible Task dataset with either
Claude or OpenAI (Operator) agents. Supports both asyncio-based concurrency
(for small datasets) and process-based parallelization (for large datasets).

Prerequisites:
• `uv add hud-python`
• Set `HUD_API_KEY` in your env.
• Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your env.

Usage examples
──────────────
# Run a single OSWorld-Verified task with OpenAI Operator agent (default single-task mode)
python examples/run_evaluation.py hud-evals/OSWorld-Verified-XLang --agent openai

# Evaluate the FULL SheetBench dataset with Claude (Single Process concurrency)
python examples/run_evaluation.py hud-evals/SheetBench-50 --full --agent claude --max-concurrent 25

# Run OSWorld-Verified dataset full with PARALLEL execution (300+ tasks)
python examples/run_evaluation.py hud-evals/OSWorld-Verified-XLang --agent openai --full --parallel

# Parallel mode with manual configuration (16 workers, 25 tasks each)
python examples/run_evaluation.py hud-evals/OSWorld-Verified-XLang --agent openai --full --parallel --max-workers 16 --tasks-per-worker 25
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Literal

import hud
from datasets import load_dataset
from hud.agents import ClaudeAgent, OperatorAgent
from hud.agents.misc.response_agent import ResponseAgent
from hud.clients import MCPClient
from hud.datasets import (
    Task, 
    run_dataset, 
    run_dataset_parallel, 
    run_dataset_parallel_auto
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent factory helpers
# ---------------------------------------------------------------------------


def _build_agent(
    agent_type: Literal["claude", "openai"],
    *,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
) -> ClaudeAgent | OperatorAgent:
    """Create and return the requested agent type."""

    if agent_type == "openai":
        allowed_tools = allowed_tools or ["openai_computer"]

        return OperatorAgent(
            allowed_tools=allowed_tools,
            response_agent=ResponseAgent(),
        )

    # Fallback Claude agent (Anthropic)
    model = model or "claude-sonnet-4-20250514"
    allowed_tools = allowed_tools or ["anthropic_computer"]

    return ClaudeAgent(
        model=model,
        allowed_tools=allowed_tools,
        response_agent=ResponseAgent(),
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

    # Get a simple task from dataset (Open last tab task)
    sample_task = dataset[1]  # type: ignore[index]
    task_prompt = sample_task.get("prompt", f"Task {sample_task.get('id', 0)}")  # type: ignore[attr-defined]

    with hud.trace(name=task_prompt):
        task = Task(**sample_task)  # type: ignore[arg-type]

        agent = _build_agent(
            agent_type,
            model=model,
            allowed_tools=allowed_tools,
        )
        print(task.prompt)
        result = await agent.run(task, max_steps=10)
        print("✅ Reward:", result.reward)


# ---------------------------------------------------------------------------
# Full-dataset runner
# ---------------------------------------------------------------------------


async def run_full_dataset(
    dataset_name: str,
    *,
    agent_type: Literal["claude", "openai"] = "claude",
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_concurrent: int = 30,
    max_steps: int = 50,
    parallel: bool = False,
    max_workers: int | None = None,
    tasks_per_worker: int = 25,
) -> list[Any]:
    """Run evaluation across the entire dataset.
    
    Uses either asyncio-based run_dataset or process-based run_dataset_parallel
    depending on the parallel flag.
    """

    # Build agent class + config for run_dataset – we pass the *class* and a minimal
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

    eval_name = f"Evaluation {dataset_name.split('/')[-1]}"
    
    if parallel:
        print(f"🚀 Running PARALLEL evaluation (workers: {max_workers or 'auto'})…")
        if max_workers is None:
            # Use auto-optimization
            return await run_dataset_parallel_auto(
                name=eval_name,
                dataset=dataset_name,
                agent_class=agent_class,
                agent_config=agent_config,
                metadata={"dataset": dataset_name, "parallel": True},
                max_steps=max_steps,
                auto_respond=True,
            )
        else:
            # Use manual configuration
            return await run_dataset_parallel(
                name=eval_name,
                dataset=dataset_name,
                agent_class=agent_class,
                agent_config=agent_config,
                max_workers=max_workers,
                tasks_per_worker=tasks_per_worker,
                max_concurrent_per_worker=10,  # Reasonable default
                metadata={"dataset": dataset_name, "parallel": True},
                max_steps=max_steps,
                auto_respond=True,
            )
    else:
        print(f"🚀 Running evaluation (max_concurrent: {max_concurrent})…")
        return await run_dataset(
            name=eval_name,
            dataset=dataset_name,
            agent_class=agent_class,
            agent_config=agent_config,
            max_concurrent=max_concurrent,
            metadata={"dataset": dataset_name},
            max_steps=max_steps,
            auto_respond=True,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # type: ignore[valid-type]
    parser = argparse.ArgumentParser(
        description="Evaluate HUD datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s hud-evals/SheetBench-50                    # Single task test
  %(prog)s hud-evals/SheetBench-50 --full             # Full dataset (<100 tasks)
  %(prog)s hud-evals/LargeDataset --full --parallel   # Large dataset (100+ tasks)
        """,
    )
    
    parser.add_argument("dataset", help="HuggingFace dataset ID")
    parser.add_argument("--full", action="store_true", help="Run entire dataset")
    
    # Agent
    parser.add_argument("--agent", choices=["claude", "openai"], default="claude")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--allowed-tools", dest="allowed_tools", help="Tool allowlist (comma-separated)")
    
    # Concurrency
    parser.add_argument("--max-concurrent", dest="max_concurrent", type=int, default=50, 
                        help="Max concurrent tasks (default: 50)")
    
    # Parallel mode (100+ tasks)
    parser.add_argument("--parallel", action="store_true", help="Use parallel execution for large datasets")
    parser.add_argument("--max-workers", dest="max_workers", type=int, help="Worker processes")
    parser.add_argument("--tasks-per-worker", dest="tasks_per_worker", type=int, default=25)
    
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    allowed_tools = (
        [t.strip() for t in args.allowed_tools.split(",") if t.strip()]
        if args.allowed_tools
        else None
    )

    if args.full:
        import time
        start_time = time.time()
        
        results = await run_full_dataset(
            args.dataset,
            agent_type=args.agent,
            model=args.model,
            allowed_tools=allowed_tools,
            max_concurrent=args.max_concurrent,
            parallel=args.parallel,
            max_workers=args.max_workers,
            tasks_per_worker=args.tasks_per_worker,
        )
        
        elapsed = time.time() - start_time
        
        # Print statistics
        print("\n" + "=" * 50)
        print("📊 Evaluation Complete!")
        print("=" * 50)
        print(f"Total tasks: {len(results)}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Throughput: {len(results)/elapsed:.2f} tasks/second")
        
        if args.parallel:
            print(f"Execution mode: PARALLEL (workers: {args.max_workers or 'auto'})")
        else:
            print(f"Execution mode: ASYNCIO (max_concurrent: {args.max_concurrent})")
        
        # Count successes
        successful = sum(1 for r in results if getattr(r, "reward", 0) > 0)
        print(f"Successful tasks: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
        
    else:
        await run_single_task(
            args.dataset,
            agent_type=args.agent,
            model=args.model,
            allowed_tools=allowed_tools,
        )


if __name__ == "__main__":
    asyncio.run(main())
