#!/usr/bin/env python3
"""
SheetBench Gold File Parallel Testing

Test the entire SheetBench dataset using gold files for evaluation,
running multiple tasks in parallel for speed.

Prerequisites:
- uv add hud-python
- Set HUD_API_KEY environment variable

Usage:
- python sheet_bench_test_gold_parallel.py
- python sheet_bench_test_gold_parallel.py --max-concurrent 10
"""

import asyncio
import json
import sys
import hud
from hud.clients import MCPClient
from datasets import load_dataset
from hud.datasets import Task
from typing import Any


async def test_gold_files_parallel(job_id: str, max_concurrent: int = 50):
    """Test all SheetBench tasks using gold files in parallel."""
    # Load the dataset
    print("📊 Loading SheetBench dataset...")
    dataset = load_dataset("hud-evals/SheetBench-50", split="train")
    
    print(f"📋 Found {len(dataset)} tasks to test")
    print(f"🚀 Running with max {max_concurrent} concurrent tasks\n")
    
    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrent)
    
    async def _worker(idx: int, task_dict: dict[str, Any]) -> None:
        async with sem:
            # Create trace for this task with explicit job_id
            with hud.trace(task_dict.get("prompt"), job_id=job_id, task_id=task_dict.get("id")):
                # Convert to Task
                task = Task(**task_dict)
                
                # Extract gold_file_url from metadata
                gold_file_url = task.metadata.get('gold_file_url')
                
                if not gold_file_url:
                    print(f"⚠️  Task {idx}: No gold_file_url found in metadata")
                    return
                
                # Create fresh MCP client per task
                client = MCPClient(mcp_config=task.mcp_config)
                
                try:
                    await client.initialize()
                    
                    # Run setup with gold file
                    await client.call_tool(
                        "setup",
                        {
                            "name": "sheets_from_xlsx",
                            "arguments": {
                                "file_url": gold_file_url
                            }
                        }
                    )

                    await client.call_tool(
                        "anthropic_computer",
                        {
                            "action": "screenshot"
                        }
                    )
                    
                    # Run evaluation directly (no agent actions)
                    if task.evaluate_tool:
                        await client.call_tool(
                            task.evaluate_tool.name, 
                            task.evaluate_tool.arguments
                        )
                    
                    await client.call_tool(
                        "anthropic_computer",
                        {
                            "action": "screenshot"
                        }
                    )
                except Exception as e:
                    print(f"❌ Task {idx}: {e}")
                    
                finally:
                    await client.close()
    
    # Execute all tasks
    await asyncio.gather(
        *[_worker(i, task_dict) for i, task_dict in enumerate(dataset)],
        return_exceptions=True
    )
    
    print("\n✅ Completed gold file evaluation run")


async def main():
    """Main entry point."""
    # Parse command line arguments
    max_concurrent = 50
    if len(sys.argv) > 1 and sys.argv[1] == "--max-concurrent":
        try:
            max_concurrent = int(sys.argv[2])
        except (IndexError, ValueError):
            print("Usage: python sheet_bench_test_gold_parallel.py [--max-concurrent N]")
            return
    
    with hud.job(
        "SheetBench Gold File Validation",
        metadata={
            "test_type": "gold_file_evaluation",
            "max_concurrent": max_concurrent,
        },
        dataset_link="hud-evals/SheetBench-50"
    ) as job_obj:
        await test_gold_files_parallel(job_obj.id, max_concurrent)


if __name__ == "__main__":
    asyncio.run(main())
