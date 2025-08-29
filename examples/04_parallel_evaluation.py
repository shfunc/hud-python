#!/usr/bin/env python
"""
Example of using parallel dataset execution for large-scale evaluations.

This example demonstrates how to run 400+ tasks efficiently using process-based
parallelism, bypassing Python's GIL limitations.
"""

import asyncio
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run a large-scale parallel evaluation."""
    
    # Import the parallel functions
    from hud.datasets_parallel import (
        run_dataset_parallel,
        run_dataset_parallel_auto,
        calculate_optimal_workers
    )
    from hud.agents import ClaudeAgent
    
    # Example 1: Manual configuration for maximum control
    logger.info("=" * 60)
    logger.info("Example 1: Manual configuration")
    logger.info("=" * 60)
    
    # Create a large test dataset
    test_tasks = []
    for i in range(100):  # 100 tasks for demo
        test_tasks.append({
            "id": f"task_{i:03d}",
            "prompt": f"Task {i}: Solve a simple problem",
            "mcp_config": {
                "test_env": {
                    "command": "echo",
                    "args": [f"Processing task {i}"]
                }
            },
            "metadata": {
                "batch": i // 25,  # Group into batches of 25
                "difficulty": "easy" if i < 50 else "hard"
            }
        })
    
    # Calculate optimal configuration (simple: 1 worker per CPU core)
    num_workers, tasks_per_worker = calculate_optimal_workers(len(test_tasks))
    logger.info(
        f"Optimal configuration for {len(test_tasks)} tasks: "
        f"{num_workers} workers × {tasks_per_worker} tasks/worker"
    )
    
    # Run with manual configuration
    start_time = datetime.now()
    results = await run_dataset_parallel(
        name="Manual Parallel Evaluation",
        dataset=test_tasks,
        agent_class=ClaudeAgent,
        agent_config={"model": "claude-3-haiku-20240307"},  # Use fast model for demo
        max_workers=4,  # Use 4 processes
        tasks_per_worker=25,  # 25 tasks per process
        max_concurrent_per_worker=10,  # 10 concurrent tasks within each worker
        metadata={
            "experiment": "parallel_test",
            "timestamp": start_time.isoformat()
        },
        max_steps=10
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Analyze results
    successful = sum(1 for r in results if not getattr(r, "isError", False))
    logger.info(f"Completed in {elapsed:.2f} seconds")
    logger.info(f"Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    logger.info(f"Average time per task: {elapsed/len(results):.3f} seconds")
    
    # Example 2: Automatic configuration
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Automatic configuration")
    logger.info("=" * 60)
    
    # Create an even larger dataset
    large_tasks = []
    for i in range(400):  # 400 tasks - would fail with regular run_dataset!
        large_tasks.append({
            "id": f"large_task_{i:04d}",
            "prompt": f"Large scale task {i}",
            "mcp_config": {
                "test_env": {
                    "command": "echo",
                    "args": [f"Task {i}"]
                }
            }
        })
    
    logger.info(f"Running {len(large_tasks)} tasks with automatic optimization...")
    
    start_time = datetime.now()
    results = await run_dataset_parallel_auto(
        name="Auto-Optimized Large Scale Evaluation",
        dataset=large_tasks,
        agent_class=ClaudeAgent,
        agent_config={"model": "claude-3-haiku-20240307"},
        metadata={
            "scale": "large",
            "auto_optimized": True
        },
        max_steps=5  # Fewer steps for speed
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Completed {len(results)} tasks in {elapsed:.2f} seconds")
    logger.info(f"Throughput: {len(results)/elapsed:.2f} tasks/second")
    
    # Example 3: Using with HuggingFace datasets
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: HuggingFace dataset (configuration only)")
    logger.info("=" * 60)
    
    # Show how it would work with a real dataset
    logger.info("Example configuration for HuggingFace datasets:")
    logger.info("""
    results = await run_dataset_parallel_auto(
        name="SheetBench Parallel Evaluation",
        dataset="hud-evals/SheetBench-50",
        agent_class=ClaudeAgent,
        agent_config={"model": "claude-3-5-sonnet-20241022"},
        max_steps=40
    )
    """)
    
    # Show scaling recommendations
    logger.info("\n" + "=" * 60)
    logger.info("Scaling Recommendations")
    logger.info("=" * 60)
    
    recommendations = [
        (50, "Use regular run_dataset (asyncio is sufficient)"),
        (100, "Consider run_dataset_parallel with 4 workers"),
        (200, "Use run_dataset_parallel with 8 workers"),
        (400, "Use run_dataset_parallel with 16 workers"),
        (1000, "Use run_dataset_parallel_auto for optimal configuration"),
        (5000, "Consider distributed solutions (Ray, Kubernetes)")
    ]
    
    for num_tasks, recommendation in recommendations:
        workers, per_worker = calculate_optimal_workers(num_tasks)
        logger.info(
            f"{num_tasks:5d} tasks: {workers:2d} workers × {per_worker:3d} tasks/worker - {recommendation}"
        )


if __name__ == "__main__":
    # Check environment
    if not os.getenv("HUD_API_KEY"):
        logger.warning("HUD_API_KEY not set - telemetry will be disabled")
        logger.info("Set HUD_API_KEY to enable telemetry tracking")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set - agent calls will fail")
        logger.info("Please set ANTHROPIC_API_KEY to run this example")
        exit(1)
    
    # Run the examples
    asyncio.run(main())
