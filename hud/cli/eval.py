"""HUD evaluation command for running tasks and datasets."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal
from statistics import mean, stdev

import typer

import hud
from hud.settings import settings
from hud.utils.hud_console import HUDConsole
from hud.types import Trace

logger = logging.getLogger(__name__)
hud_console = HUDConsole()


def get_available_models() -> list[dict[str, str | None]]:
    """Fetch available models from the HUD API (only ready models).
    
    Returns:
        List of dicts with 'name' and 'vllm_url' keys
    """
    try:
        from hud.cli.rl import rl_api
        
        hud_console.info("Fetching your models from https://app.hud.so/models")
        models = rl_api.list_models()
        
        # Filter for ready models only and sort by recency
        ready_models = [m for m in models if m.status == "ready"]
        ready_models.sort(key=lambda m: m.created_at or "", reverse=True)
        
        # Count other statuses for informational purposes
        training_count = sum(1 for m in models if m.status == "training")
        other_count = len(models) - len(ready_models) - training_count
        
        if ready_models:
            hud_console.success(f"Found {len(ready_models)} ready models:")
            for model in ready_models:
                vllm_status = " (vLLM deployed)" if model.vllm_url else ""
                hud_console.info(f"  ✅ {model.name}{vllm_status}")
            
            if training_count > 0:
                hud_console.info(f"\n({training_count} models currently training)")
            
            return [{"name": model.name, "vllm_url": model.vllm_url} for model in ready_models]
        else:
            if training_count > 0:
                hud_console.warning(f"No ready models found. You have {training_count} models currently training.")
            else:
                hud_console.warning("No models found in your account.")
            return []
    except Exception as e:
        logger.debug(f"Error fetching models: {e}")
        # Don't show the error to the user, just proceed without HUD models
        return []


class GroupedEvaluator:
    """Evaluator that runs tasks in groups, following the RL pattern."""
    
    def __init__(
        self,
        agent_class: type | Any,
        agent_config: dict[str, Any] | None = None,
        group_size: int = 1,
        max_parallel_episodes: int = 48,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        self.agent_class = agent_class
        self.agent_config = agent_config or {}
        self.group_size = group_size
        self.max_parallel_episodes = max_parallel_episodes
        self.max_steps = max_steps
        self.verbose = verbose
    
    async def run_tasks_grouped(self, tasks: list[Any], job_id: str | None = None) -> list[dict[str, Any]]:
        """Run tasks with grouping, following the RL Actor pattern."""
        from hud.datasets import Task
        
        # Duplicate tasks according to group_size, exactly like RL
        grouped_tasks = []
        task_mapping = []  # Track which group each result belongs to
        
        for i, task in enumerate(tasks):
            for _ in range(self.group_size):
                grouped_tasks.append(task)
                task_mapping.append(i)
        
        hud_console.info(f"Running {len(tasks)} tasks with group_size={self.group_size} ({len(grouped_tasks)} total runs)")
        
        # Run all episodes, respecting max_parallel_episodes
        all_traces = []
        
        for batch_start in range(0, len(grouped_tasks), self.max_parallel_episodes):
            batch_end = min(batch_start + self.max_parallel_episodes, len(grouped_tasks))
            batch = grouped_tasks[batch_start:batch_end]
            
            # Run batch in parallel
            async def run_single_episode(task_data: dict[str, Any] | Task, idx: int) -> Trace:
                """Run a single episode."""
                try:
                    # Create task if needed
                    if isinstance(task_data, dict):
                        task = Task(**task_data)
                    else:
                        task = task_data
                    
                    # Create fresh agent instance
                    if isinstance(self.agent_class, type):
                        agent = self.agent_class(**self.agent_config)
                    else:
                        # Agent is already instantiated
                        agent = self.agent_class
                    
                    # Run the task
                    trace_name = f"Eval | {task.id if hasattr(task, 'id') else 'Task'} | Group {task_mapping[idx]}"
                    with hud.trace(trace_name, job_id=job_id):
                        result = await agent.run(task, max_steps=self.max_steps)
                        return result
                        
                except Exception as e:
                    hud_console.warning_log(f"Episode failed: {e}")
                    return Trace(isError=True, content=str(e), reward=0.0, done=True)
            
            # Run batch
            batch_results = await asyncio.gather(
                *[run_single_episode(t, batch_start + i) for i, t in enumerate(batch)],
                return_exceptions=True,
            )
            
            # Normalize exceptions to error traces
            for res in batch_results:
                if isinstance(res, Exception):
                    hud_console.warning_log(f"Episode error: {res}")
                    all_traces.append(Trace(isError=True, content=str(res), reward=0.0, done=True))
                else:
                    all_traces.append(res)
            
            if self.verbose:
                hud_console.info(f"Completed batch: {len(all_traces)}/{len(grouped_tasks)} episodes")
        
        # Group results back by original task and calculate statistics
        return self._calculate_group_statistics(tasks, all_traces, task_mapping)
    
    def _calculate_group_statistics(
        self, 
        original_tasks: list[Any], 
        traces: list[Trace], 
        task_mapping: list[int]
    ) -> list[dict[str, Any]]:
        """Calculate statistics for each group, similar to preprocess_advantages."""
        from hud.datasets import Task
        import numpy as np
        
        stats = []
        
        # Process each original task
        for task_idx, task in enumerate(original_tasks):
            # Get all traces for this task
            task_traces = [
                traces[i] for i, mapping_idx in enumerate(task_mapping) 
                if mapping_idx == task_idx
            ]
            
            # Extract rewards
            rewards = np.array([t.reward for t in task_traces])
            errors = [t for t in task_traces if t.isError]
            
            # Calculate statistics
            task_stats = {
                "task_id": task.id if isinstance(task, Task) and hasattr(task, "id") else f"task_{task_idx}",
                "prompt": task.prompt if isinstance(task, Task) else task.get("prompt", ""),
                "group_size": self.group_size,
                "rewards": rewards.tolist(),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "success_rate": float(np.sum(rewards > 0) / len(rewards)) if len(rewards) > 0 else 0.0,
                "error_rate": len(errors) / len(task_traces) if len(task_traces) > 0 else 0.0,
                "traces": task_traces,  # Keep full traces for detailed analysis
            }
            
            # Add variance info like RL does
            if task_stats["std_reward"] > 1e-6:
                task_stats["normalized_rewards"] = [
                    (r - task_stats["mean_reward"]) / task_stats["std_reward"] 
                    for r in rewards
                ]
            else:
                task_stats["normalized_rewards"] = [0.0] * len(rewards)
            
            stats.append(task_stats)
        
        return stats


def display_group_statistics(stats: list[dict[str, Any]], show_details: bool = True) -> None:
    """Display statistics from grouped evaluation."""
    from rich.table import Table
    from rich.console import Console
    console = Console()
    
    # Overall statistics
    all_means = [s["mean_reward"] for s in stats]
    overall_mean = mean(all_means) if all_means else 0.0
    overall_std = stdev(all_means) if len(all_means) > 1 else 0.0
    
    hud_console.success("\n📊 Evaluation Summary")
    hud_console.info(f"Tasks evaluated: {len(stats)}")
    hud_console.info(f"Episodes per task: {stats[0]['group_size'] if stats else 0}")
    hud_console.info(f"Total episodes: {sum(len(s['rewards']) for s in stats)}")
    hud_console.info(f"Overall mean reward: {overall_mean:.3f} ± {overall_std:.3f}")
    
    # Detailed table
    if show_details and len(stats) <= 20:  # Only show for reasonable dataset sizes
        table = Table(title="\nPer-Task Performance Distribution")
        table.add_column("Task", style="cyan", no_wrap=True)
        table.add_column("Mean±Std", justify="right", style="green")
        table.add_column("Min/Max", justify="right")
        table.add_column("Success%", justify="right", style="yellow")
        table.add_column("Rewards", style="dim")
        
        for stat in stats:
            task_name = stat["prompt"][:30] + "..." if len(stat["prompt"]) > 30 else stat["prompt"]
            rewards_str = " ".join([f"{r:.2f}" for r in stat["rewards"][:5]])
            if len(stat["rewards"]) > 5:
                rewards_str += " ..."
                
            table.add_row(
                task_name,
                f"{stat['mean_reward']:.3f}±{stat['std_reward']:.3f}",
                f"{stat['min_reward']:.2f}/{stat['max_reward']:.2f}",
                f"{stat['success_rate']*100:.0f}%",
                rewards_str,
            )
        
        console.print(table)
    
    # High variance tasks
    high_variance_tasks = [s for s in stats if s["std_reward"] > 0.3 and s["group_size"] > 1]
    if high_variance_tasks:
        hud_console.warning(f"\n⚠️  {len(high_variance_tasks)} tasks show high variance (std > 0.3)")
        for task in high_variance_tasks[:3]:
            hud_console.info(f"  • {task['task_id']}: μ={task['mean_reward']:.3f}, σ={task['std_reward']:.3f}")


def build_agent(
    agent_type: Literal["claude", "openai", "vllm"],
    *,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    verbose: bool = False,
    vllm_base_url: str | None = None,
) -> Any:
    """Create and return the requested agent type."""

    # Import agents lazily to avoid dependency issues
    if agent_type == "vllm":
        # Create a generic OpenAI agent for vLLM server
        try:
            from openai import AsyncOpenAI

            from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
        except ImportError as e:
            hud_console.error(
                "OpenAI dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e
        
        # Determine the base URL to use
        if vllm_base_url is not None:
            # Use the provided vLLM URL (for custom/local servers)
            base_url = str(vllm_base_url)
            hud_console.info(f"Using vLLM server at {base_url}")
            api_key = settings.api_key if base_url.startswith(settings.hud_rl_url) else "token-abc123"
        elif model:
            # Always use standard HUD vLLM endpoint for HUD models
            base_url = f"{settings.hud_rl_url}/models/{model}/vllm"
            api_key = settings.api_key
            hud_console.info(f"Using HUD vLLM endpoint: {base_url}")
        else:
            # Default to localhost
            base_url = "http://localhost:8000/v1"
            api_key = "token-abc123"
        
        # Create OpenAI client for vLLM
        openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        
        return GenericOpenAIChatAgent(
            openai_client=openai_client,
            model_name=model or "served-model",  # Default model name
            verbose=verbose,
            completion_kwargs={
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        )
    
    elif agent_type == "openai":
        try:
            from hud.agents import OperatorAgent
        except ImportError as e:
            hud_console.error(
                "OpenAI agent dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        if allowed_tools:
            return OperatorAgent(
                allowed_tools=allowed_tools,
                verbose=verbose,
            )
        else:
            return OperatorAgent(verbose=verbose)

    # Fallback Claude agent (Anthropic)
    try:
        from hud.agents import ClaudeAgent
    except ImportError as e:
        hud_console.error(
            "Claude agent dependencies are not installed. "
            "Please install with: pip install 'hud-python[agent]'"
        )
        raise typer.Exit(1) from e

    model = model or "claude-sonnet-4-20250514"

    if allowed_tools:
        return ClaudeAgent(
            model=model,
            allowed_tools=allowed_tools,
            verbose=verbose,
        )
    else:
        return ClaudeAgent(
            model=model,
            verbose=verbose,
        )


async def run_single_task(
    source: str,
    *,
    agent_type: Literal["claude", "openai", "vllm"] = "claude",
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_steps: int = 10,
    verbose: bool = False,
    vllm_base_url: str | None = None,
    group_size: int = 1,
) -> None:
    """Load one task and execute it, or detect if JSON contains a list and run as dataset."""

    # Import Task and run_dataset lazily
    try:
        from hud.datasets import run_dataset
        from hud.utils.tasks import load_tasks
    except ImportError as e:
        hud_console.error(
            "Dataset dependencies are not installed. "
            "Please install with: pip install 'hud-python\u27e6agent\u27e7'"
        )
        raise typer.Exit(1) from e

    # Check if it's a file
    path = Path(source)
    if path.exists() and (path.suffix in [".json", ".jsonl"]):
        hud_console.info("📊 Loading task file…")
        
        # Use unified loader for both JSON and JSONL
        tasks = load_tasks(str(path))
        
        # Check if we have multiple tasks
        if len(tasks) > 1:
            hud_console.info(f"Found {len(tasks)} tasks in file, running as dataset…")

            # Build agent class and config for run_dataset
            if agent_type == "vllm":
                try:
                    from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
                    agent_class = GenericOpenAIChatAgent
                except ImportError as e:
                    hud_console.error(
                        "OpenAI dependencies are not installed. "
                        "Please install with: pip install 'hud-python\u27E6agent\u27E7'"
                    )
                    raise typer.Exit(1) from e

                # Use build_agent to create a sample agent to get the config
                sample_agent = build_agent(
                    agent_type,
                    model=model,
                    allowed_tools=allowed_tools,
                    verbose=verbose,
                    vllm_base_url=vllm_base_url,
                )
                
                # Extract the config from the sample agent
                agent_config: dict[str, Any] = {
                    "openai_client": sample_agent.oai,
                    "model_name": sample_agent.model_name,
                    "verbose": verbose,
                    "completion_kwargs": sample_agent.completion_kwargs,
                }
                if allowed_tools:
                    agent_config["allowed_tools"] = allowed_tools

            elif agent_type == "openai":
                try:
                    from hud.agents import OperatorAgent

                    agent_class = OperatorAgent
                except ImportError as e:
                    hud_console.error(
                        "OpenAI agent dependencies are not installed. "
                        "Please install with: pip install 'hud-python\u27e6agent\u27e7'"
                    )
                    raise typer.Exit(1) from e

                agent_config = {"verbose": verbose}
                if allowed_tools:
                    agent_config["allowed_tools"] = allowed_tools

            else:
                try:
                    from hud.agents import ClaudeAgent

                    agent_class = ClaudeAgent
                except ImportError as e:
                    hud_console.error(
                        "Claude agent dependencies are not installed. "
                        "Please install with: pip install 'hud-python[agent]'"
                    )
                    raise typer.Exit(1) from e

                agent_config = {
                    "model": model or "claude-sonnet-4-20250514",
                    "verbose": verbose,
                }
                if allowed_tools:
                    agent_config["allowed_tools"] = allowed_tools

            # Convert Task objects to dicts for run_dataset
            task_dicts = [task.model_dump() for task in tasks]
            
            # Run as dataset with single-task concurrency to maintain debug behavior
            results = await run_dataset(
                name=f"Dataset: {path.name}",
                dataset=task_dicts,  # Pass the list of task dicts
                agent_class=agent_class,
                agent_config=agent_config,
                max_concurrent=1,  # Run sequentially for debug mode
                metadata={"source": str(path)},
                max_steps=max_steps,
            )

            # Display summary
            successful = sum(1 for r in results if getattr(r, "reward", 0) > 0)
            hud_console.success(f"Completed {len(results)} tasks: {successful} successful")
            return

        # Single task - use the first (and only) task
        task = tasks[0]
        hud_console.info("Found 1 task, running as single task…")
    else:
        # Load from HuggingFace dataset or non-file source
        hud_console.info(f"📊 Loading tasks from: {source}…")
        tasks = load_tasks(source)
        
        if not tasks:
            hud_console.error(f"No tasks found in: {source}")
            raise typer.Exit(1)
            
        # Single task - use the first task
        task = tasks[0]
        hud_console.info("Using first task from dataset (run with --full to run the entire dataset)...")

    task_prompt = task.prompt[:50] + "..." if len(task.prompt) > 50 else task.prompt

    # Use grouped evaluation if group_size > 1
    if group_size > 1:
        hud_console.info(f"🔄 Running task with group_size={group_size}")
        
        # Build agent configuration
        if agent_type == "vllm":
            # Special handling for vLLM
            sample_agent = build_agent(
                agent_type,
                model=model,
                allowed_tools=allowed_tools,
                verbose=verbose,
                vllm_base_url=vllm_base_url,
            )
            agent_config = {
                "openai_client": sample_agent.oai,
                "model_name": sample_agent.model_name,
                "verbose": verbose,
                "completion_kwargs": sample_agent.completion_kwargs,
            }
            if allowed_tools:
                agent_config["allowed_tools"] = allowed_tools
            
            from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
            agent_class = GenericOpenAIChatAgent
        elif agent_type == "openai":
            from hud.agents import OperatorAgent
            agent_class = OperatorAgent
            agent_config = {"verbose": verbose}
            if allowed_tools:
                agent_config["allowed_tools"] = allowed_tools
        else:
            from hud.agents import ClaudeAgent
            agent_class = ClaudeAgent
            agent_config = {
                "model": model or "claude-sonnet-4-20250514",
                "verbose": verbose,
            }
            if allowed_tools:
                agent_config["allowed_tools"] = allowed_tools
        
        # Create grouped evaluator
        evaluator = GroupedEvaluator(
            agent_class=agent_class,
            agent_config=agent_config,
            group_size=group_size,
            max_parallel_episodes=48,  # Same as RL default
            max_steps=max_steps,
            verbose=verbose,
        )
        
        # Run with grouping
        with hud.trace(name=f"{task_prompt} (group_size={group_size})"):
            stats = await evaluator.run_tasks_grouped([task])
            
        # Display results
        display_group_statistics(stats, show_details=True)
        
    else:
        # Original single-run logic
        with hud.trace(name=task_prompt):
            agent = build_agent(
                agent_type,
                model=model,
                allowed_tools=allowed_tools,
                verbose=verbose,
                vllm_base_url=vllm_base_url,
            )
            hud_console.info(task.prompt)
            result = await agent.run(task, max_steps=max_steps)
            hud_console.success(f"Reward: {result.reward}")


async def run_full_dataset(
    source: str,
    *,
    agent_type: Literal["claude", "openai", "vllm"] = "claude",
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_concurrent: int = 50,
    max_steps: int = 10,
    parallel: bool = False,
    max_workers: int | None = None,
    max_concurrent_per_worker: int = 25,
    verbose: bool = False,
    vllm_base_url: str | None = None,
    group_size: int = 1,
) -> list[Any]:
    """Run evaluation across the entire dataset.

    Uses either asyncio-based run_dataset or process-based parallel execution
    depending on the parallel flag."""

    # Import run_dataset lazily
    try:
        from hud.datasets import run_dataset, run_dataset_parallel, run_dataset_parallel_manual
        from hud.utils.tasks import load_tasks
    except ImportError as e:
        hud_console.error(
            "Dataset dependencies are not installed. "
            "Please install with: pip install 'hud-python[agent]'"
        )
        raise typer.Exit(1) from e

    # Load tasks using unified loader
    hud_console.info(f"📊 Loading tasks from: {source}…")
    tasks = load_tasks(source)
    
    if not tasks:
        hud_console.error(f"No tasks found in: {source}")
        raise typer.Exit(1)
    
    # Convert Task objects to dicts for dataset runners
    dataset_or_tasks = [task.model_dump() for task in tasks]
    
    # Determine dataset name
    path = Path(source)
    if path.exists():
        dataset_name = f"Dataset: {path.name}"
    else:
        dataset_name = source.split("/")[-1]
    
    hud_console.info(f"Found {len(tasks)} tasks")

    # Build agent class + config for run_dataset
    if agent_type == "vllm":
        try:
            from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
            agent_class = GenericOpenAIChatAgent
        except ImportError as e:
            hud_console.error(
                "OpenAI dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        # Use build_agent to create a sample agent to get the config
        sample_agent = build_agent(
            agent_type,
            model=model,
            allowed_tools=allowed_tools,
            verbose=verbose,
            vllm_base_url=vllm_base_url,
        )
        
        # Extract the config from the sample agent
        agent_config: dict[str, Any] = {
            "openai_client": sample_agent.oai,
            "model_name": sample_agent.model_name,
            "verbose": verbose,
            "completion_kwargs": sample_agent.completion_kwargs,
        }
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools

    elif agent_type == "openai":
        try:
            from hud.agents import OperatorAgent

            agent_class = OperatorAgent
        except ImportError as e:
            hud_console.error(
                "OpenAI agent dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        agent_config = {"verbose": verbose}
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools

    else:
        try:
            from hud.agents import ClaudeAgent

            agent_class = ClaudeAgent
        except ImportError as e:
            hud_console.error(
                "Claude agent dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        agent_config = {
            "model": model or "claude-sonnet-4-20250514",
            "verbose": verbose,
        }
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools

    # Use grouped evaluation if group_size > 1
    if group_size > 1:
        hud_console.info(f"🔄 Running dataset with group_size={group_size}")
        
        # Create grouped evaluator
        evaluator = GroupedEvaluator(
            agent_class=agent_class,
            agent_config=agent_config,
            group_size=group_size,
            max_parallel_episodes=max_concurrent if not parallel else max_concurrent_per_worker * (max_workers or 4),
            max_steps=max_steps,
            verbose=verbose,
        )
        
        # Run with job tracking
        with hud.job(
            name=f"Evaluation {dataset_name} (group_size={group_size})",
            metadata={
                "dataset": source,
                "group_size": group_size,
                "tasks": len(dataset_or_tasks),
                "total_episodes": len(dataset_or_tasks) * group_size,
            }
        ) as job:
            # Convert dicts to Task objects if needed
            from hud.datasets import Task
            tasks = []
            for item in dataset_or_tasks:
                if isinstance(item, dict):
                    tasks.append(Task(**item))
                else:
                    tasks.append(item)
            
            stats = await evaluator.run_tasks_grouped(tasks, job_id=job.id)
        
        # Display results
        display_group_statistics(stats, show_details=len(stats) <= 20)
        
        # Return stats for consistency with other modes
        return stats
        
    # Original logic for non-grouped evaluation
    elif parallel:
        hud_console.info(
            f"🚀 Running PARALLEL evaluation (workers: {max_workers or 'auto'}, max_concurrent: {max_concurrent})…"  # noqa: E501
        )
        if max_workers is None:
            # Use auto-optimization (now the default run_dataset_parallel)
            return await run_dataset_parallel(
                name=f"Evaluation {dataset_name}",
                dataset=dataset_or_tasks,
                agent_class=agent_class,
                agent_config=agent_config,
                max_concurrent=max_concurrent,
                metadata={"dataset": source, "parallel": True},
                max_steps=max_steps,
                auto_respond=True,
            )
        else:
            # Use manual configuration
            return await run_dataset_parallel_manual(
                name=f"Evaluation {dataset_name}",
                dataset=dataset_or_tasks,
                agent_class=agent_class,
                agent_config=agent_config,
                max_workers=max_workers,
                max_concurrent_per_worker=max_concurrent_per_worker,
                max_concurrent=max_concurrent,
                metadata={"dataset": source, "parallel": True},
                max_steps=max_steps,
                auto_respond=True,
            )
    else:
        hud_console.info(f"🚀 Running evaluation (max_concurrent: {max_concurrent})…")
        return await run_dataset(
            name=f"Evaluation {dataset_name}",
            dataset=dataset_or_tasks,
            agent_class=agent_class,
            agent_config=agent_config,
            max_concurrent=max_concurrent,
            metadata={"dataset": source},
            max_steps=max_steps,
        )


def eval_command(
    source: str = typer.Argument(
        ...,
        help="HuggingFace dataset identifier (e.g. 'hud-evals/SheetBench-50'), JSON file (array of tasks), or JSONL file (one task per line)",  # noqa: E501
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run the entire dataset (omit for single-task debug mode)",
    ),
    agent: Literal["claude", "openai", "vllm"] = typer.Option(
        "claude",
        "--agent",
        help="Agent backend to use (claude, openai, or vllm for local server)",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model name for the chosen agent",
    ),
    allowed_tools: str | None = typer.Option(
        None,
        "--allowed-tools",
        help="Comma-separated list of allowed tools",
    ),
    max_concurrent: int = typer.Option(
        50,
        "--max-concurrent",
        help="Concurrency level for asyncio mode (ignored in parallel mode)",
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        help="Maximum steps per task (default: 10 for single, 50 for full)",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Use process-based parallel execution for large datasets (100+ tasks)",
    ),
    max_workers: int | None = typer.Option(
        None,
        "--max-workers",
        help="Number of worker processes for parallel mode (auto-optimized if not set)",
    ),
    max_concurrent_per_worker: int = typer.Option(
        20,
        "--max-concurrent-per-worker",
        help="Maximum concurrent tasks per worker in parallel mode",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output from the agent",
    ),
    vllm_base_url: str | None = typer.Option(
        None,
        "--vllm-base-url",
        help="Base URL for vLLM server (when using --agent vllm)",
    ),
    group_size: int = typer.Option(
        1,
        "--group-size",
        help="Number of times to run each task (similar to RL training)",
    ),
) -> None:
    """🚀 Run evaluation on datasets or individual tasks with agents.

    Examples:
        # Evaluate a single task from SheetBench
        hud eval hud-evals/SheetBench-50

        # Evaluate the FULL SheetBench dataset with Claude (asyncio mode)
        hud eval hud-evals/SheetBench-50 --full --agent claude

        # Run large dataset with PARALLEL execution (auto-optimized)
        hud eval hud-evals/OSWorld-Verified-XLang --full --parallel

        # Parallel mode with manual configuration (16 workers, 25 tasks each)
        hud eval hud-evals/OSWorld-Verified-XLang --full --parallel --max-workers 16

        # Limit total concurrent tasks to prevent rate limits
        hud eval hud-evals/SheetBench-50 --full --parallel --max-concurrent 20

        # Run a single task from a JSON file
        hud eval task.json

        # Run multiple tasks from a JSON file with parallel execution
        hud eval tasks.json --full --parallel

        # Run with OpenAI Operator agent
        hud eval hud-evals/OSWorld-Gold-Beta --agent openai

        # Use local vLLM server (default: localhost:8000)
        hud eval task.json --agent vllm --model Qwen/Qwen2.5-VL-3B-Instruct

        # Use custom vLLM server URL
        hud eval task.json --agent vllm --vllm-base-url http://192.168.1.100:8000/v1

        # Run with verbose output for debugging
        hud eval task.json --verbose
    """
    from hud.settings import settings

    # Check for required API keys
    if agent == "claude":
        if not settings.anthropic_api_key:
            hud_console.error("ANTHROPIC_API_KEY is required for Claude agent")
            hud_console.info(
                "Set it in your environment or .env file: ANTHROPIC_API_KEY=your-key-here"
            )
            raise typer.Exit(1)
    elif agent == "openai" and not settings.openai_api_key:
        hud_console.error("OPENAI_API_KEY is required for OpenAI agent")
        hud_console.info("Set it in your environment or .env file: OPENAI_API_KEY=your-key-here")
        raise typer.Exit(1)
    elif agent == "vllm":
        if model:
            hud_console.info(f"Using vLLM with model: {model}")
        else:
            # Default to served-model if no model specified
            model = "served-model"
            hud_console.info("Using vLLM with default model: served-model")

    # Check for HUD_API_KEY if using HUD services
    if not settings.api_key:
        hud_console.warning("HUD_API_KEY not set. Some features may be limited.")
        hud_console.info("Get your API key at: https://app.hud.so")

    # Parse allowed tools
    allowed_tools_list = (
        [t.strip() for t in allowed_tools.split(",") if t.strip()] if allowed_tools else None
    )

    # Set default max_steps if not provided
    if max_steps is None:
        max_steps = 50 if full else 10

    # Run evaluation
    if full:
        asyncio.run(
            run_full_dataset(
                source,
                agent_type=agent,
                model=model,
                allowed_tools=allowed_tools_list,
                max_concurrent=max_concurrent,
                max_steps=max_steps,
                parallel=parallel,
                max_workers=max_workers,
                max_concurrent_per_worker=max_concurrent_per_worker,
                verbose=verbose,
                vllm_base_url=vllm_base_url,
                group_size=group_size,
            )
        )
    else:
        asyncio.run(
            run_single_task(
                source,
                agent_type=agent,
                model=model,
                allowed_tools=allowed_tools_list,
                max_steps=max_steps,
                verbose=verbose,
                vllm_base_url=vllm_base_url,
                group_size=group_size,
            )
        )
