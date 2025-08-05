"""Dataset utilities for working with HuggingFace datasets and TaskConfigs."""

from __future__ import annotations

import asyncio
import logging
from string import Template
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolRequestParams as MCPToolParams
from pydantic import BaseModel, Field, field_validator

from hud.telemetry.job import job

if TYPE_CHECKING:
    from datasets import Dataset

    from hud.mcp.base import AgentResult, BaseMCPAgent

logger = logging.getLogger("hud.datasets")


class TaskConfig(BaseModel):
    """
    A task configuration that can be used to create a task.

    The mcp_config field supports environment variable substitution using
    template placeholders in the format ${VAR_NAME} or ${VAR_NAME:default_value}.

    Example:
        mcp_config: {
            "hud": {
                "url": "${HUD_MCP_URL:https://mcp.hud.so/v3/mcp}",
                "headers": {
                    "Authorization": "Bearer ${HUD_API_KEY}",
                    "Run-Id": "${RUN_ID}",
                    "Mcp-Image": "your-mcp-image"
                }
            }
        }
    """

    id: str | None = None
    prompt: str
    mcp_config: dict[str, Any]
    setup_tool: MCPToolParams | None = None
    evaluate_tool: MCPToolParams | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mcp_config", mode="before")
    @classmethod
    def resolve_env_vars(cls, v: dict[str, Any]) -> dict[str, Any]:
        """
        Automatically resolve environment variables in mcp_config using Template.

        Supports ${VAR_NAME} syntax with variable substitution from:
        1. System environment variables (including HUD_API_KEY, etc.)
        2. Runtime context variables (e.g., RUN_ID from telemetry context)

        Missing variables resolve to empty strings.
        """
        import os

        from hud.telemetry.context import get_current_task_run_id

        # Start with current environment variables
        mapping = dict(os.environ)

        # Add runtime context variables if available
        run_id = get_current_task_run_id()
        if run_id:
            mapping["RUN_ID"] = run_id

        def substitute_in_value(obj: Any) -> Any:
            """Recursively substitute variables in nested structures."""
            if isinstance(obj, str):
                # Use Template's substitute with defaultdict - missing vars become empty strings
                from collections import defaultdict

                safe_mapping = defaultdict(str, mapping)
                return Template(obj).substitute(safe_mapping)
            elif isinstance(obj, dict):
                return {k: substitute_in_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_in_value(item) for item in obj]
            else:
                return obj

        return substitute_in_value(v)


def to_taskconfigs(dataset: Dataset) -> Dataset:
    """
    Convert a HuggingFace dataset to contain TaskConfig objects.

    Args:
        dataset: HuggingFace dataset with task data

    Returns:
        Dataset with 'task' column containing TaskConfig objects

    Example:
        >>> dataset = load_dataset("hud/sheetbench-v1", split="test")
        >>> tasks = to_taskconfigs(dataset)
        >>> tasks[0]["task"]  # This is a TaskConfig object
    """

    def _convert(example: dict[str, Any]) -> dict[str, TaskConfig]:
        return {"task": TaskConfig(**example)}

    # Map and keep only the task column
    return dataset.map(_convert, remove_columns=dataset.column_names)


async def run_dataset(
    name: str,
    dataset: Dataset,
    agent_class: type[BaseMCPAgent],
    agent_config: dict[str, Any] | None = None,
    max_concurrent: int = 5,
    metadata: dict[str, Any] | None = None,
) -> list[Any]:
    """
    Run all tasks in a dataset with automatic job tracking.

    Args:
        name: Name for the job
        dataset: HuggingFace Dataset (raw, not converted)
        agent_class: Agent class to instantiate (e.g., ClaudeMCPAgent)
        agent_config: Configuration for agent (model, etc.)
        max_concurrent: Maximum parallel task execution
        metadata: Optional metadata for the job

    Returns:
        List of results from agent.run() in dataset order

    Example:
        >>> from datasets import load_dataset
        >>> from hud.mcp import ClaudeMCPAgent
        >>> dataset = load_dataset("hud/sheetbench-v1", split="test")
        >>> results = await run_dataset(
        ...     "sheetbench_eval",
        ...     dataset,
        ...     ClaudeMCPAgent,
        ...     {"model": "claude-3-5-sonnet-20241022"},
        ...     max_concurrent=3,
        ... )
    """
    # Import here to avoid circular imports
    import hud
    from hud.mcp.client import MCPClient

    # Convert dataset to TaskConfigs internally
    tasks = to_taskconfigs(dataset)

    # Create job context
    job_metadata = metadata or {}
    job_metadata["agent_class"] = agent_class.__name__
    if agent_config:
        job_metadata["agent_config"] = agent_config

    with job(name, metadata=job_metadata):
        # Run tasks with semaphore for concurrency control
        sem = asyncio.Semaphore(max_concurrent)
        results: list[AgentResult | None] = [None] * len(tasks)

        async def _worker(index: int, row: Any) -> None:
            async with sem:
                task = row["task"]

                # Create trace for this task
                with hud.trace(f"task_{index}"):
                    # Create fresh MCP client per task
                    if task.mcp_config:
                        client = MCPClient(mcp_config=task.mcp_config)
                        agent = agent_class(mcp_client=client, **(agent_config or {}))

                        try:
                            results[index] = await agent.run(task)
                        finally:
                            await client.close()
                    else:
                        logger.warning("Task %d has no mcp_config defined", index)
                        results[index] = None

        # Execute all tasks
        await asyncio.gather(
            *[_worker(i, row) for i, row in enumerate(tasks)],
            return_exceptions=True,  # Don't fail entire batch on one error
        )

    return results
