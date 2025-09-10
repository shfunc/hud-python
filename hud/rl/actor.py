"""Actor for episode collection using vLLM and HUD."""

import asyncio
import json
import logging
import random
from typing import List, Any
from pathlib import Path

import httpx
from openai import AsyncOpenAI

import hud
from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
from hud.datasets import Task

from .config import Config
from .types import Episode

logger = logging.getLogger(__name__)


class Actor:
    """Collects episodes using vLLM-served models via HUD agents."""
    
    def __init__(self, config: Config, tasks: List[Task], verbose: bool = False):
        self.config = config
        self.actor_config = config.actor
        self.tasks = self._validate_tasks(tasks)
        self.current_adapter = config.model.base_model
        self.verbose = verbose
        
        # Setup OpenAI client for vLLM
        base_url = self.actor_config.vllm_base_url.replace("localhost", "127.0.0.1")
        self.openai_client = self._create_openai_client(base_url)
    
    def _create_openai_client(self, base_url: str) -> AsyncOpenAI:
        """Create OpenAI client with optimized settings for vLLM."""
        # Match connection limits to parallel_episodes to avoid bottlenecks
        max_parallel = self.actor_config.parallel_episodes
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=30.0,  # Increased timeout for vLLM
                connect=5.0,
                read=25.0,
            ),
            limits=httpx.Limits(
                max_connections=max_parallel + 2,  # Allow some overhead
                max_keepalive_connections=max_parallel,
            ),
            http2=False,
        )
        
        return AsyncOpenAI(
            base_url=base_url,
            api_key=self.actor_config.vllm_api_key,
            http_client=http_client,
        )
    
    def _validate_tasks(self, tasks: List[Task]) -> List[Task]:
        """Validate that all tasks are proper HUD Task objects."""
        if not tasks:
            raise ValueError("No tasks provided to Actor")
        
        validated_tasks = []
        for i, task in enumerate(tasks):
            if not isinstance(task, Task):
                raise TypeError(f"Task at index {i} is not a HUD Task object, got {type(task)}")
            
            # Validate required fields
            if not task.prompt:
                raise ValueError(f"Task at index {i} (id={task.id}) has no prompt")
            if not task.mcp_config:
                raise ValueError(f"Task at index {i} (id={task.id}) has no mcp_config")
            
            validated_tasks.append(task)
        
        logger.info(f"[Actor] Validated {len(validated_tasks)} tasks")
        return validated_tasks
    
    def update_adapter(self, adapter_name: str):
        """Update the current adapter being used."""
        self.current_adapter = adapter_name
        logger.info(f"[Actor] Using adapter: {adapter_name}")
    
    async def collect_groups(self, group_size: int, n_groups: int, job_id: str) -> List[Episode]:
        """Collect episodes in groups from the same task for GRPO.
        
        Args:
            group_size: Number of episodes per task
            n_groups: Number of task groups to collect
            job_id: Job ID for tracking
            
        Returns:
            List of episodes grouped by task
        """
        episodes = []
        
        # Sample n_groups tasks
        sampled_tasks = random.choices(self.tasks, k=n_groups)
        
        logger.info(f"[Actor] Collecting {n_groups} groups of {group_size} episodes each")
        # Create all episode tasks with their group info
        all_tasks = []
        group_info = []  # Track which episodes belong to which group
        for i, task in enumerate(sampled_tasks):
            for j in range(group_size):
                all_tasks.append(self._run_episode(task, job_id))
                group_info.append((i, task.id))
        
        # Process episodes in batches respecting parallel_episodes limit
        batch_num = 0
        for batch_start in range(0, len(all_tasks), self.actor_config.parallel_episodes):
            batch_end = min(batch_start + self.actor_config.parallel_episodes, len(all_tasks))
            batch = all_tasks[batch_start:batch_end]
            batch_num += 1
            
            # Log batch info
            batch_groups = set(group_info[i][0] for i in range(batch_start, batch_end))
            logger.debug(f"[Actor] Processing batch {batch_num}: {len(batch)} episodes from groups {sorted(batch_groups)}")
            
            # Run batch in parallel
            batch_results = await asyncio.gather(*batch)
            episodes.extend(batch_results)
        
        # Log group-wise statistics
        logger.debug("\n[Actor] Group-wise statistics:")
        for i in range(n_groups):
            group_start = i * group_size
            group_end = (i + 1) * group_size
            group_episodes = episodes[group_start:group_end]
            successes = sum(1 for ep in group_episodes if ep.success)
            task_id = sampled_tasks[i].id
            logger.debug(f"  Group {i+1} (task {task_id}): {successes}/{group_size} successes")
        
        # Log overall statistics
        total_successes = sum(1 for ep in episodes if ep.success)
        logger.info(f"\n[Actor] Total collected: {len(episodes)} episodes, "
                   f"{total_successes} successes ({total_successes/len(episodes)*100:.1f}%)")
        
        return episodes
    
    async def _run_episode(self, task: Task, job_id: str) -> Episode:
        """Run a single episode with a task."""
        try:
            # Create agent with current adapter

            agent = GenericOpenAIChatAgent(
                openai_client=self.openai_client,
                model_name=self.current_adapter,
                allowed_tools=["computer"],
                append_setup_output=False,
                system_prompt=self.actor_config.system_prompt if not task.system_prompt else "",
                verbose=self.verbose,  # Pass through verbose flag
                completion_kwargs={
                    "temperature": self.actor_config.temperature,
                    "max_tokens": self.actor_config.max_tokens,
                    "tool_choice": "required",  # Removed to allow flexible tool use
                }
            )
            
            # Run the episode
            with hud.trace(f"Training {task.id or 'unknown'}", job_id=job_id):
                result = await agent.run(
                    task,
                    max_steps=self.actor_config.max_steps_per_episode
                )
            
            # Extract conversation history
            conversation_history = agent.conversation_history if hasattr(agent, 'conversation_history') else []

            return Episode(
                task_id=task.id or "unknown",
                terminal_reward=float(result.reward),
                conversation_history=conversation_history,
                tool_spec=agent.mcp_schemas,
                info=result.info if hasattr(result, 'info') else {},
                metadata={
                    "adapter": self.current_adapter,
                    "steps": len(conversation_history),
                }
            )
            
        except Exception as e:
            logger.error(f"[Actor] Episode failed: {e}")
            # Print full traceback for debugging
            import traceback
            traceback.print_exc()
            
            return Episode(
                task_id=task.id or "unknown",
                terminal_reward=0.0,
                conversation_history=[],
                tool_spec=[],
                info={"error": str(e)},
                metadata={"adapter": self.current_adapter}
            )

# Standalone actor worker mode
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", choices=["test", "worker"])
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--tasks-file", type=str, default="browser_2048_tasks.jsonl")
    args = parser.parse_args()
    
    config = Config()
    
    # Load tasks for test mode
    tasks = []
    if Path(args.tasks_file).exists():
        with open(args.tasks_file) as f:
            for line in f:
                item = json.loads(line.strip())
                task = Task(
                    id=item.get("id"),
                    prompt=item["prompt"],
                    mcp_config=item["mcp_config"],
                    setup_tool=item.get("setup_tool"),
                    evaluate_tool=item.get("evaluate_tool"),
                    system_prompt=item.get("system_prompt", config.actor.system_prompt),
                    metadata=item.get("metadata", {})
                )
                tasks.append(task)
    else:
        logger.error(f"Tasks file not found: {args.tasks_file}")
        exit(1)
    
    actor = Actor(config, tasks)
    
    if args.mode == "test":
        # Test collection
        episodes = asyncio.run(actor.collect_groups(group_size=args.n_episodes, n_groups=1, job_id="test-job"))
        logger.info(f"[Actor] Collected {len(episodes)} episodes")
        for ep in episodes:
            logger.info(f"  Task {ep.task_id}: reward={ep.terminal_reward:.2f}, steps={ep.num_steps}")
    else:
        # Worker mode would connect to a queue
        logger.error("Worker mode not implemented yet")