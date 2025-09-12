"""Main training loop for GRPO RL."""

import os
# Force training to use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
import logging
import uuid
from pathlib import Path

import hud
from hud.datasets import Task
from hud.rl.actor import Actor
from hud.rl.buffer import DatasetBuffer, ReplayBuffer
from hud.rl.config import Config
from hud.rl.learner import GRPOLearner
from hud.rl.utils import ensure_dir, load_tasks, set_seed
from hud.rl.vllm_adapter import VLLMAdapter
from hud.utils.design import HUDDesign

design = HUDDesign(logging.getLogger(__name__))


async def train(config: Config, tasks: list[Task]) -> None:
    """Main training loop."""
    # Initialize components
    set_seed(config.seed)
    ensure_dir(config.out_dir)
    if config.verbose:
        logging.basicConfig(level=logging.INFO)
        # Remove httpx logger
        logging.getLogger("httpx").setLevel(logging.WARNING)

    design.header("Starting GRPO Training")
    design.section_title("\n[1/3] Initializing components...")

    # Actor is responsible for running tasks and collecting episodes
    actor = Actor(config)

    # Learner is responsible for updating the policy
    learner = GRPOLearner(config)

    # Dataset buffer is responsible for storing tasks
    dataset_buffer = DatasetBuffer(
        tasks,
        config
    )
    design.key_value_table(dataset_buffer.info)

    # Replay buffer is responsible for storing episodes for training
    trace_buffer = ReplayBuffer(config)

    # VLLM adapter is responsible for loading and unloading adapters
    vllm = VLLMAdapter(
        config.actor.vllm_base_url,
        config.actor.vllm_api_key
    )
    
    # Training state
    step = 0
    
    design.section_title("\n[2/3] Running training loop...")
    
    with hud.job(name=config.job_name, metadata={"config": config.to_dict()}) as job_obj:
        while len(dataset_buffer) > 0:
            design.section_title(f"Step {step + 1}/{dataset_buffer.training_steps}")
            design.info(f"{len(dataset_buffer)} tasks remaining")

            # Get batch of tasks
            tasks = dataset_buffer.get_tasks()

            # Run tasks and collect traces
            traces = await actor.run_tasks(tasks, job_id=job_obj.id)
            design.info(f"Sampled {len(traces)} traces")
            trace_buffer.add(traces)

            design.info(f"Buffer has {len(trace_buffer)} traces to train from, selecting with {config.training.select_strategy} strategy")

            # Get batch of traces
            traces = trace_buffer.sample_traces()
            design.info(f"Selected {len(traces)} traces")
            
            design.section_title(f"Training on {len(traces)} traces")
            metrics = learner.update(traces)
            
            if step % config.stats_interval == 0:
                design.key_value_table(metrics.to_dict())
            
            # Save checkpoint and update vLLM
            step += 1
            if step % config.training.save_every_batches == 0:
                design.section_title("Saving checkpoint and updating vLLM")
                checkpoint_id = uuid.uuid4()
                checkpoint_path = Path(config.out_dir) / f"{config.adapter_prefix}-{checkpoint_id}"
                learner.save(str(checkpoint_path))

                adapter_name = f"{config.adapter_prefix}-{checkpoint_id}"
                if vllm.load_adapter(adapter_name, str(checkpoint_path)):
                    actor.update_adapter(adapter_name)
                    design.info(f"✓ Checkpoint saved and loaded: {adapter_name}")
                else:
                    design.warning(f"Failed to hot-load adapter {adapter_name}")
    
    design.section_title("\n[3/3] Training completed!")


async def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO RL Training")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    # Task input arguments
    parser.add_argument("--tasks", type=str, help="Path to tasks JSONL file or HuggingFace dataset name")
    parser.add_argument("--tasks-json", type=json.loads, help="Tasks as JSON list string")
    
    # Override config values
    parser.add_argument("--episodes-per-batch", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--out-dir", type=str)
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    # Apply test mode settings
    if args.test:
        design.info("[TEST MODE] Using minimal configuration")
        eps = 6
        config.training.episodes_per_batch = eps
        config.actor.parallel_episodes = eps
        config.training.group_size = eps
        config.training.mini_batch_size = 3
        config.training.max_training_steps = 4
        config.actor.max_steps_per_episode = 4

    # Calculate the memory usage
    INITIAL_MEMORY = 8.0
    SCALING_FACTOR = 5
    constant = config.training.mini_batch_size * config.training.max_training_steps
    quadratic = (config.model.max_pixels / (28 * 28 * 256)) ** 2
    total_memory = INITIAL_MEMORY + SCALING_FACTOR * constant * quadratic
    design.info(f"Total memory usage: {total_memory:.2f} GB")
    if total_memory > 75.0:
        design.error("Potential memory usage is too high, decrease either training steps or mini batch size")
        exit(1)
    
    # Apply command-line overrides
    if args.episodes_per_batch:
        config.training.episodes_per_batch = args.episodes_per_batch
    if args.max_steps:
        config.training.max_training_steps = args.max_steps
    if args.lr:
        config.training.lr = args.lr
    if args.out_dir:
        config.out_dir = args.out_dir
    if args.debug:
        config.verbose = True
    
    # Load tasks
    if args.tasks_json:
        # Tasks provided as JSON list via command line
        tasks = load_tasks(args.tasks_json, config.actor.system_prompt)
    elif args.tasks:
        # Tasks provided as file path or HuggingFace dataset
        tasks = load_tasks(args.tasks, config.actor.system_prompt)
    elif hasattr(config.actor, 'tasks_file') and config.actor.tasks_file:
        # Fallback to config file path for backwards compatibility
        tasks = load_tasks(config.actor.tasks_file, config.actor.system_prompt)
    else:
        # Default to browser_2048_tasks.jsonl if it exists
        default_tasks_path = "browser_2048_tasks.jsonl"
        if Path(default_tasks_path).exists():
            design.info(f"No tasks specified, using default: {default_tasks_path}")
            tasks = load_tasks(default_tasks_path, config.actor.system_prompt)
        else:
            raise ValueError("No tasks specified. Use --tasks, --tasks-json, or specify tasks_file in config")
    
    # Run training
    await train(config, tasks)


if __name__ == "__main__":
    asyncio.run(main())