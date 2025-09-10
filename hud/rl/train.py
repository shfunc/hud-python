"""Main training loop for GRPO RL."""

import os
# Force training to use GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Disable tokenizer parallelism warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import asyncio
import argparse
import json
import hud
from pathlib import Path
import uuid
import time
import logging
from typing import List

from hud.rl.config import Config
from hud.rl.actor import Actor
from hud.rl.learner import GRPOLearner
from hud.rl.buffer import GroupedReplayBuffer
from hud.rl.vllm_adapter import VLLMAdapter
from hud.rl.utils import set_seed, ensure_dir, prepare_training_samples, load_tasks
from hud.rl.types import Batch
from hud.datasets import Task

logger = logging.getLogger(__name__)


async def train(config: Config, tasks: List[Task], verbose: bool = False):
    """Main training loop."""
    logger.info("=" * 50)
    logger.info("Starting GRPO Training")
    logger.info("=" * 50)
    
    # Set random seed
    set_seed(config.seed)
    ensure_dir(config.out_dir)
    
    # Initialize components
    logger.info("\n[1/4] Initializing components...")
    actor = Actor(config, tasks, verbose=verbose)
    learner = GRPOLearner(config)
    buffer = GroupedReplayBuffer(
        max_size=1000,
        success_buffer_size=64,
        group_size=config.training.group_size
    )
    vllm = VLLMAdapter(
        config.actor.vllm_base_url,
        config.actor.vllm_api_key
    )
    
    # Training state
    step = 0
    total_episodes = 0
    
    logger.info("\n[2/4] Starting training loop...")
    logger.info(f"Config: {config.training.episodes_per_batch} episodes/batch, "
               f"{config.training.max_training_steps} max steps")
    
    with hud.job(name=f"RL Training - {config.model.base_model.split('/')[-1]}", metadata={"config": config.to_dict()}) as job_obj:
        while (step < config.training.max_training_steps and 
            total_episodes < config.training.max_total_episodes):
            
            logger.info(f"\n--- Step {step + 1}/{config.training.max_training_steps} ---")
            
            # Collect episodes in groups for GRPO
            n_groups = config.training.episodes_per_batch // config.training.group_size
            if n_groups == 0:
                n_groups = 1
                logger.warning(f"[3/4] Warning: episodes_per_batch ({config.training.episodes_per_batch}) < group_size ({config.training.group_size})")
                logger.info(f"      Collecting 1 group of {config.training.group_size} episodes")
            else:
                logger.info(f"[3/4] Collecting {n_groups} groups × {config.training.group_size} episodes = {n_groups * config.training.group_size} total")
            
            episodes = await actor.collect_groups(config.training.group_size, n_groups, job_id=job_obj.id)
            buffer.add(episodes)
            total_episodes += len(episodes)
            
            # Get buffer statistics
            stats = buffer.get_stats()
            logger.debug(f"Buffer stats: {json.dumps(stats, indent=2)}")
            
            # Sample training groups
            groups = buffer.sample_groups()
            
            if not groups:
                logger.info("No complete groups available yet, continuing collection...")
                continue
            
            # Process each group
            logger.info(f"[4/4] Training on {len(groups)} groups...")
            
            # Collect metrics across all groups
            all_rewards = []
            all_losses = []
            all_kls = []
            all_clipped_fractions = []
            all_advantage_stds = []
            all_ratio_means = []
            all_ratio_stds = []
            all_grad_norms = []
            all_completion_lengths = []
            
            for group_key, group_episodes in groups.items():
                logger.debug(f"  Processing group '{group_key}' with {len(group_episodes)} episodes")
                
                # Prepare training samples
                all_samples = []
                for episode in group_episodes:
                    samples = prepare_training_samples(
                        episode,
                        learner.processor,
                        learner,
                        config
                    )
                    all_samples.extend(samples)
                
                if not all_samples:
                    continue
                
                # Create batch and update
                batch = Batch(
                    samples=all_samples,
                    episodes=group_episodes
                )

                logger.debug(f"Batch samples: {len(batch.samples)}")
                
                learner.update(batch)
                
                # Collect metrics for aggregation
                group_rewards = [ep.terminal_reward for ep in group_episodes]
                all_rewards.extend(group_rewards)
                all_losses.append(learner.last_loss)
                
                # Collect completion lengths from episodes
                for ep in group_episodes:
                    if hasattr(ep, 'conversation_history') and ep.conversation_history:
                        # Count assistant responses
                        completion_length = sum(
                            len(str(msg.get('content', ''))) 
                            for msg in ep.conversation_history 
                            if msg.get('role') == 'assistant' and msg.get('content')
                        )
                        all_completion_lengths.append(completion_length)
                
                if hasattr(learner, 'last_metrics') and learner.last_metrics:
                    metrics = learner.last_metrics
                    all_kls.append(metrics.get('kl_loss', 0))
                    all_clipped_fractions.append(metrics.get('clipped_fraction', 0))
                    
                    # Get gradient norm
                    if 'grad_norm' in metrics:
                        all_grad_norms.append(metrics['grad_norm'])
                    
                    if 'advantages' in metrics:
                        advantages = metrics['advantages']
                        all_advantage_stds.append(float(advantages.std()))
                    
                    if 'ratios' in metrics:
                        ratios = metrics['ratios']
                        all_ratio_means.append(float(ratios.mean()))
                        all_ratio_stds.append(float(ratios.std()))
            
            # Log aggregated metrics for this training step
            if all_rewards:
                reward_mean = sum(all_rewards) / len(all_rewards)
                reward_std = (sum((r - reward_mean) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
                
                step_metrics = {
                    "step": step,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "loss": sum(all_losses) / len(all_losses) if all_losses else 0,
                }
                
                if all_kls:
                    step_metrics["kl"] = sum(all_kls) / len(all_kls)
                    step_metrics["clipped_fraction"] = sum(all_clipped_fractions) / len(all_clipped_fractions)
                
                if all_advantage_stds:
                    step_metrics["advantage_std"] = sum(all_advantage_stds) / len(all_advantage_stds)
                
                if all_ratio_means:
                    step_metrics["ratio_mean"] = sum(all_ratio_means) / len(all_ratio_means)
                    step_metrics["ratio_std"] = sum(all_ratio_stds) / len(all_ratio_stds)
                
                if all_grad_norms:
                    step_metrics["grad_norm"] = sum(all_grad_norms) / len(all_grad_norms)
                
                if all_completion_lengths:
                    step_metrics["avg_completion_length"] = sum(all_completion_lengths) / len(all_completion_lengths)
                
                job_obj.log_sync(step_metrics)
            
            # Save checkpoint and update vLLM
            step += 1
            if step % config.training.save_every_batches == 0:
                checkpoint_id = uuid.uuid4()
                checkpoint_path = Path(config.out_dir) / f"{config.adapter_prefix}-{checkpoint_id}"
                learner.save(str(checkpoint_path))

                # Hot-load to vLLM
                adapter_name = f"{config.adapter_prefix}-{checkpoint_id}"
                if vllm.load_adapter(adapter_name, str(checkpoint_path)):
                    actor.update_adapter(adapter_name)
                    logger.info(f"✓ Checkpoint saved and loaded: {adapter_name}")
                else:
                    logger.warning(f"Failed to hot-load adapter {adapter_name}")
            
            # Log progress
            logger.info(f"\nProgress: Step {step}, Total episodes: {total_episodes}")
    
    logger.info("\n" + "=" * 50)
    logger.info("Training completed!")
    logger.info(f"Final: {step} steps, {total_episodes} episodes")
    logger.info("=" * 50)


async def main():
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
        logger.info("[TEST MODE] Using minimal configuration")
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
    logger.info(f"Total memory usage: {total_memory:.2f} GB")
    if total_memory > 75.0:
        logger.error("Potential memory usage is too high, decrease either training steps or mini batch size")
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
        config.debug = True
    
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
            logger.info(f"No tasks specified, using default: {default_tasks_path}")
            tasks = load_tasks(default_tasks_path, config.actor.system_prompt)
        else:
            raise ValueError("No tasks specified. Use --tasks, --tasks-json, or specify tasks_file in config")
    
    # Run training
    await train(config, tasks, verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main())