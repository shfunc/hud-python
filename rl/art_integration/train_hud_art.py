"""Train agents on HUD MCP environments using ART."""

import asyncio
import logging
import os
import random
from typing import Any

import art
import hud
import yaml
from art.local import LocalBackend
from art.utils import iterate_dataset
from dotenv import load_dotenv
from hud.clients import MCPClient

from .art_agent import ARTTrainingAgent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def train_hud_with_art(
    environment: str = "2048",
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    model_name: str | None = None,
    project_name: str = "hud-art",
    num_training_scenarios: int = 16,
    num_val_scenarios: int = 8,
    groups_per_step: int = 2,
    rollouts_per_group: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 1e-5,
    max_turns: int = 10,
    config_path: str | None = None,
    output_dir: str = "./.art",
) -> None:
    """
    Train an agent on HUD MCP environments using ART.
    
    Args:
        environment: Environment name (2048, browser, custom)
        base_model: Base model to fine-tune
        model_name: Name for the trained model (defaults to env-base_model)
        project_name: Project name for tracking
        num_training_scenarios: Number of training scenarios
        num_val_scenarios: Number of validation scenarios
        groups_per_step: Scenarios to process per training step
        rollouts_per_group: Different responses per scenario for comparison
        num_epochs: Number of passes through training data
        learning_rate: Learning rate for training
        max_turns: Maximum turns per episode
        config_path: Path to config file with MCP settings
        output_dir: Directory for checkpoints
    """
    # Load configuration
    config = {}
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Get MCP configuration for environment
    mcp_config = None
    if environment == "2048":
        mcp_config = {
            "local": {
                "command": "docker",
                "args": ["run", "--rm", "-i", "hud-text-2048"],
            }
        }
    elif environment == "browser":
        mcp_config = {
            "local": {
                "command": "docker",
                "args": ["run", "--rm", "-i", "hud-browser"],
            }
        }
    else:
        # Try to load from config
        if config and "environments" in config:
            env_config = config["environments"].get(environment)
            if env_config:
                mcp_config = env_config.get("mcp_config")
                
    if not mcp_config:
        raise ValueError(f"No MCP configuration found for environment: {environment}")
        
    # Default model name
    if not model_name:
        model_name = f"{environment}-{base_model.split('/')[-1].lower()}"
        
    logger.info(f"Training {model_name} on {environment} environment")
    logger.info(f"Base model: {base_model}")
    logger.info(f"MCP config: {mcp_config}")
    
    # Initialize HUD MCP client
    mcp_client = MCPClient(mcp_config=mcp_config)
    await mcp_client.initialize()
    logger.info("MCP client initialized")
    
    # Load scenarios from HuggingFace datasets
    from datasets import load_dataset
    from hud.datasets import to_taskconfigs
    
    # Map environment names to HF dataset names
    dataset_map = {
        "2048": "hud-evals/2048-taskset",
        "browser": "hud-evals/SheetBench-50",
        "gmail": "hud-evals/gmail-taskset",
    }
    
    if environment not in dataset_map:
        raise ValueError(f"Unknown environment: {environment}. Available: {list(dataset_map.keys())}")
        
    logger.info(f"Loading scenarios from HuggingFace: {dataset_map[environment]}")
    dataset = load_dataset(dataset_map[environment], split="train")
    all_scenarios = to_taskconfigs(dataset)[:num_training_scenarios + num_val_scenarios]
        
    # Split into train/val
    random.shuffle(all_scenarios)
    train_scenarios = all_scenarios[:num_training_scenarios]
    val_scenarios = all_scenarios[num_training_scenarios:]
    
    logger.info(f"Train scenarios: {len(train_scenarios)}")
    logger.info(f"Val scenarios: {len(val_scenarios)}")
    
    # Initialize ART model
    art_model = art.TrainableModel(
        name=model_name,
        project=project_name,
        base_model=base_model,
    )
    
    # Configure for smaller GPUs if needed
    art_model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=8192,
        ),
    )
    
    # Initialize ART backend
    backend = LocalBackend(
        in_process=True,
        path=output_dir,
    )
    
    await art_model.register(backend)
    logger.info("ART model registered with backend")
    
    # Create dataset iterator
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=groups_per_step,
        num_epochs=num_epochs,
        initial_step=await art_model.get_step(),
    )
    
    # Training loop
    for batch in train_iterator:
        logger.info(f"Training step {batch.step}")
        
        # Collect trajectory groups
        groups = []
        for scenario in batch.items:
            # Create multiple rollouts for each scenario
            trajectories = []
            for _ in range(rollouts_per_group):
                agent = ARTTrainingAgent(
                    art_model=art_model,
                    mcp_client=mcp_client,
                    max_turns=max_turns,
                )
                trajectory = await agent.run_with_trajectory(scenario)
                trajectories.append(trajectory)
                
            group = art.TrajectoryGroup(trajectories)
            groups.append(group)
            
        # Train the model
        logger.info(f"Training on {len(groups)} groups...")
        await art_model.train(
            groups,
            config=art.TrainConfig(learning_rate=learning_rate),
        )
        
        # Optional: Run validation
        if batch.step % 5 == 0 and val_scenarios:
            logger.info("Running validation...")
            val_trajectories = []
            for scenario in val_scenarios[:2]:  # Quick val on 2 scenarios
                agent = ARTTrainingAgent(
                    art_model=art_model,
                    mcp_client=mcp_client,
                    max_turns=max_turns,
                )
                trajectory = await agent.run_with_trajectory(scenario)
                val_trajectories.append(trajectory)
                
            avg_reward = sum(t.reward for t in val_trajectories) / len(val_trajectories)
            logger.info(f"Validation average reward: {avg_reward:.3f}")
            
    logger.info("Training completed!")
    logger.info(f"Model saved to: {output_dir}/{project_name}/models/{model_name}")
    
    # Close MCP client
    await mcp_client.close()


async def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train HUD agents with ART")
    parser.add_argument(
        "--env",
        default="2048",
        help="Environment to train on (2048, browser, or custom)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--model-name",
        help="Name for the trained model",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=16,
        help="Number of training scenarios",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of training steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per episode",
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    
    # Calculate training parameters
    groups_per_step = max(1, args.scenarios // args.steps)
    
    await train_hud_with_art(
        environment=args.env,
        base_model=args.base_model,
        model_name=args.model_name,
        num_training_scenarios=args.scenarios,
        groups_per_step=groups_per_step,
        learning_rate=args.learning_rate,
        max_turns=args.max_turns,
        config_path=args.config,
    )


if __name__ == "__main__":
    asyncio.run(main())