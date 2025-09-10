"""HUD RL command for reinforcement learning training."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from hud.utils.design import HUDDesign

console = Console()
design = HUDDesign()


def create_default_config(
    dataset: str,
    env_image: Optional[str] = None,
    model: str = "Qwen/Qwen2.5-3B",
) -> dict[str, Any]:
    """Create a default RL configuration.
    
    Args:
        dataset: Dataset path or HuggingFace ID
        env_image: Environment Docker image (auto-detected if None)
        model: Base model to use
        
    Returns:
        Configuration dictionary
    """
    # Try to auto-detect environment from dataset if not provided
    if env_image is None:
        # Simple heuristic based on common dataset names
        dataset_lower = dataset.lower()
        if "2048" in dataset_lower:
            env_image = "hudpython/hud-text-2048:latest"
        elif "sheet" in dataset_lower:
            env_image = "hudpython/hud-sheets:latest"
        elif "browser" in dataset_lower or "web" in dataset_lower:
            env_image = "hudpython/hud-browser:latest"
        else:
            env_image = "hudpython/hud-browser:latest"  # Default
    
    config = {
        "dataset": dataset,
        "environment": {
            "image": env_image,
            "mcp_config": {
                "hud": {
                    "url": "https://mcp.hud.so/v3/mcp",
                    "headers": {
                        "Authorization": "Bearer ${HUD_API_KEY}",
                        "Mcp-Image": env_image,
                    }
                }
            }
        },
        "model": {
            "name": model,
            "lora": {
                "r": 128,
                "lora_alpha": 128,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        },
        "training": {
            "batch_size": 32,
            "num_epochs": 3,
            "learning_rate": 1e-4,
            "warmup_steps": 100,
            "gradient_checkpointing": True,
            "fp16": True,
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 100,
            "max_grad_norm": 1.0,
        },
        "ppo": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "value_clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_steps_per_episode": 100,
            "num_rollout_workers": 4,
            "num_envs_per_worker": 4,
        },
        "dataset_config": {
            "max_samples": None,  # Use all samples
            "shuffle": True,
            "seed": 42,
        },
        "output": {
            "checkpoint_dir": "./checkpoints",
            "tensorboard_dir": "./tensorboard",
            "final_model_dir": "./final_model",
        }
    }
    
    return config


def rl_command(
    dataset: str = typer.Argument(
        ...,
        help="Dataset path (JSON file) or HuggingFace dataset ID (e.g., 'hud-evals/2048-taskset')",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="RL configuration file (auto-created if not provided)",
    ),
    env_image: Optional[str] = typer.Option(
        None,
        "--env",
        "-e",
        help="Environment Docker image (auto-detected from dataset if not provided)",
    ),
    model: str = typer.Option(
        "Qwen/Qwen2.5-3B",
        "--model",
        "-m",
        help="Base model to use for training",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Override batch size",
    ),
    num_epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Override number of epochs",
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Override learning rate",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples",
        help="Limit number of samples from dataset",
    ),
    fp16: Optional[bool] = typer.Option(
        None,
        "--fp16/--no-fp16",
        help="Enable/disable FP16 training",
    ),
    wandb: bool = typer.Option(
        False,
        "--wandb",
        help="Enable Weights & Biases logging",
    ),
    wandb_project: str = typer.Option(
        "hud-rl",
        "--wandb-project",
        help="W&B project name",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
) -> None:
    """🧠 Run RL training on HUD environments.
    
    This command handles the complete RL training workflow:
    1. Creates or loads configuration
    2. Validates dataset and environment
    3. Runs PPO training with the specified model
    
    Examples:
        # Basic usage - auto-creates config
        hud rl hud-evals/2048-taskset
        
        # With custom model
        hud rl tasks.json --model meta-llama/Llama-3.2-3B
        
        # Override training parameters
        hud rl dataset.json --batch-size 16 --epochs 5 --lr 5e-5
        
        # Use existing config
        hud rl dataset.json --config my_rl_config.yaml
        
        # Enable W&B logging
        hud rl dataset.json --wandb --wandb-project my-project
        
        # Skip confirmation
        hud rl dataset.json --yes
    """
    console.print(Panel.fit("🧠 [bold cyan]HUD RL Training[/bold cyan]", border_style="cyan"))
    
    # Handle config file
    config_path = config or Path("rl_config.yaml")
    config_exists = config_path.exists()
    
    if config_exists:
        # Load existing config
        design.info(f"Loading config from: {config_path}")
        try:
            import yaml
        except ImportError:
            design.error("PyYAML is required. Install with: pip install pyyaml")
            raise typer.Exit(1)
        
        with open(config_path) as f:
            rl_config = yaml.safe_load(f)
            
        # Update dataset if different
        if rl_config.get("dataset") != dataset:
            design.warning(f"Updating dataset in config: {rl_config.get('dataset')} → {dataset}")
            rl_config["dataset"] = dataset
    else:
        # Create new config
        design.info("No config file found. Creating default configuration...")
        rl_config = create_default_config(dataset, env_image, model)
        
        # Save config
        try:
            import yaml
        except ImportError:
            design.error("PyYAML is required. Install with: pip install pyyaml")
            raise typer.Exit(1)
        
        with open(config_path, "w") as f:
            yaml.dump(rl_config, f, default_flow_style=False, sort_keys=False)
        design.success(f"Created config file: {config_path}")
    
    # Apply command-line overrides
    if env_image and env_image != rl_config["environment"]["image"]:
        rl_config["environment"]["image"] = env_image
        rl_config["environment"]["mcp_config"]["hud"]["headers"]["Mcp-Image"] = env_image
    if model != "Qwen/Qwen2.5-3B":  # Only override if not default
        rl_config["model"]["name"] = model
    if batch_size is not None:
        rl_config["training"]["batch_size"] = batch_size
    if num_epochs is not None:
        rl_config["training"]["num_epochs"] = num_epochs
    if learning_rate is not None:
        rl_config["training"]["learning_rate"] = learning_rate
    if max_samples is not None:
        rl_config["dataset_config"]["max_samples"] = max_samples
    if fp16 is not None:
        rl_config["training"]["fp16"] = fp16
    
    # Add W&B config if enabled
    if wandb:
        rl_config["wandb"] = {
            "enabled": True,
            "project": wandb_project,
            "name": f"rl-{Path(dataset).stem}-{rl_config['model']['name'].split('/')[-1]}",
        }
    
    # Display configuration summary
    design.section_title("Training Configuration")
    design.info(f"Dataset: {dataset}")
    design.info(f"Environment: {rl_config['environment']['image']}")
    design.info(f"Model: {rl_config['model']['name']}")
    design.info(f"Batch Size: {rl_config['training']['batch_size']}")
    design.info(f"Epochs: {rl_config['training']['num_epochs']}")
    design.info(f"Learning Rate: {rl_config['training']['learning_rate']}")
    
    if verbose:
        design.section_title("Full Configuration")
        console.print(rl_config)
    
    # Check for required environment variables
    if not os.getenv("HUD_API_KEY"):
        design.warning("HUD_API_KEY not set. You'll need it for remote environments.")
        design.info("Get your API key at: https://app.hud.so")
    
    # Confirm before starting
    if not yes:
        if not config_exists:
            design.info(f"\n💡 Config saved to: {config_path}")
            design.info("You can edit this file to customize training parameters")
        
        if not Confirm.ask("\n[bold yellow]Start training?[/bold yellow]"):
            design.info("Training cancelled. Config file saved for future use.")
            raise typer.Exit(0)
    
    # Run training (placeholder for now)
    design.section_title("Starting Training")
    
    # Construct the command that would be run
    train_cmd = [
        sys.executable, "-m", "rl.train",
        "--dataset", dataset,
        "--config", str(config_path),
    ]
    
    if verbose:
        train_cmd.append("--verbose")
    
    design.info("🚀 Launching RL training...")
    design.info(f"Command: {' '.join(train_cmd)}")
    
    # For now, show placeholder message
    design.warning("\n⚠️  RL training integration is pending.")
    design.info("The training script will:")
    design.info("1. Load the dataset from " + dataset)
    design.info("2. Initialize PPO with " + rl_config['model']['name'])
    design.info("3. Create " + str(rl_config['ppo']['num_rollout_workers']) + " rollout workers")
    design.info("4. Train for " + str(rl_config['training']['num_epochs']) + " epochs")
    design.info("5. Save checkpoints to " + rl_config['output']['checkpoint_dir'])
    
    if wandb:
        design.info("6. Log metrics to W&B project: " + wandb_project)
    
    design.info("\nOnce integrated, run this command again to start actual training!")
    
    # Save updated config if there were overrides
    if any([
        env_image and env_image != rl_config["environment"]["image"],
        model != "Qwen/Qwen2.5-3B",
        batch_size is not None,
        num_epochs is not None,
        learning_rate is not None,
        max_samples is not None,
        fp16 is not None,
        wandb
    ]):
        with open(config_path, "w") as f:
            yaml.dump(rl_config, f, default_flow_style=False, sort_keys=False)
        design.info(f"\n📝 Updated config saved to: {config_path}")