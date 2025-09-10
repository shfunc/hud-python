"""Configuration generation and management for RL training."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import typer
from hud.rl.config import Config, ModelConfig, TrainingConfig, ActorConfig
from rich.console import Console
from hud.utils.design import design
from .presets import estimate_memory_usage
from .display import display_preset_table

console = Console()


def generate_config_interactive(
    model_name: str,
    tasks_count: int,
    presets: List[Dict[str, Any]],
    output_dir: str = "checkpoints",
) -> Tuple[Config, float]:
    """Generate RL training configuration interactively."""
    # Display preset options
    display_preset_table(presets, 80.0)  # Assuming A100 80GB
    
    # Let user select preset
    preset_choice = design.select(
        "Select a training configuration preset:",
        choices=[{"name": p["name"], "value": i} for i, p in enumerate(presets)],
        default=1 if len(presets) > 1 else 0,  # Default to "Balanced" if available
    )
    
    selected_preset = presets[preset_choice]
    
    # Use preset values directly
    max_steps_per_episode = selected_preset['max_steps_per_episode']
    temperature = 0.7
    
    # Calculate memory estimate
    max_pixels = 256 * 28 * 28
    estimated_memory = estimate_memory_usage(
        selected_preset['mini_batch_size'],
        max_steps_per_episode,
        max_pixels
    )
    
    # Create config
    config = Config(
        seed=42,
        out_dir=output_dir,
        adapter_prefix="rl-adapter",
        model=ModelConfig(
            base_model=model_name,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=(
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj", "vision_language_merger.*"
            ),
            min_pixels=max_pixels,
            max_pixels=max_pixels,
        ),
        training=TrainingConfig(
            lr=selected_preset['lr'],
            epochs=selected_preset['epochs'],
            mini_batch_size=selected_preset['mini_batch_size'],
            episodes_per_batch=selected_preset['episodes_per_batch'],
            group_size=selected_preset['group_size'],
            kl_beta=0.001,
            save_every_batches=1,
        ),
        actor=ActorConfig(
            vllm_base_url="http://localhost:8000/v1",
            vllm_api_key="token-abc123",
            temperature=temperature,
            max_tokens=2048,
            max_steps_per_episode=max_steps_per_episode,
            parallel_episodes=selected_preset['episodes_per_batch'],
            system_prompt="You are an expert agent. Complete the task efficiently.",
        ),
    )
    
    return config, estimated_memory


def save_config(config: Config, path: Path) -> None:
    """Save configuration to a JSON file."""
    config_dict = {
        "seed": config.seed,
        "out_dir": config.out_dir,
        "adapter_prefix": config.adapter_prefix,
        "model": {
            "base_model": config.model.base_model,
            "lora_r": config.model.lora_r,
            "lora_alpha": config.model.lora_alpha,
            "lora_dropout": config.model.lora_dropout,
            "target_modules": list(config.model.target_modules),
            "min_pixels": config.model.min_pixels,
            "max_pixels": config.model.max_pixels,
        },
        "training": {
            "lr": config.training.lr,
            "epochs": config.training.epochs,
            "mini_batch_size": config.training.mini_batch_size,
            "episodes_per_batch": config.training.episodes_per_batch,
            "group_size": config.training.group_size,
            "max_training_steps": config.training.max_training_steps,
            "kl_beta": config.training.kl_beta,
            "save_every_batches": config.training.save_every_batches,
        },
        "actor": {
            "vllm_base_url": config.actor.vllm_base_url,
            "vllm_api_key": config.actor.vllm_api_key,
            "temperature": config.actor.temperature,
            "max_tokens": config.actor.max_tokens,
            "max_steps_per_episode": config.actor.max_steps_per_episode,
            "parallel_episodes": config.actor.parallel_episodes,
            "system_prompt": config.actor.system_prompt,
        },
    }
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)
        f.write('\n')  # Add newline at end of file
    
    if not path.name.startswith('.'):  # Don't show message for temp files
        console.print(f"[green]✅ Configuration saved to {path}[/green]")


def load_config(path: Path) -> Config:
    """Load configuration from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    return Config(
        seed=data.get("seed", 42),
        out_dir=data.get("out_dir", "checkpoints"),
        adapter_prefix=data.get("adapter_prefix", "rl-adapter"),
        model=ModelConfig(**data["model"]),
        training=TrainingConfig(**data["training"]),
        actor=ActorConfig(**data["actor"]),
    )
