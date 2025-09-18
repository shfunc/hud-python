"""Display utilities for RL training configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from hud.rl.config import Config

console = Console()


def display_gpu_info(gpu_info: dict[str, Any]) -> None:
    """Display GPU information in a table."""
    if not gpu_info["available"]:
        console.print(f"[red]❌ CUDA not available: {gpu_info.get('error', 'Unknown error')}[/red]")
        return

    gpu_table = Table(title="🖥️  Available GPUs", title_style="bold cyan")
    gpu_table.add_column("Index", style="yellow")
    gpu_table.add_column("Name", style="cyan")
    gpu_table.add_column("Memory", style="green")

    for device in gpu_info["devices"]:
        gpu_table.add_row(f"GPU {device['index']}", device["name"], f"{device['memory_gb']:.1f} GB")

    console.print(gpu_table)


def display_preset_table(presets: list[dict[str, Any]], gpu_memory_gb: float) -> None:
    """Display training configuration presets in a table."""
    preset_table = Table(title="📊 Training Configuration Presets", title_style="bold cyan")
    preset_table.add_column("Option", style="yellow")
    preset_table.add_column("Steps", style="cyan")
    preset_table.add_column("Mini-batch", style="cyan")
    preset_table.add_column("Group", style="cyan")
    preset_table.add_column("Episodes/batch", style="cyan")

    # Add time columns for A100
    if gpu_memory_gb >= 40:
        preset_table.add_column("Tasks/hour", style="green")
        preset_table.add_column("Updates/hour", style="green")

    for i, preset in enumerate(presets):
        row = [
            f"{i + 1}. {preset['name']}",
            str(preset["max_steps_per_episode"]),
            str(preset["mini_batch_size"]),
            str(preset["group_size"]),
            str(preset["batch_size"]),
        ]
        if "tasks_per_hour" in preset:
            row.extend(
                [
                    str(preset["tasks_per_hour"]),
                    str(preset["steps_per_hour"]),
                ]
            )
        preset_table.add_row(*row)

    console.print("\n")
    console.print(preset_table)
    console.print("\n")


def display_config_summary(
    config: Config, tasks_count: int, gpu_info: dict[str, Any], estimated_memory: float
) -> None:
    """Display comprehensive configuration summary for review."""
    console.print("\n[bold cyan]📋 RL Training Configuration Summary[/bold cyan]\n")

    # GPU Information
    if gpu_info["available"]:
        gpu_table = Table(title="🖥️  GPU Information", title_style="bold yellow")
        gpu_table.add_column("Property", style="cyan")
        gpu_table.add_column("Value", style="green")

        device = gpu_info["devices"][0]  # Primary GPU
        gpu_table.add_row("GPU 0", device["name"])
        gpu_table.add_row("Memory", f"{device['memory_gb']:.1f} GB")
        gpu_table.add_row("Compute Capability", "8.0")  # Assuming A100

        console.print(gpu_table)

    # Model Configuration
    model_table = Table(title="🤖 Model Configuration", title_style="bold yellow")
    model_table.add_column("Parameter", style="cyan")
    model_table.add_column("Value", style="green")

    model_table.add_row("Base Model", config.model.base_model)
    model_table.add_row("LoRA Rank (r)", str(config.model.lora_r))
    model_table.add_row("LoRA Alpha", str(config.model.lora_alpha))
    model_table.add_row("LoRA Dropout", str(config.model.lora_dropout))

    console.print(model_table)

    # Training Configuration
    training_table = Table(title="🎯 Training Configuration", title_style="bold yellow")
    training_table.add_column("Parameter", style="cyan")
    training_table.add_column("Value", style="green")

    training_table.add_row("Tasks Count", str(tasks_count))
    training_table.add_row("Learning Rate", f"{config.training.lr:.1e}")
    training_table.add_row("Epochs", str(config.training.epochs))
    training_table.add_row("Mini Batch Size", str(config.training.mini_batch_size))
    training_table.add_row("Batch Size", str(config.training.batch_size))
    training_table.add_row("Group Size", str(config.training.group_size))
    training_table.add_row("Training Steps", str(config.training.training_steps))
    training_table.add_row("Max Parallel Episodes", str(config.actor.max_parallel_episodes))

    console.print(training_table)

    # Memory Estimation
    memory_table = Table(title="💾 Memory Estimation", title_style="bold yellow")
    memory_table.add_column("Metric", style="cyan")
    memory_table.add_column("Value", style="green")

    memory_table.add_row("Estimated GPU Memory", f"{estimated_memory:.1f} GB")
    if gpu_info["available"]:
        available_memory = gpu_info["devices"][0]["memory_gb"]
        memory_table.add_row("Available GPU Memory", f"{available_memory:.1f} GB")

        if estimated_memory > available_memory:
            status = "[red]⚠️  May exceed available memory[/red]"
        else:
            status = "[green]✅ Within memory limits[/green]"
        memory_table.add_row("Status", status)

    console.print(memory_table)
    console.print("\n")
