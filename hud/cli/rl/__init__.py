"""RL training command for HUD CLI."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Union

import typer
from rich.console import Console
from rich.progress import Progress

# Import local modules first
from .gpu import detect_cuda_devices, validate_gpu_memory, select_gpu_for_vllm
from .presets import get_training_presets, estimate_memory_usage
from .vllm import check_vllm_server, start_vllm_server, wait_for_vllm_server, kill_vllm_server
from .display import display_gpu_info, display_config_summary
from .config import generate_config_interactive, save_config, load_config

# Then import HUD modules
from hud.utils.design import design
from hud.rl.utils import load_tasks
from hud.rl.train import train
from hud.datasets import Task

console = Console()


def rl_command(
    tasks_file: Optional[str] = typer.Argument(
        None,
        help="Path to tasks file (JSON/JSONL) or HuggingFace dataset name",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to train (default: interactive selection)",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to existing configuration file",
    ),
    output_dir: str = typer.Option(
        "checkpoints",
        "--output-dir",
        "-o",
        help="Output directory for checkpoints",
    ),
    restart: bool = typer.Option(
        False,
        "--restart",
        help="Restart the vLLM server before training",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    """Run GRPO reinforcement learning training on tasks."""
    # Configure logging based on verbose flag BEFORE any output
    if not verbose:
        # Set environment variable for HUD components
        os.environ["HUD_LOG_LEVEL"] = "WARNING"
        
        # Configure logging levels
        logging.basicConfig(level=logging.WARNING, force=True)
        
        # Get root logger and set its level
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        
        # Suppress INFO logs from various components
        for logger_name in ["httpx", "hud.agents", "hud.utils.design", "hud", "asyncio", "transformers"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        # Also set HUD agent logger explicitly
        logging.getLogger("hud.agents.base").setLevel(logging.WARNING)
    else:
        # In verbose mode, show everything
        logging.basicConfig(level=logging.INFO)
    
    console.print("\n[bold cyan]🚀 HUD RL Training[/bold cyan]\n")
    
    # Check Python version compatibility
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 13:
        console.print("[red]⚠️  Warning: Python 3.13+ detected![/red]")
        console.print("[yellow]vLLM has compatibility issues with Python 3.13.[/yellow]")
        console.print("[yellow]Recommended: Use Python 3.12 or 3.11[/yellow]")
        console.print("\n[dim]To create a new environment with Python 3.12:[/dim]")
        console.print("[dim]  1. Exit this shell: exit[/dim]")
        console.print("[dim]  2. Remove current venv: sudo rm -rf .venv[/dim]")
        console.print("[dim]  3. Create new venv: uv venv --python 3.12[/dim]")
        console.print("[dim]  4. Install dependencies: uv pip install -e '.[rl]'[/dim]")
        
        if not typer.confirm("\nDo you want to continue anyway?", default=False):
            raise typer.Exit(1)
    
    # Step 1: Validate CUDA devices
    console.print("[yellow]Checking GPU availability...[/yellow]")
    gpu_info = detect_cuda_devices()
    
    if not gpu_info["available"]:
        console.print(f"[red]❌ {gpu_info['error']}[/red]")
        console.print("[yellow]RL training requires CUDA-capable GPUs[/yellow]")
        raise typer.Exit(1)
    
    display_gpu_info(gpu_info)
    
    # Get primary GPU memory for configuration
    primary_gpu = gpu_info["devices"][0]
    gpu_memory_gb = primary_gpu["memory_gb"]
    
    # Validate GPU memory for 3B model
    if not validate_gpu_memory(gpu_memory_gb, "3B"):
        console.print(f"[red]❌ Insufficient GPU memory ({gpu_memory_gb:.1f} GB)[/red]")
        console.print("[yellow]Qwen 2.5 VL 3B requires at least 12 GB of GPU memory[/yellow]")
        raise typer.Exit(1)
    
    # Step 2: Load tasks
    if tasks_file:
        console.print(f"\n[cyan]Loading tasks from: {tasks_file}[/cyan]")
    else:
        # Auto-detect tasks file
        possible_files = ["tasks.json", "tasks.jsonl", "browser_2048_tasks.jsonl"]
        for f in possible_files:
            if Path(f).exists():
                tasks_file = f
                console.print(f"[green]Auto-detected tasks file: {f}[/green]")
                break
        
        if not tasks_file:
            console.print("[red]❌ No tasks file specified or auto-detected[/red]")
            console.print("[yellow]Please provide a tasks file or create one of: tasks.json, tasks.jsonl[/yellow]")
            raise typer.Exit(1)
    
    # Load the tasks
    tasks = load_tasks(tasks_file)
    console.print(f"[green]✅ Loaded {len(tasks)} tasks[/green]")
    
    # Validate tasks
    invalid_tasks = []
    for i, task in enumerate(tasks):
        if not hasattr(task, 'prompt') or not task.prompt:
            invalid_tasks.append(f"Task {i}: missing 'prompt' field")
        if not hasattr(task, 'mcp_config') or not task.mcp_config:
            invalid_tasks.append(f"Task {i}: missing 'mcp_config' field")
    
    if invalid_tasks:
        console.print("[red]❌ Invalid tasks found:[/red]")
        for error in invalid_tasks[:5]:  # Show first 5 errors
            console.print(f"  - {error}")
        if len(invalid_tasks) > 5:
            console.print(f"  ... and {len(invalid_tasks) - 5} more")
        raise typer.Exit(1)
    
    # Step 3: Model selection (if not provided)
    if model is None and not config_file:
        model = design.select(
            "Select a model for RL training:",
            choices=[
                {"name": "Qwen 2.5 VL 3B (Recommended - Vision-Language)", "value": "Qwen/Qwen2.5-VL-3B-Instruct"},
                {"name": "Custom model", "value": "custom"},
            ],
            default="Qwen/Qwen2.5-VL-3B-Instruct",
        )
        
        if model == "custom":
            console.print("Enter the model name (HuggingFace ID):")
            model = input().strip()
    
    # Step 4: Generate or load configuration
    if config_file:
        console.print(f"\n[cyan]Loading configuration from: {config_file}[/cyan]")
        config = load_config(config_file)
        # Estimate memory for display
        from .presets import estimate_memory_usage
        estimated_memory = estimate_memory_usage(
            config.training.mini_batch_size,
            config.training.max_training_steps,
            config.model.max_pixels
        )
    else:
        console.print("\n[cyan]Generating training configuration...[/cyan]")
        presets = get_training_presets(gpu_memory_gb)
        config, estimated_memory = generate_config_interactive(
            model_name=model,
            tasks_count=len(tasks),
            presets=presets,
            output_dir=output_dir,
        )
    
    # Step 5: Save temporary config and display summary
    temp_config_path = Path(".rl_config_temp.json")
    save_config(config, temp_config_path)
    console.print(f"\n[cyan]📝 Configuration saved to: {temp_config_path.absolute()}[/cyan]")
    console.print("[yellow]You can edit this file before starting training.[/yellow]")
    
    # Display configuration summary
    display_config_summary(config, len(tasks), gpu_info, estimated_memory)
    
    # Step 6: Ask for confirmation
    console.print("\n[bold yellow]Options:[/bold yellow]")
    console.print("  • Type [green]'start'[/green] to begin training")
    console.print("  • Type [cyan]'edit'[/cyan] to open config in your editor")
    console.print("  • Type [red]'cancel'[/red] to abort")
    console.print("\n[bold]Your choice:[/bold] ", end="")
    
    while True:
        choice = input().strip().lower()
        
        if choice == "start":
            # Reload config in case it was edited
            config = load_config(temp_config_path)
            break
        elif choice == "edit":
            # Default to nano if EDITOR is not set
            editor = os.environ.get('EDITOR', 'nano')
            
            # Show nano instructions if using nano
            if editor == 'nano':
                console.print("\n[cyan]Opening config in nano editor...[/cyan]")
                console.print("[yellow]Tips:[/yellow]")
                console.print("  • Edit the configuration values as needed")
                console.print("  • Press [bold]Ctrl+O[/bold] then [bold]Enter[/bold] to save")
                console.print("  • Press [bold]Ctrl+X[/bold] to exit")
                console.print("  • Press [bold]Ctrl+C[/bold] to cancel without saving\n")
                input("Press Enter to continue...")
            
            try:
                subprocess.run([editor, str(temp_config_path)], check=True)
                # Reload and display updated config
                config = load_config(temp_config_path)
                estimated_memory = estimate_memory_usage(
                    config.training.mini_batch_size,
                    config.training.max_training_steps,
                    config.model.max_pixels
                )
                display_config_summary(config, len(tasks), gpu_info, estimated_memory)
                console.print("\n[bold]Type 'start' to begin or 'cancel' to abort:[/bold] ", end="")
            except subprocess.CalledProcessError:
                console.print(f"\n[yellow]Editor closed without saving or was cancelled.[/yellow]")
                console.print("[bold]Your choice:[/bold] ", end="")
            except Exception as e:
                console.print(f"\n[red]Failed to open editor: {e}[/red]")
                console.print(f"[yellow]Please edit {temp_config_path} manually and type 'start' when ready.[/yellow]")
                console.print("[bold]Your choice:[/bold] ", end="")
        elif choice == "cancel":
            console.print("[red]Training cancelled[/red]")
            
            # Ask if they want to save the config
            if typer.confirm("Save this configuration for later?", default=True):
                config_path = Path("rl_config.json")
                save_config(config, config_path)
            
            # Clean up temp file
            try:
                temp_config_path.unlink()
            except:
                pass
                
            raise typer.Exit(0)
        else:
            console.print("[red]Invalid choice. Type 'start', 'edit', or 'cancel':[/red] ", end="")
    
    # Step 7: Start vLLM server
    vllm_gpu_index = select_gpu_for_vllm(gpu_info["devices"])
    console.print(f"\n[cyan]Setting up vLLM server on GPU {vllm_gpu_index}...[/cyan]")
    
    start_vllm_server(config.model.base_model, vllm_gpu_index, restart=restart)
    
    # Wait for server to be ready
    server_ready = asyncio.run(wait_for_vllm_server())
    if not server_ready:
        console.print("[red]❌ Failed to start vLLM server[/red]")
        raise typer.Exit(1)
    
    # Step 8: Run training
    console.print("\n[bold green]🎯 Starting RL training...[/bold green]\n")
    
    try:
        # Run the async training function
        asyncio.run(train(config, tasks, verbose=verbose))
        console.print("\n[green]✅ Training completed successfully![/green]")
        
        # Clean up temp config file
        try:
            temp_config_path.unlink()
        except:
            pass
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Training failed: {e}[/red]")
        raise typer.Exit(1)


# Export the command function
__all__ = ["rl_command"]
