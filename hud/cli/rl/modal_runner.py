"""Modal runner for HUD RL training - integrates with existing CLI."""

import modal
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Modal app
app = modal.App("hud-rl-training")

# Define volumes
checkpoint_volume = modal.Volume.from_name("hud-rl-checkpoints", create_if_missing=True)
dataset_volume = modal.Volume.from_name("hud-rl-datasets", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("hud-vllm-cache", create_if_missing=True)

# Base image with HUD - Option 1: debian_slim with PyTorch (current approach)
hud_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "curl")
    .pip_install(
        "torch==2.4.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install numpy explicitly first to avoid the torch numpy warning
    .pip_install("numpy")
    # Install hud-python from the GitHub fork
    .pip_install("git+https://github.com/lorenss-m/hud-python.git#egg=hud-python[rl]")
    # Skip flash-attn as it requires nvcc/CUDA dev tools during build
    # It's optional anyway - just provides better performance
    .env({
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

# Option 2: Official NVIDIA CUDA base image (uncomment to use)
# This includes full CUDA toolkit and might help with packages like flash-attn
# hud_image = (
#     modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.12")
#     .entrypoint([])  # Remove verbose logging from base image
#     .apt_install("git", "build-essential", "curl")
#     .pip_install("numpy")
#     .pip_install(
#         "torch==2.4.0",
#         index_url="https://download.pytorch.org/whl/cu121",
#     )
#     .pip_install("git+https://github.com/lorenss-m/hud-python.git#egg=hud-python[rl]")
#     # With CUDA devel image, you could potentially install flash-attn:
#     # .pip_install("ninja", "packaging", "wheel")  # Required for building
#     # .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
#     .env({
#         "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
#         "TOKENIZERS_PARALLELISM": "false",
#     })
# )


# Define functions for each GPU type
@app.function(
    image=hud_image,
    gpu="H100:2",
    timeout=24 * 60 * 60,
    volumes={
        "/checkpoints": checkpoint_volume,
        "/data": dataset_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("hud-api-key")  # For telemetry and job tracking
    ],
)
def run_rl_training_h100(
    tasks_file: Optional[str],
    tasks_content: Optional[str],
    config_json: str,
    model: Optional[str] = None,
    output_dir: str = "/checkpoints",
    verbose: bool = False,
):
    """Run RL training on H100 GPUs."""
    return _run_training(tasks_file, tasks_content, config_json, model, output_dir, verbose)


@app.function(
    image=hud_image,
    gpu="A100:2",
    timeout=24 * 60 * 60,
    volumes={
        "/checkpoints": checkpoint_volume,
        "/data": dataset_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("hud-api-key")  # For telemetry and job tracking
    ],
)
def run_rl_training_a100(
    tasks_file: Optional[str],
    tasks_content: Optional[str],
    config_json: str,
    model: Optional[str] = None,
    output_dir: str = "/checkpoints",
    verbose: bool = False,
):
    """Run RL training on A100 GPUs."""
    return _run_training(tasks_file, tasks_content, config_json, model, output_dir, verbose)


def _run_training(
    tasks_file: Optional[str],
    tasks_content: Optional[str],
    config_json: str,
    model: Optional[str] = None,
    output_dir: str = "/checkpoints",
    verbose: bool = False,
):
    """
    Common training logic for all GPU types.
    """
    import sys
    import subprocess
    import json
    from pathlib import Path
    
    # Write config to a temporary file
    config_path = Path("/tmp/rl_config.json")
    config_path.write_text(config_json)
    
    # Handle tasks file
    if tasks_content:
        # Write tasks content to a temporary file
        tasks_path = Path("/tmp/tasks.jsonl")
        tasks_path.write_text(tasks_content)
        tasks_file_to_use = str(tasks_path)
    else:
        tasks_file_to_use = tasks_file
    
    # Build the command exactly as the local CLI would
    # But we override some settings for Modal environment
    cmd = [sys.executable, "-m", "hud.cli", "rl"]
    
    # Add the tasks file
    if tasks_file_to_use:
        cmd.append(tasks_file_to_use)
    
    # Add all the options
    if model:
        cmd.extend(["--model", model])
    
    # Use the config we wrote
    cmd.extend(["--config", str(config_path)])
    
    cmd.extend(["--output-dir", output_dir])
    
    if verbose:
        cmd.append("--verbose")
    
    # Modal always has 2 GPUs, so we'll use GPU 1 for vLLM and GPU 0 for training
    # (no DDP for simplicity, but could be enabled)
    # cmd.extend(["--vllm-gpu", "1"])
    # cmd.extend(["--no-ddp"])  # For now, use single GPU training
    
    # Run the training
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    # Ensure checkpoints are persisted
    checkpoint_volume.commit()
    
    return result.returncode


def launch_on_modal(
    tasks_file: Optional[str],
    model: Optional[str],
    config_file: Optional[Path],
    output_dir: str,
    verbose: bool,
    modal_gpu: str = "H100",
    no_ddp: bool = False,
    ddp_gpus: Optional[str] = None,
    vllm_gpu: Optional[int] = None,
) -> None:
    """
    Launch RL training on Modal. Called from the CLI when --modal flag is used.
    """
    from rich.console import Console
    from hud.utils.design import design
    import json
    from pathlib import Path
    
    console = Console()
    
    console.print("\n[bold cyan]🚀 Launching HUD RL Training on Modal[/bold cyan]\n")
    
    # Check if HUD API key is set up in Modal
    console.print("[yellow]Checking Modal setup...[/yellow]")
    console.print("[dim]Note: HUD API key must be set as a Modal secret[/dim]")
    console.print("[dim]If you haven't set it up yet, run:[/dim]")
    console.print("[dim]  modal secret create hud-api-key HUD_API_KEY=your_key_here[/dim]\n")
    
    # GPU Selection
    gpu_choice = design.select(
        "Select GPU type for Modal training (2 GPUs will be allocated):",
        choices=[
            {"name": "H100 (80GB) - Fastest, recommended for large batches", "value": "H100"},
            {"name": "A100 (80GB) - Good performance, more availability", "value": "A100"},
        ],
        default="H100",
    )
    
    # Import the necessary modules to run configuration locally
    from hud.rl.utils import load_tasks
    from hud.rl.config import Config
    from .config import generate_config_interactive, load_config
    from .presets import get_training_presets, estimate_memory_usage
    
    # Load tasks locally to validate and count them
    console.print(f"\n[cyan]Loading tasks from: {tasks_file or 'auto-detect'}[/cyan]")
    
    # Auto-detect tasks file if not provided
    if not tasks_file:
        from pathlib import Path
        possible_files = ["tasks.json", "tasks.jsonl", "browser_2048_tasks.jsonl"]
        for f in possible_files:
            if Path(f).exists():
                tasks_file = f
                console.print(f"[green]Auto-detected tasks file: {f}[/green]")
                break
        
        if not tasks_file:
            console.print("[red]❌ No tasks file specified or auto-detected[/red]")
            return
    
    tasks = load_tasks(tasks_file)
    console.print(f"[green]✅ Loaded {len(tasks)} tasks[/green]")
    
    # Model selection (if not provided)
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
    
    # Generate or load configuration
    if config_file and config_file.exists():
        console.print(f"\n[cyan]Loading configuration from: {config_file}[/cyan]")
        config = load_config(config_file)
        estimated_memory = estimate_memory_usage(
            config.training.mini_batch_size,
            config.training.max_training_steps,
            config.model.max_pixels
        )
    else:
        console.print("\n[cyan]Generating training configuration...[/cyan]")
        # Get GPU memory for the selected type
        gpu_memory_gb = 80.0  # Both H100 and A100 have 80GB variants
        
        presets = get_training_presets(gpu_memory_gb)
        config, estimated_memory = generate_config_interactive(
            model_name=model,
            tasks_count=len(tasks),
            presets=presets,
            output_dir=output_dir,
        )
    
    # Show Modal-specific configuration
    console.print("\n[yellow]Modal Configuration:[/yellow]")
    console.print(f"  • GPU Type: {gpu_choice} x2 (1 for vLLM, 1 for training)")
    console.print(f"  • Tasks: {tasks_file} ({len(tasks)} tasks)")
    console.print(f"  • Model: {config.model.base_model}")
    console.print(f"  • Estimated memory: {estimated_memory:.1f} GB")
    console.print(f"  • Output: Modal volume 'hud-rl-checkpoints'")
    console.print("\n[dim]Note: Checkpoints will be saved to Modal storage[/dim]")
    console.print("[dim]Download with: modal volume get hud-rl-checkpoints /local/path[/dim]")
    
    if not console.input("\n[bold]Start training on Modal? (y/N):[/bold] ").lower().startswith('y'):
        console.print("[red]Cancelled[/red]")
        return
    
    # Convert config to JSON string for passing to Modal
    config_json = json.dumps(config.to_dict())
    
    # Read tasks file content if it's a local file
    tasks_content = None
    if tasks_file and Path(tasks_file).exists():
        console.print(f"[dim]Reading tasks from {tasks_file}...[/dim]")
        tasks_content = Path(tasks_file).read_text()
        # Pass None for tasks_file since we're passing content
        tasks_file_remote = None
    else:
        # It might be a HuggingFace dataset name
        tasks_file_remote = tasks_file
    
    # Select the appropriate function based on GPU choice
    if gpu_choice == "H100":
        run_function = run_rl_training_h100
    else:  # A100
        run_function = run_rl_training_a100
    
    console.print("\n[cyan]Submitting job to Modal...[/cyan]")
    console.print("[dim]This may take a moment to provision GPUs...[/dim]\n")
    
    # Run on Modal
    with app.run():
        console.print("\n[cyan]Starting Modal execution...[/cyan]\n")
        
        # Use enable_output() to see logs in real-time
        with modal.enable_output():
            result = run_function.remote(
                tasks_file=tasks_file_remote,
                tasks_content=tasks_content,
                config_json=config_json,
                model=model,
                output_dir=output_dir,
                verbose=verbose,
            )
    
    if result == 0:
        console.print("\n[green]✅ Training completed successfully![/green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  1. Download checkpoints: [white]modal volume get hud-rl-checkpoints ./checkpoints[/white]")
        console.print("  2. View logs: [white]modal logs -f[/white]")
        console.print("  3. List volumes: [white]modal volume ls hud-rl-checkpoints[/white]")
    else:
        console.print(f"\n[red]❌ Training failed with exit code {result}[/red]")
        console.print("\n[yellow]Debug tips:[/yellow]")
        console.print("  • Check logs: [white]modal logs -f[/white]")
        console.print("  • Ensure Modal secrets are set (if using private models)")
        console.print("  • Verify tasks file format")


# For direct testing
@app.local_entrypoint()
def test():
    """Test entry point for modal run."""
    import json
    from pathlib import Path
    
    # Example config for testing
    test_config = {
        "model": {
            "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "lora_r": 32,
            "lora_alpha": 64,
        },
        "training": {
            "training_steps": 10,
            "batch_size": 6,
            "group_size": 3,
            "mini_batch_size": 2,
            "lr": 1e-5,
        },
        "actor": {
            "max_steps_per_episode": 4,
            "max_parallel_episodes": 6,
            "vllm_base_url": "http://127.0.0.1:8000/v1",
            "vllm_api_key": "token-abc123",
        },
    }
    
    # Read tasks file if it exists locally
    tasks_file = "browser_2048_tasks.jsonl"
    tasks_content = None
    if Path(tasks_file).exists():
        tasks_content = Path(tasks_file).read_text()
        tasks_file = None  # Don't pass filename if we have content
    
    run_rl_training_h100.remote(
        tasks_file=tasks_file,
        tasks_content=tasks_content,
        config_json=json.dumps(test_config),
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        verbose=True,
    )
