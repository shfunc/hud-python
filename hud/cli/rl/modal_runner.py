"""
Modal runner for HUD RL training.

This module provides functionality to run HUD RL training on Modal's cloud infrastructure
with GPU support (H100/A100).
"""
from __future__ import annotations

from pathlib import Path

import modal
import typer
from rich.console import Console
from rich.prompt import Prompt

# Modal configuration
app = modal.App("hud-rl-training")

console = Console()

# Volumes for persistent storage
checkpoint_volume = modal.Volume.from_name("hud-rl-checkpoints", create_if_missing=True)
dataset_volume = modal.Volume.from_name("hud-rl-datasets", create_if_missing=True)

# Get the local hud-python path
current_file = Path(__file__)
local_hud_path = current_file.parent.parent.parent.parent  # Go up to hud-python root

# Image for RL training
hud_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
    .apt_install("git", "build-essential", "curl", "ca-certificates")
    .pip_install("git+https://github.com/lorenss-m/hud-python.git#egg=hud-python[rl]", force_build=True)
    .env({
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
)


# Training Functions
@app.function(
    image=hud_image,
    gpu="H100",
    timeout=24 * 60 * 60,  # 24 hours
    volumes={
        "/checkpoints": checkpoint_volume,
        "/datasets": dataset_volume,
    },
    secrets=[modal.Secret.from_name("hud-api-key")],
)
def run_rl_training_h100(
    tasks_file: str | None,
    tasks_content: str | None,
    config_json: str,
    model: str | None = None,
    output_dir: str = "/checkpoints",
    vllm_url: str = "http://localhost:8000/v1",
):
    """Run RL training on H100 GPUs."""
    return _run_training(
        tasks_file, tasks_content, config_json, model, output_dir, vllm_url
    )


@app.function(
    image=hud_image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,  # 24 hours
    volumes={
        "/checkpoints": checkpoint_volume,
        "/datasets": dataset_volume,
    },
    secrets=[modal.Secret.from_name("hud-api-key")],
)
def run_rl_training_a100(
    tasks_file: str | None,
    tasks_content: str | None,
    config_json: str,
    model: str | None = None,
    output_dir: str = "/checkpoints",
    vllm_url: str = "http://localhost:8000/v1",
):
    """Run RL training on A100 GPUs."""
    return _run_training(
        tasks_file, tasks_content, config_json, model, output_dir, vllm_url
    )


def _run_training(
    tasks_file: str | None,
    tasks_content: str | None,
    config_json: str,
    model: str | None = None,
    output_dir: str = "/checkpoints",
    vllm_url: str = "http://localhost:8000/v1",
):
    """Common training logic for all GPU types."""
    import json
    from pathlib import Path

    from hud.cli.rl import rl_command
    
    # Write config file with unique name to avoid conflicts
    import uuid
    config_id = str(uuid.uuid4())[:8]
    config_path = Path(f"/tmp/rl_config_{config_id}.json")
    config_path.write_text(config_json)
    
    # Update config to use provided vLLM URL
    config_data = json.loads(config_json)
    
    # Ensure actor section exists before setting vllm_base_url
    if "actor" not in config_data:
        config_data["actor"] = {}
    
    if vllm_url:
        config_data["actor"]["vllm_base_url"] = vllm_url
    
    # Ensure model is set in config
    if "model" not in config_data:
        config_data["model"] = {}
    
    # Use provided model or ensure we have a default
    if model:
        config_data["model"]["base_model"] = model
    elif not config_data["model"].get("base_model"):
        # Default to Qwen if no model specified
        config_data["model"]["base_model"] = "Qwen/Qwen2.5-VL-3B-Instruct"

    config_data["out_dir"] = output_dir or "/checkpoints"
    
    # Debug: Print the final config
    print(f"Final config model: {config_data['model'].get('base_model')}")
    
    config_path.write_text(json.dumps(config_data, indent=2))
    
    # Write tasks file with unique name if content provided
    if tasks_content:
        tasks_path = Path(f"/tmp/tasks_{config_id}.jsonl")
        tasks_path.write_text(tasks_content)
        tasks_file_to_use = str(tasks_path)
    else:
        tasks_file_to_use = tasks_file
    
    print(f"Starting RL training with vLLM at: {vllm_url}")
    print(f"Tasks file: {tasks_file_to_use}")
    print(f"Model: {model}")
    print(f"Config: {config_path}")
    print(f"Output dir: {output_dir}")
    
    # Launch DDP training with torchrun across all visible GPUs in the container
    from hud.cli.rl import launch_ddp_training
    import torch

    visible_gpus = list(range(max(1, torch.cuda.device_count())))

    launch_ddp_training(
        training_gpus=visible_gpus,
        tasks_file=tasks_file_to_use,
        config_path=config_path,
        verbose=True,
    )
    
    # Commit checkpoint volume
    checkpoint_volume.commit()
    
    return 0


def get_vllm_server_url() -> str | None:
    """Get the URL of the deployed vLLM server."""
    try:
        from modal import Function
        # Get the deployed function
        serve_vllm = Function.from_name("hud-vllm-server", "serve_vllm")
        url = serve_vllm.get_web_url()
        if url:
            console.print(f"[green]Found existing vLLM server at: {url}[/green]")
            return url
    except Exception as e:
        console.print(f"[yellow]No existing vLLM server found: {e}[/yellow]")
    
    return None


def deploy_vllm_server(gpu_type: str, model: str) -> str:
    """Deploy the vLLM server programmatically using Modal API."""
    import importlib.util
    import sys
    
    console.print(f"[cyan]Deploying vLLM server on {gpu_type}...[/cyan]")
    console.print("[yellow]Note: vLLM server always uses Qwen/Qwen2.5-3B-Instruct[/yellow]")
    
    # Import the vLLM server module
    vllm_server_path = Path(__file__).parent / "modal_vllm_server.py"
    spec = importlib.util.spec_from_file_location("modal_vllm_server", vllm_server_path)
    vllm_module = importlib.util.module_from_spec(spec)
    sys.modules["modal_vllm_server"] = vllm_module
    spec.loader.exec_module(vllm_module)
    
    # Deploy the app programmatically
    try:
        with modal.enable_output():
            vllm_module.app.deploy()
        
        # Get the URL after deployment
        url = get_vllm_server_url()
        
        if url:
            console.print(f"[green]✅ vLLM server deployed at: {url}[/green]")
            return url
        else:
            console.print("[red]Could not get vLLM server URL after deployment[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Failed to deploy vLLM server: {e}[/red]")
        raise typer.Exit(1)


def launch_on_modal(
    tasks_file: str,
    config_json: str,
    model: str | None = None,
    output_dir: str = "checkpoints",
    modal_gpu: str = "",
    **kwargs  # Accept additional args that we don't use
):
    """Launch RL training on Modal with optional vLLM server."""
    import json
    from pathlib import Path
    
    # Extract model from config if not provided
    if not model:
        config_data = json.loads(config_json)
        model = config_data.get("model", {}).get("base_model", "Qwen/Qwen2.5-3B-Instruct")
    
    # Read tasks file content if it's a local file
    tasks_path = Path(tasks_file)
    if tasks_path.exists():
        tasks_content = tasks_path.read_text()
        console.print(f"[cyan]Read {len(tasks_content.splitlines())} tasks from {tasks_file}[/cyan]")
    else:
        # Assume it's a HuggingFace dataset
        tasks_content = None
        console.print(f"[cyan]Using HuggingFace dataset: {tasks_file}[/cyan]")
    
    # Determine GPU type for training
    if modal_gpu:
        gpu_choice = modal_gpu
    else:
        gpu_choice = Prompt.ask(
            "Select GPU type for training",
            choices=["H100", "A100"],
            default="A100"
        )
    
    # Check for existing vLLM server
    vllm_url = get_vllm_server_url()
    
    if vllm_url:
        use_existing = Prompt.ask(
            "Use existing vLLM server?",
            choices=["yes", "no"],
            default="yes"
        )
        
        if use_existing == "no":
            # Deploy new one
            vllm_url = deploy_vllm_server("A100", model or "Qwen/Qwen2.5-3B-Instruct")
    else:
        # No existing server, deploy one
        vllm_url = deploy_vllm_server("A100", model or "Qwen/Qwen2.5-3B-Instruct")
    
    # Run training
    console.print(f"[cyan]Starting RL training on Modal with {gpu_choice}...[/cyan]")
    console.print(f"[cyan]vLLM server URL: {vllm_url}[/cyan]")
    
    with app.run(), modal.enable_output():
        if gpu_choice == "H100":
            result = run_rl_training_h100.remote(
                tasks_file=None,
                tasks_content=tasks_content,
                config_json=config_json,
                model=model,
                output_dir=output_dir,
                vllm_url=f"{vllm_url}/v1",
            )
        else:
            result = run_rl_training_a100.remote(
                tasks_file=None,
                tasks_content=tasks_content,
                config_json=config_json,
                model=model,
                output_dir=output_dir,
                vllm_url=f"{vllm_url}/v1",
            )
    
    console.print(f"[green]Training completed with result: {result}[/green]")
    return result
