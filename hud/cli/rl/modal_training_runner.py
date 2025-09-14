"""
Modal training runner that can be configured dynamically.

This script is used by the API server to launch training jobs with unique app names.
"""
import json
import os
import sys
from pathlib import Path
import logging
import uuid

logger = logging.getLogger(__name__)

import modal

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components from modal_runner
from hud.cli.rl.modal_runner import hud_image, checkpoint_volume, _run_training

# Define all GPU configurations we support
GPU_CONFIGS = {
    # A100 configurations
    "a100-1": "A100-80GB",
    "a100-2": "A100-80GB:2",
    "a100-4": "A100-80GB:4",
    "a100-8": "A100-80GB:8",
    # H100 configurations
    "h100-1": "H100",
    "h100-2": "H100:2",
    "h100-4": "H100:4",
    "h100-8": "H100:8",
}

# Create volumes once
info_volume = modal.Volume.from_name("hud-rl-info", create_if_missing=True)

# App will be created in main() after parsing args
app = None

def create_training_function(model_id: str, base_model: str, vllm_url: str, 
                           output_dir: str, config_file: str, tasks_file: str):
    """The actual training logic, shared across all GPU configurations."""
    logger.info(f"Starting training job for model ID: {model_id}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"vLLM URL: {vllm_url}")
    
    # Handle empty or missing file paths
    if not config_file or not tasks_file:
        raise ValueError(f"Missing required file paths: config_file={config_file}, tasks_file={tasks_file}")
    
    # Read config and tasks from the mounted volume
    config_path = f"/info{config_file}"
    tasks_path = f"/info{tasks_file}"
    
    logger.info(f"config_file param: {config_file}")
    logger.info(f"tasks_file param: {tasks_file}")
    logger.info(f"Reading config from: {config_path}")
    logger.info(f"Reading tasks from: {tasks_path}")
    
    # Check if files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(tasks_path):
        raise FileNotFoundError(f"Tasks file not found at {tasks_path}")
    
    # Read config
    with open(config_path, 'r') as f:
        config_json = f.read()
    
    # Read tasks (JSONL format)
    with open(tasks_path, 'r') as f:
        tasks_content = f.read()
    
    result = _run_training(
        tasks_file=None,
        tasks_content=tasks_content,
        config_json=config_json,
        model=base_model,
        output_dir=output_dir,
        vllm_url=vllm_url,
    )
    
    logger.info(f"Training completed: {result}")
    return result

# Define the training function (will be decorated in main())
def run_training_job(
    model_id: str,
    base_model: str,
    vllm_url: str,
    output_dir: str,
    config_file: str,
    tasks_file: str,
):
    """Run the training job."""
    logger.info(f"Running training job {model_id}")
    return create_training_function(model_id, base_model, vllm_url, 
                                  output_dir, config_file, tasks_file)


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch Modal training job")
    parser.add_argument("--app-name", required=True, help="Unique app name for this training run")
    parser.add_argument("--gpu-type", default="A100", help="GPU type (A100 or H100)")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs to use (1, 2, 4, or 8)")
    parser.add_argument("--model-id", required=True, help="Model ID")
    parser.add_argument("--base-model", required=True, help="Base model name")
    parser.add_argument("--vllm-url", required=True, help="vLLM server URL")
    parser.add_argument("--output-dir", default="/checkpoints/rl_output", help="Output directory")
    parser.add_argument("--config-file", required=True, help="Config file path in volume")
    parser.add_argument("--tasks-file", required=True, help="Tasks file path in volume")
    parser.add_argument("--modal-token-id", required=True, help="Modal token ID")
    parser.add_argument("--modal-token-secret", required=True, help="Modal token secret")
    parser.add_argument("--hud-api-key", required=True, help="HUD API key for the client")
    
    args = parser.parse_args()
    
    # Set Modal credentials
    os.environ["MODAL_TOKEN_ID"] = args.modal_token_id
    os.environ["MODAL_TOKEN_SECRET"] = args.modal_token_secret
    
    # Build the GPU configuration key
    gpu_type_key = args.gpu_type.lower()
    if gpu_type_key == "a100":
        gpu_type_key = "a100"
    elif gpu_type_key == "h100":
        gpu_type_key = "h100"
    else:
        logger.error(f"Unsupported GPU type: {args.gpu_type}. Use A100 or H100.")
        sys.exit(1)
    
    # Validate GPU count
    if args.gpu_count not in [1, 2, 4, 8]:
        logger.error(f"Unsupported GPU count: {args.gpu_count}. Use 1, 2, 4, or 8.")
        sys.exit(1)
    
    # Build config name (e.g., "a100-2" for 2x A100)
    config_name = f"{gpu_type_key}-{args.gpu_count}"
    
    # Get the GPU spec
    if config_name not in GPU_CONFIGS:
        logger.error(f"Configuration {config_name} not found")
        sys.exit(1)
    
    gpu_spec = GPU_CONFIGS[config_name]
    logger.info(f"Using GPU configuration: {config_name} ({gpu_spec})")
    
    # Create the app with the provided name
    app = modal.App(args.app_name)
    
    # Create the function with the specific GPU configuration
    # Create a secret with the HUD API key
    hud_api_secret = modal.Secret.from_dict({"HUD_API_KEY": args.hud_api_key})
    
    training_fn = app.function(
        image=hud_image,
        gpu=gpu_spec,
        scaledown_window=5 * 60,
        timeout=8 * 60 * 60,  # 8 hours
        volumes={
            "/checkpoints": checkpoint_volume,
            "/info": info_volume,
        },
        secrets=[hud_api_secret],
    )(run_training_job)
    
    # Deploy the app and spawn the training job
    logger.info(f"Deploying training app: {args.app_name}")
    logger.info(f"  Model ID: {args.model_id}")
    logger.info(f"  Base Model: {args.base_model}")
    logger.info(f"  GPU Configuration: {config_name} ({gpu_spec})")
    
    try:
        # Deploy the app first
        with modal.enable_output():
            app.deploy()
        
        # Spawn the training job
        call = training_fn.spawn(
            model_id=args.model_id,
            base_model=args.base_model,
            vllm_url=args.vllm_url,
            output_dir=args.output_dir,
            config_file=args.config_file,
            tasks_file=args.tasks_file,
        )
        
        # Exit immediately - the training will continue in Modal
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error deploying/spawning training: {e}")
        sys.exit(1)
