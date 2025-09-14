"""
Modal training runner that can be configured via command line arguments.

This script is used by the API server to launch training jobs with unique app names.
"""
import json
import os
import sys
import argparse
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

import modal

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Parse command line arguments
parser = argparse.ArgumentParser(description="Launch Modal training job")
parser.add_argument("--app-name", required=True, help="Modal app name")
parser.add_argument("--gpu-type", default="A100", help="GPU type (A100 or H100)")
parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--model-id", required=True, help="Model ID")
parser.add_argument("--base-model", required=True, help="Base model name")
parser.add_argument("--vllm-url", required=True, help="vLLM server URL")
parser.add_argument("--output-dir", default="/checkpoints/rl_output", help="Output directory")
parser.add_argument("--config-file", required=True, help="Config file path in volume")
parser.add_argument("--tasks-file", required=True, help="Tasks file path in volume")
parser.add_argument("--modal-token-id", required=True, help="Modal token ID")
parser.add_argument("--modal-token-secret", required=True, help="Modal token secret")

args = parser.parse_args()

# Set Modal credentials
os.environ["MODAL_TOKEN_ID"] = args.modal_token_id
os.environ["MODAL_TOKEN_SECRET"] = args.modal_token_secret

# Volume will be mounted in the container
info_volume = modal.Volume.from_name("hud-rl-info", create_if_missing=True)

# Create the app with configurable name
app = modal.App(args.app_name)

# Import components from modal_runner
from hud.cli.rl.modal_runner import hud_image, checkpoint_volume, _run_training

# Configure GPU
if args.gpu_count > 1:
    gpu_config = f"{args.gpu_type}:{args.gpu_count}" if args.gpu_type == "H100" else f"{args.gpu_type}-80GB:{args.gpu_count}"
else:
    gpu_config = f"{args.gpu_type}-80GB" if args.gpu_type == "A100" else args.gpu_type

@app.function(
    image=hud_image,
    gpu=gpu_config,
    scaledown_window=5 * 60,
    timeout=8 * 60 * 60,  # 8 hours
    volumes={
        "/checkpoints": checkpoint_volume,
        "/info": info_volume,
    },
)
def run_training_job(
    model_id: str,
    base_model: str,
    vllm_url: str,
    output_dir: str,
    config_file: str,
    tasks_file: str,
):
    """Run the training for this specific model."""
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


if __name__ == "__main__":
    # Deploy the app and spawn the training job
    logger.info(f"Deploying training app: {args.app_name}")
    logger.info(f"  Model ID: {args.model_id}")
    logger.info(f"  Base Model: {args.base_model}")
    logger.info(f"  GPU Type: {args.gpu_type}")
    
    try:
        # Deploy the app first
        with modal.enable_output():
            app.deploy()
        
        # Get the function from the deployed app and spawn it
        training_fn = modal.Function.lookup(args.app_name, "run_training_job")
        
        # Use spawn() which returns immediately without waiting
        # The function will continue running in Modal even after this script exits
        call = training_fn.spawn(
            model_id=args.model_id,
            base_model=args.base_model,
            vllm_url=args.vllm_url,
            output_dir=args.output_dir,
            config_file=args.config_file,
            tasks_file=args.tasks_file,
        )
        
        logger.info(f"Training job spawned on app: {args.app_name}")
        logger.info(f"Call ID: {call.object_id}")
        
        # Exit immediately - the training will continue in Modal
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error deploying/spawning training: {e}")
        sys.exit(1)
