"""
Modal training runner that can be configured via environment variables.

This script is used by the API server to launch training jobs with unique app names.
"""
import json
import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

import modal

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configuration from environment variables
APP_NAME = os.environ.get("MODAL_APP_NAME", "hud-training-default")
GPU_TYPE = os.environ.get("TRAINING_GPU_TYPE", "A100")
MODEL_ID = os.environ.get("TRAINING_MODEL_ID", "")
BASE_MODEL = os.environ.get("TRAINING_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
VLLM_URL = os.environ.get("TRAINING_VLLM_URL", "")
OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", "/checkpoints/rl_output")

# Read config and tasks from files
CONFIG_FILE = os.environ.get("TRAINING_CONFIG_FILE", "")
TASKS_FILE = os.environ.get("TRAINING_TASKS_FILE", "")

# Volume will be mounted in the container
info_volume = modal.Volume.from_name("hud-rl-info", create_if_missing=True)

# These will be read inside the Modal function where the volume is mounted
CONFIG_JSON = CONFIG_FILE
TASKS_JSON = TASKS_FILE

# Create the app with configurable name
app = modal.App(APP_NAME)

# Import components from modal_runner
from hud.cli.rl.modal_runner import hud_image, checkpoint_volume, _run_training

# Configure GPU
gpu_config = f"{GPU_TYPE}:2" if GPU_TYPE == "H100" else f"{GPU_TYPE}-80GB"

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
def run_training_job():
    """Run the training for this specific model."""
    logger.info(f"Starting training job for model ID: {MODEL_ID}")
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"vLLM URL: {VLLM_URL}")
    
    # Read config and tasks from the mounted volume
    config_path = f"/info{CONFIG_FILE}"
    tasks_path = f"/info{TASKS_FILE}"
    
    logger.info(f"Reading config from: {config_path}")
    logger.info(f"Reading tasks from: {tasks_path}")
    
    # Read config
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    
    # Read tasks (JSONL format)
    tasks_content = None
    if os.path.exists(tasks_path):
        with open(tasks_path, 'r') as f:
            tasks_content = json.load(f)
    
    result = _run_training(
        tasks_file=None,
        tasks_content=tasks_content,
        config_json=config_json,
        model=BASE_MODEL,
        output_dir=OUTPUT_DIR,
        vllm_url=VLLM_URL,
    )
    
    logger.info(f"Training completed: {result}")
    return result


if __name__ == "__main__":
    # Deploy the app and spawn the training job
    logger.info(f"Deploying training app: {APP_NAME}")
    logger.info(f"  Model ID: {MODEL_ID}")
    logger.info(f"  Base Model: {BASE_MODEL}")
    logger.info(f"  GPU Type: {GPU_TYPE}")
    
    try:
        # Deploy the app first
        with modal.enable_output():
            app.deploy()
        
        # Get the function from the deployed app and spawn it
        training_fn = modal.Function.lookup(APP_NAME, "run_training_job")
        
        # Use spawn() which returns immediately without waiting
        # The function will continue running in Modal even after this script exits
        call = training_fn.spawn()
        
        logger.info(f"Training job spawned on app: {APP_NAME}")
        logger.info(f"Call ID: {call.object_id}")
        
        # Exit immediately - the training will continue in Modal
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error deploying/spawning training: {e}")
        sys.exit(1)
