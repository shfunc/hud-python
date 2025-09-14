"""
Modal training runner that can be configured via environment variables.

This script is used by the API server to launch training jobs with unique app names.
"""
import os
import sys
from pathlib import Path
from typing import Optional

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

if CONFIG_FILE and os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        CONFIG_JSON = f.read()
else:
    CONFIG_JSON = "{}"

if TASKS_FILE and os.path.exists(TASKS_FILE):
    with open(TASKS_FILE, 'r') as f:
        TASKS_JSON = f.read()
else:
    TASKS_JSON = "[]"

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
    },
)
def run_training_job():
    """Run the training for this specific model."""
    print(f"Starting training job for model ID: {MODEL_ID}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"vLLM URL: {VLLM_URL}")
    
    result = _run_training(
        tasks_file=None,
        tasks_content=TASKS_JSON,
        config_json=CONFIG_JSON,
        model=BASE_MODEL,
        output_dir=OUTPUT_DIR,
        vllm_url=VLLM_URL,
    )
    
    print(f"Training completed: {result}")
    return result


if __name__ == "__main__":
    # Deploy the app and spawn the training job
    print(f"Deploying training app: {APP_NAME}")
    print(f"  Model ID: {MODEL_ID}")
    print(f"  Base Model: {BASE_MODEL}")
    print(f"  GPU Type: {GPU_TYPE}")
    
    try:
        # Deploy the app first
        with modal.enable_output():
            app.deploy()
        
        # Get the function from the deployed app and spawn it
        training_fn = modal.Function.lookup(APP_NAME, "run_training_job")
        
        # Use spawn() which returns immediately without waiting
        # The function will continue running in Modal even after this script exits
        call = training_fn.spawn()
        
        print(f"Training job spawned on app: {APP_NAME}")
        print(f"Call ID: {call.object_id}")
        
        # Exit immediately - the training will continue in Modal
        sys.exit(0)
        
    except Exception as e:
        print(f"Error deploying/spawning training: {e}")
        sys.exit(1)
