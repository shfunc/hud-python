"""
Dynamic training job deployment for Modal API.

This module handles programmatic deployment of training jobs with unique names for each model.
"""
import os
import subprocess
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional
import io

import modal
from hud.settings import settings
import logging

logger = logging.getLogger(__name__)

info_volume = modal.Volume.from_name("hud-rl-info", create_if_missing=True)

def launch_training_for_model(
    model_id: str,
    model_name: str,
    base_model: str,
    gpu_type: str,
    gpu_count: int,
    config_json: str,
    tasks_json: str,
    vllm_url: str,
    output_dir: str,
    hud_api_key: str
) -> str:
    """Launch a training job for a specific model.
    
    Args:
        model_id: Unique ID for the model (used in app name)
        model_name: Human-readable model name
        base_model: Base model to train
        gpu_type: GPU type to use ("A100" or "H100")
        gpu_count: Number of GPUs to use
        config_json: JSON string of training config
        tasks_json: JSON string of training tasks
        vllm_url: URL of the vLLM server
        output_dir: Directory to save checkpoints
    
    Returns:
        App name of the training deployment
    """
    # Upload config and tasks to the Modal volume
    config_path = f"/training/{model_id}/config.json"
    tasks_path = f"/training/{model_id}/tasks.jsonl"
    
    try:
        # First, try to clean up any existing files for this model
        try:
            # Use subprocess to remove existing directory if it exists
            subprocess.run(
                ["modal", "volume", "rm", "-r", "hud-rl-info", f"/training/{model_id}"],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "MODAL_TOKEN_ID": settings.modal_token_id,
                    "MODAL_TOKEN_SECRET": settings.modal_token_secret
                }
            )
            logger.info(f"Cleaned up existing files for model {model_id}")
        except Exception as e:
            logger.debug(f"No existing files to clean up: {e}")
        
        # Upload files to Modal volume
        with info_volume.batch_upload() as batch:
            # Upload config JSON
            batch.put_file(io.BytesIO(config_json.encode()), config_path)
            
            # Convert tasks to JSONL format and upload
            tasks_list = json.loads(tasks_json)
            tasks_jsonl = "\n".join(json.dumps(task) for task in tasks_list)
            batch.put_file(io.BytesIO(tasks_jsonl.encode()), tasks_path)
        
        logger.info(f"Uploaded config to volume: {config_path}")
        logger.info(f"Uploaded tasks to volume: {tasks_path}")
        # Path to the training runner script
        training_runner_path = Path(__file__).parent.parent / "hud" / "cli" / "rl" / "modal_training_runner.py"
        
        # Generate unique app name here in the deployer
        import uuid
        unique_id = str(uuid.uuid4())[:8]  # Just first 8 chars to keep name short
        gpu_config = f"{gpu_type.lower()}-{gpu_count}"
        app_name = f"hud-training-{model_id[:8]}-{gpu_config}-{unique_id}"
        
        # Build command with arguments including app name
        launch_command = [
            sys.executable,
            str(training_runner_path),
            "--app-name", app_name,
            "--gpu-type", gpu_type,
            "--gpu-count", str(gpu_count),
            "--model-id", model_id,
            "--base-model", base_model,
            "--config-file", config_path,
            "--tasks-file", tasks_path,
            "--vllm-url", vllm_url,
            "--output-dir", output_dir,
            "--modal-token-id", settings.modal_token_id,
            "--modal-token-secret", settings.modal_token_secret,
            "--hud-api-key", hud_api_key,
        ]
        
        # Launch the training runner script
        log_file = f"/tmp/modal-training-{model_id[:8]}.log"
        print(f"Running training job with command: {launch_command}")
        print(f"Logging to: {log_file}")
        
        
        with open(log_file, 'w') as log:
            # Launch the process completely detached
            process = subprocess.Popen(
                launch_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent process group
            )
        
        return app_name
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Training deployment failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        raise Exception(f"Modal training deployment failed: {e.stderr}")
    except Exception as e:
        raise Exception(f"Training deployment error: {str(e)}")
