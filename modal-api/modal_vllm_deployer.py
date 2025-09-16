"""
Dynamic vLLM server deployment for Modal API.

This module handles programmatic deployment of vLLM servers with unique names for each model.
"""
import os
import subprocess
import sys
from hud.settings import settings
from pathlib import Path
from typing import Optional

import modal


def deploy_vllm_for_model(model_id: str, model_name: str, base_model: str, gpu_type: str = "A100") -> str:
    # Create unique app name using model ID
    app_name = f"hud-vllm-{model_id[:8]}"
    
    # Set up environment variables for the deployment
    env = {
        **os.environ,
        "MODAL_APP_NAME": app_name,
        "VLLM_MODEL_NAME": base_model,
        "VLLM_GPU_TYPE": gpu_type,
        "MODAL_TOKEN_ID": settings.modal_token_id,
        "MODAL_TOKEN_SECRET": settings.modal_token_secret,
    }
    
    # Path to the vLLM server script
    vllm_server_path = Path(__file__).parent.parent / "hud" / "cli" / "rl" / "modal_vllm_server.py"
    
    try:
        # First, check if app already exists and stop it if needed
        subprocess.run(
            ["modal", "app", "stop", app_name],
            capture_output=True,
            env=env
        )
        # Ignore errors - app might not exist
        
        # Deploy using the Python script directly (which will call app.deploy())
        result = subprocess.run(
            [sys.executable, str(vllm_server_path)],
            capture_output=True,
            text=True,
            env=env,
            check=True
        )
        
        print(f"Deployment output:\n{result.stdout}")
        
        if result.stderr:
            print(f"Deployment stderr:\n{result.stderr}")
        
        # Get the deployed function URL
        serve_fn = modal.Function.from_name(app_name, "serve_vllm")
        vllm_url = serve_fn.get_web_url()
        
        return vllm_url
        
    except subprocess.CalledProcessError as e:
        print(f"Deployment failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        raise Exception(f"Modal deployment failed: {e.stderr}")
    except Exception as e:
        raise Exception(f"Deployment error: {str(e)}")