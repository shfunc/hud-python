"""
Modal vLLM server deployment for HUD RL training.

This is deployed separately and persists to serve multiple training runs.
Deploy with: modal deploy modal_vllm_server.py
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

# Configuration from environment variables
APP_NAME = os.environ.get("MODAL_APP_NAME", "hud-vllm-server")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
GPU_TYPE = os.environ.get("VLLM_GPU_TYPE", "A100")

# Create the app with configurable name
app = modal.App(APP_NAME)

# Build the image - always install vLLM first
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1")
    .env({
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_LOGGING_LEVEL": "INFO",
        "CUDA_LAUNCH_BLOCKING": "1",
    })
)

# Add chat template if available (at build time)
# This happens when the module is imported during deployment
if __name__ != "__main__":  # Only during import, not direct execution
    try:
        chat_template_path = Path(__file__).parent.parent.parent / "rl" / "chat_template.jinja"
        if chat_template_path.exists():
            vllm_image = vllm_image.add_local_file(
                chat_template_path,
                "/root/chat_template.jinja",
                copy=True  # Copy at build time
            )
            print("Chat template added to image")
    except Exception as e:
        print(f"Chat template not added: {e}")


# Volumes for model cache and checkpoints
vllm_cache_volume = modal.Volume.from_name("hud-vllm-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("hud-rl-checkpoints", create_if_missing=True)

# vLLM Server Function
@app.function(
    image=vllm_image,
    gpu=GPU_TYPE + ("-80GB" if GPU_TYPE == "A100" else ""),  # Use configured GPU
    min_containers=1,  # Keep one instance always running
    scaledown_window=30 * 60,  # Stay up for 3 hours without requests
    timeout=24 * 60 * 60,  # 24 hour timeout (max for training)
    volumes={
        "/root/.cache/vllm": vllm_cache_volume,
        "/checkpoints": checkpoint_volume,  # Share checkpoints with training
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve_vllm():
    """Serve vLLM."""
    import os
    from pathlib import Path
    
    # Use the configured model (from environment or default)
    model = MODEL_NAME
    
    # Build vLLM args inline (same as get_vllm_args)
    vllm_args = [
        "serve", model,
        "--api-key", "token-abc123",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
        "--max-model-len", "16384",
        "--enable-lora",
        "--max-lora-rank", "64",
        "--max-cpu-loras", "4",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--disable-log-requests",
        "--dtype", "auto",
    ]
    
    # Try to add chat template if it exists in the image
    template_path = "/root/chat_template.jinja"
    if Path(template_path).exists():
        vllm_args.extend(["--chat-template", template_path])
        print(f"Using chat template from: {template_path}")
    else:
        print("Using model's default chat template")
    
    cmd = ["vllm"] + vllm_args
    
    print(f"Starting vLLM server with model: {model}")
    print(f"Command: {' '.join(cmd)}")
    print("Checkpoint directory mounted at: /checkpoints")
    
    # List existing checkpoints for debugging
    if os.path.exists("/checkpoints"):
        checkpoints = os.listdir("/checkpoints")
        print(f"Existing checkpoints: {checkpoints}")
    
    # Set environment variable to allow runtime LoRA updates
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    print("Runtime LoRA updates enabled (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)")
    
    # Run vLLM server in the background
    subprocess.Popen(cmd)


# This app can be deployed programmatically via app.deploy()
# or from the command line with: modal deploy modal_vllm_server.py

# For programmatic deployment
if __name__ == "__main__":
    # If run directly, deploy the app
    print(f"Deploying vLLM server with:")
    print(f"  App Name: {APP_NAME}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  GPU Type: {GPU_TYPE}")
    
    with modal.enable_output():
        app.deploy()
        print(f"\nvLLM server deployed at: {serve_vllm.get_web_url()}")
