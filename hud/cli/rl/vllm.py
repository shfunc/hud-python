"""vLLM server management utilities."""

import os
import time
import asyncio
import subprocess
from typing import Optional
from pathlib import Path
import httpx

from rich.console import Console

console = Console()


def check_vllm_server() -> bool:
    """Check if vLLM server is running."""
    try:
        response = httpx.get("http://localhost:8000/health", timeout=2.0)
        return response.status_code == 200
    except:
        return False


def kill_vllm_server():
    """Kill any running vLLM server processes."""
    try:
        # Check for PID file first
        pid_file = Path("/tmp/vllm_server.pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                subprocess.run(["kill", "-TERM", str(pid)], check=False)
                time.sleep(2)
                # Force kill if still running
                subprocess.run(["kill", "-9", str(pid)], check=False)
                pid_file.unlink()
            except:
                pass
        
        # Also try to kill by process name
        subprocess.run(["pkill", "-f", "vllm serve"], check=False)
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
        time.sleep(2)
        
        # Check for any process using port 8000
        result = subprocess.run(
            ["lsof", "-ti:8000"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.stdout.strip():
            for pid in result.stdout.strip().split('\n'):
                try:
                    subprocess.run(["kill", "-9", pid], check=False)
                except:
                    pass
        
        console.print("[yellow]Killed existing vLLM server processes[/yellow]")
    except Exception as e:
        console.print(f"[red]Error killing vLLM server: {e}[/red]")


def start_vllm_server(model_name: str, gpu_index: int = 1, restart: bool = False):
    """Start vLLM server in the background with dynamic GPU selection."""
    if restart:
        kill_vllm_server()
        time.sleep(3)
    
    # Check if already running
    if check_vllm_server():
        console.print("[green]vLLM server is already running[/green]")
        return
    
    console.print(f"[cyan]Starting vLLM server with {model_name} on GPU {gpu_index}...[/cyan]")
    
    # Set up environment variables
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_LOGGING_LEVEL": "INFO",  # Changed from DEBUG to reduce noise
        "CUDA_LAUNCH_BLOCKING": "1",  # Better error messages
    })
    
    # Get the path to chat template
    chat_template_path = Path(__file__).parent.parent.parent / "rl" / "chat_template.jinja"
    
    # Build the vLLM command
    cmd = [
        "uv", "run", "vllm", "serve",
        model_name,
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
        "--chat-template", str(chat_template_path.absolute()),
        "--disable-log-requests",
        "--dtype", "auto",
    ]
    
    # Start the server in the background
    log_file = open("/tmp/vllm_server.log", "w")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp,  # Detach from parent process group
        cwd=Path.cwd(),  # Use current working directory
    )
    
    console.print("[yellow]vLLM server starting in background...[/yellow]")
    console.print(f"[yellow]Process ID: {process.pid}[/yellow]")
    console.print(f"[yellow]Check logs at: /tmp/vllm_server.log[/yellow]")
    
    # Save PID for later management
    pid_file = Path("/tmp/vllm_server.pid")
    pid_file.write_text(str(process.pid))


async def wait_for_vllm_server(timeout: int = 360) -> bool:
    """Wait for vLLM server to be ready."""
    start_time = time.time()
    console.print("[yellow]Waiting for vLLM server to be ready (up to 6 minutes)...[/yellow]")
    
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get("http://localhost:8000/health", timeout=2.0)
                if response.status_code == 200:
                    console.print("[green]✅ vLLM server is ready![/green]")
                    return True
            except:
                pass
            
            await asyncio.sleep(2)
            elapsed = int(time.time() - start_time)
            console.print(f"[yellow]Waiting... ({elapsed}s / {timeout}s)[/yellow]", end="\r")
    
    console.print("\n[red]❌ vLLM server failed to start within timeout[/red]")
    console.print("[yellow]Check /tmp/vllm_server.log for details[/yellow]")
    return False
