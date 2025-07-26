#!/usr/bin/env python3
"""Launch script for the todo app with parameterized ports."""

import argparse
import subprocess
import sys
import os
import signal
import time
from pathlib import Path


def launch_app(frontend_port: int = 3000, backend_port: int = 5000):
    """Launch the todo app with specified ports."""
    app_dir = Path(__file__).parent
    frontend_dir = app_dir / "frontend"
    backend_dir = app_dir / "backend"
    
    processes = []
    
    def cleanup(signum=None, frame=None):
        """Clean up processes on exit."""
        print("\nShutting down services...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
                p.kill()
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Start backend with uv
        print(f"Starting backend on port {backend_port}...")
        backend_env = os.environ.copy()
        backend_env["PYTHONUNBUFFERED"] = "1"
        
        backend_cmd = [
            "uv", "run", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", str(backend_port)
        ]
        
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=backend_dir,
            env=backend_env
        )
        processes.append(backend_process)
        
        # Give backend time to start
        time.sleep(3)
        
        # Check if node_modules exists, install if not
        if not (frontend_dir / "node_modules").exists():
            print("Installing frontend dependencies...")
            npm_install = subprocess.run(
                ["npm", "install"],
                cwd=frontend_dir,
                check=True
            )
        
        # Update next.config.js to use the correct backend port
        next_config = frontend_dir / "next.config.js"
        if next_config.exists():
            config_content = next_config.read_text()
            # Replace the backend port in the proxy configuration
            new_config = config_content.replace(
                "http://localhost:5000",
                f"http://localhost:{backend_port}"
            )
            next_config.write_text(new_config)
        
        # Start frontend
        print(f"Starting frontend on port {frontend_port}...")
        frontend_env = os.environ.copy()
        frontend_env["PORT"] = str(frontend_port)
        
        # Check if we have a production build
        if (frontend_dir / ".next").exists():
            print("Running in production mode (pre-built)...")
            frontend_cmd = [
                "npm", "run", "start",
                "--", "--port", str(frontend_port),
                "--hostname", "0.0.0.0"
            ]
        else:
            print("Running in development mode...")
            frontend_cmd = [
                "npm", "run", "dev",
                "--", "--port", str(frontend_port),
                "--hostname", "0.0.0.0"
            ]
        
        frontend_process = subprocess.Popen(
            frontend_cmd,
            cwd=frontend_dir,
            env=frontend_env
        )
        processes.append(frontend_process)
        
        print(f"\nTodo app started!")
        print(f"Frontend: http://localhost:{frontend_port}")
        print(f"Backend API: http://localhost:{backend_port}/docs")
        print("\nPress Ctrl+C to stop")
        
        # Wait for processes
        while all(p.poll() is None for p in processes):
            time.sleep(1)
            
    except Exception as e:
        print(f"Error launching app: {e}")
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="Launch the todo app")
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=3000,
        help="Frontend port (default: 3000)"
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=5000,
        help="Backend port (default: 5000)"
    )
    
    args = parser.parse_args()
    launch_app(args.frontend_port, args.backend_port)


if __name__ == "__main__":
    main() 