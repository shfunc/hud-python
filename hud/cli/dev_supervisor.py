"""Development supervisor with file watching and process management."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class ProcessReloader(FileSystemEventHandler):
    """Handles file changes and process restarting."""
    
    def __init__(self, cmd: list[str], watch_extensions: list[str] | None = None):
        self.cmd = cmd
        self.proc = None
        self.last_restart = 0
        self.watch_extensions = watch_extensions or ['.py', '.js', '.ts', '.json', '.yaml', '.yml']
        self.debounce_seconds = 1.0
        self.start_process()
    
    def start_process(self):
        """Start or restart the supervised process."""
        # Kill existing process if running
        if self.proc:
            print("🛑 Stopping process...", flush=True)
            self.terminate_process()
        
        # Start new process
        print(f"🚀 Starting: {' '.join(self.cmd)}", flush=True)
        self.proc = subprocess.Popen(self.cmd)
        self.last_restart = time.time()
    
    def terminate_process(self):
        """Gracefully terminate the process."""
        if not self.proc:
            return
            
        # Try graceful shutdown first
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Process didn't stop gracefully, forcing...", flush=True)
            self.proc.kill()
            self.proc.wait()
    
    def should_reload(self, path: str) -> bool:
        """Check if file change should trigger reload."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in Path(path).parts):
            return False
        
        # Skip common ignore patterns
        ignore_patterns = ['__pycache__', 'node_modules', '.git', '.venv', 'venv']
        if any(pattern in path for pattern in ignore_patterns):
            return False
        
        # Check extension
        return any(path.endswith(ext) for ext in self.watch_extensions)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        if not self.should_reload(event.src_path):
            return
        
        # Debounce rapid changes
        current_time = time.time()
        if current_time - self.last_restart < self.debounce_seconds:
            return
        
        print(f"🔄 Detected change in {Path(event.src_path).name}", flush=True)
        self.start_process()


def run_supervisor(cmd: list[str], watch_dir: str = "/app/src") -> None:
    """
    Run a command with file watching and auto-restart.
    
    Args:
        cmd: Command to run and supervise
        watch_dir: Directory to watch for changes
    """
    # Ensure watch directory exists
    watch_path = Path(watch_dir)
    if not watch_path.exists():
        print(f"❌ Watch directory not found: {watch_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"👀 Watching {watch_dir} for changes...", flush=True)
    print(f"🎯 Running: {' '.join(cmd)}", flush=True)
    
    # Create file watcher
    handler = ProcessReloader(cmd)
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=True)
    observer.start()
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print("\n👋 Shutting down...", flush=True)
        handler.terminate_process()
        observer.stop()
        observer.join()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            # Check if process died unexpectedly
            if handler.proc and handler.proc.poll() is not None:
                exit_code = handler.proc.returncode
                print(f"❌ Process exited with code {exit_code}", flush=True)
                if exit_code != 0:
                    # Wait a bit before restarting on crash
                    time.sleep(2)
                    handler.start_process()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
