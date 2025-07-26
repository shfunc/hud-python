#!/usr/bin/env python3
"""Simple launcher for HUD Browser Environment."""

import os
import subprocess
import sys
import time
import socket
from pathlib import Path

def is_port_open(host, port, timeout=1):
    """Check if a port is open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def wait_for_port(port, timeout=30):
    """Wait for a port to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open('localhost', port):
            return True
        time.sleep(0.5)
    return False

def main():
    # Get all configuration from environment variables
    browser_url = os.environ.get('BROWSER_URL', 'https://google.com')
    launch_apps = os.environ.get('LAUNCH_APPS', '')
    wait_for_apps = os.environ.get('WAIT_FOR_APPS', 'true').lower() == 'true'
    frontend_port_start = int(os.environ.get('FRONTEND_PORT_START', '3000'))
    backend_port_start = int(os.environ.get('BACKEND_PORT_START', '5000'))
    port_allocation = os.environ.get('PORT_ALLOCATION', 'auto')
    
    # Simple command line override: launcher.py [apps] [url]
    if len(sys.argv) > 1:
        launch_apps = sys.argv[1]
    if len(sys.argv) > 2:
        browser_url = sys.argv[2]
    
    # Display configuration
    print("\n==================================")
    print("HUD Browser Environment")
    print("==================================")
    print(f"Browser URL: {browser_url}")
    print(f"Apps to launch: {launch_apps or 'none'}")
    print("==================================\n")
    
    # Start MCP server in background
    print("Starting MCP server...")
    mcp_proc = subprocess.Popen(['hud-controller', 'mcp'])
    time.sleep(2)  # Give it time to start
    
    # Launch apps if specified
    app_procs = []
    first_app_port = None
    app_ports = []  # Track ports to wait for
    
    if launch_apps:
        apps = [app.strip() for app in launch_apps.split(',') if app.strip()]
        frontend_port = frontend_port_start
        backend_port = backend_port_start
        
        for app in apps:
            app_path = Path(f'/app/apps/{app}')
            launch_script = app_path / 'launch.py'
            
            if not launch_script.exists():
                print(f"Warning: App '{app}' not found, skipping")
                continue
            
            print(f"Launching {app}: frontend={frontend_port}, backend={backend_port}")
            
            # Launch the app using python3
            cmd = [
                'python3',
                str(launch_script),
                '--frontend-port', str(frontend_port),
                '--backend-port', str(backend_port)
            ]
            
            proc = subprocess.Popen(cmd, cwd=app_path)
            app_procs.append(proc)
            app_ports.append((app, frontend_port, backend_port))
            
            # Remember first app port for browser URL
            if first_app_port is None:
                first_app_port = frontend_port
            
            # Increment ports for next app (if using auto allocation)
            if port_allocation == 'auto':
                frontend_port += 10
                backend_port += 10
        
        # Wait for apps to be ready if configured
        if wait_for_apps and app_ports:
            print("\nWaiting for apps to be ready...")
            for app_name, f_port, b_port in app_ports:
                print(f"  Waiting for {app_name} backend on port {b_port}...", end='', flush=True)
                if wait_for_port(b_port, timeout=30):
                    print(" ✓")
                else:
                    print(" ✗ (timeout)")
                
                print(f"  Waiting for {app_name} frontend on port {f_port}...", end='', flush=True)
                if wait_for_port(f_port, timeout=30):
                    print(" ✓")
                else:
                    print(" ✗ (timeout)")
            print("Apps are ready!\n")
    
    # Determine final browser URL
    if browser_url == 'https://google.com' and first_app_port:
        # If using default URL and we launched an app, use the app URL
        browser_url = f'http://localhost:{first_app_port}'
        print(f"Using first app URL: {browser_url}")
    
    # Launch browser
    print(f"Launching browser with URL: {browser_url}")
    os.environ['DEFAULT_URL'] = browser_url
    browser_proc = subprocess.Popen(['launch-browser'])
    
    # Wait for MCP server (main process)
    try:
        mcp_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Clean shutdown
        for proc in app_procs + [browser_proc, mcp_proc]:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

if __name__ == '__main__':
    main() 