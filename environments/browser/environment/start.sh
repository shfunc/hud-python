#!/bin/bash
# Start script for browser environment
# Starts X11 and the environment API server

set -e

echo "[INFO] Starting browser environment..."

# Set working directory (handle both Docker and local dev)
if [ -d "/app/environment" ]; then
    cd /app/environment
else
    # Assume we're running from the environment directory
    cd "$(dirname "$0")"
fi

# Install dependencies if running outside Docker (for development)
if [ -f "pyproject.toml" ] && [ ! -f "/.dockerenv" ]; then
    echo "[INFO] Installing environment server dependencies..."
    uv pip install -e . || pip install -e .
fi

# Start X11 if not already running
if [ ! -e /tmp/.X11-unix/X1 ]; then
    echo "[INFO] Starting X11 display :1..."
    Xvfb :1 -screen 0 1920x1080x24 >/dev/null 2>&1 &
    
    # Wait for X11 to be ready
    for i in {1..50}; do
        if [ -e /tmp/.X11-unix/X1 ]; then
            echo "[INFO] X11 is ready"
            break
        fi
        sleep 0.1
    done
else
    echo "[INFO] X11 display :1 already running"
fi

# Set display environment variable
export DISPLAY=:1

# Start the FastAPI server
echo "[INFO] Starting browser environment API server..."
echo "[INFO] Environment API: http://localhost:8000"
echo "[INFO] VNC viewer: http://localhost:8080/vnc.html"
echo "[INFO] Apps will be available on ports 3000-3200 (frontend) and 5000-5200 (backend)"

# Run the server with proper error handling
exec python3 -m uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --access-log \
    --use-colors
