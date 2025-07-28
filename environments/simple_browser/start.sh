#!/bin/bash
set -e

# Start X server in background -- redirect all output to /dev/null to suppress xkbcomp warnings
Xvfb :1 -screen 0 1920x1080x24 >/dev/null 2>&1 &
X11_PID=$!

# Wait for X11 to be ready
echo "Waiting for X11..." >&2
while [ ! -e /tmp/.X11-unix/X1 ]; do 
    sleep 0.1
done
echo "X11 is ready" >&2

# Set display
export DISPLAY=:1

# Run the MCP server in foreground (basically runs server.py)
exec hud-controller mcp