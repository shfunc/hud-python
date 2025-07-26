#!/bin/bash
set -e

# Setup display
export DISPLAY=:1
export HOME=/root

# Create .Xauthority if it doesn't exist
touch $HOME/.Xauthority

# Start Xvfb
echo "Starting Xvfb..."
Xvfb $DISPLAY -screen 0 1920x1080x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
XVFB_PID=$!

# Wait for X server socket instead of sleeping
echo "Waiting for X server..."
for i in {1..20}; do
    if [ -e /tmp/.X11-unix/X1 ]; then
        break
    fi
    sleep 0.1
done

# Start x11vnc (quiet mode)
echo "Starting x11vnc..."
x11vnc -display $DISPLAY -forever -shared -nopw -quiet &
X11VNC_PID=$!

# Wait for VNC port instead of sleeping
echo "Waiting for VNC..."
for i in {1..20}; do
    if netstat -tuln 2>/dev/null | grep -q ':5900'; then
        break
    fi
    sleep 0.1
done

# Start noVNC (websockify)
echo "Starting noVNC..."
websockify --web /usr/share/novnc 8080 localhost:5900 &
NOVNC_PID=$!

# Display access information
echo ""
echo "=================================="
echo "Display environment is ready!"
echo "noVNC URL: http://localhost:8080/vnc.html"
echo "=================================="
echo ""

# Keep running
tail -f /dev/null 