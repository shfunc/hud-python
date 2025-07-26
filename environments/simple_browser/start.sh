#!/bin/bash
set -e

# Start base services in background
/start-base.sh &
BASE_PID=$!

# Setup display
export DISPLAY=:1
export HOME=/root

# Wait for critical services to be ready
echo "Waiting for services to be ready..."
for i in {1..50}; do
    if [ -e /tmp/.X11-unix/X1 ] && netstat -tuln 2>/dev/null | grep -q ':5900'; then
        echo "Services are ready!"
        break
    fi
    if [ $i -eq 50 ]; then
        echo "Warning: Services may not be fully ready"
    fi
    sleep 0.2
done

# Run the Python launcher with all args passed through
python3 /app/src/hud_controller/launcher.py "$@" &
LAUNCHER_PID=$!

# Handle shutdown gracefully
cleanup() {
    echo "Shutting down..."
    # Kill launcher first (it will clean up its child processes)
    if [ -n "$LAUNCHER_PID" ]; then
        kill $LAUNCHER_PID 2>/dev/null || true
        wait $LAUNCHER_PID 2>/dev/null || true
    fi
    # Then kill base services
    if [ -n "$BASE_PID" ]; then
        kill $BASE_PID 2>/dev/null || true
        wait $BASE_PID 2>/dev/null || true
    fi
    exit
}

trap cleanup EXIT INT TERM

# Wait for launcher to finish
wait $LAUNCHER_PID