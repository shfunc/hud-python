FROM hudpython/novnc-base:latest
# Install package
WORKDIR /app
COPY pyproject.toml .
COPY src/ ./src/
RUN uv pip install --system --break-system-packages -e .

# Copy apps
COPY apps/ /app/apps/

# Make launch scripts executable
RUN find /app/apps -name "launch.py" -type f -exec chmod +x {} \;

# Pre-install npm dependencies for all frontend apps
RUN for app in /app/apps/*/frontend; do \
        if [ -f "$app/package.json" ]; then \
            echo "Installing npm dependencies for $app"; \
            cd "$app" && npm install; \
            echo "Building Next.js app in $app"; \
            npm run build || echo "Build failed for $app (might not be Next.js)"; \
            cd -; \
        fi \
    done

# Set defaults
ENV MCP_TRANSPORT="stdio"
ENV HUD_LOG_STREAM="stderr"
ENV PYTHONUNBUFFERED="1"
# Suppress warnings - use module-specific filter for pyautogui
ENV PYTHONWARNINGS="ignore::SyntaxWarning:pyautogui"

# Expose all the ports we need
EXPOSE 8080 8040-8050 3000-3010 5000-5010

# Start X11, context server in background, then run MCP server
# The persistent services survive hot-reloads when running hud dev
CMD ["sh", "-c", "\
    # Start X11 if not already running \
    if [ ! -e /tmp/.X11-unix/X1 ]; then \
        Xvfb :1 -screen 0 1920x1080x24 >/dev/null 2>&1 & \
        # Wait for X11 \
        while [ ! -e /tmp/.X11-unix/X1 ]; do sleep 0.1; done; \
    fi && \
    export DISPLAY=:1 && \
    # Start context server in background \
    python3 -m hud_controller.context >&2 & \
    # Start VNC services \
    x11vnc -display :1 -forever -shared -nopw >/dev/null 2>&1 & \
    websockify --web /usr/share/novnc 8080 localhost:5900 >/dev/null 2>&1 & \
    # Wait a bit for services to start \
    sleep 3 && \
    # Run MCP server in foreground with exec \
    exec hud-controller mcp \
"] 