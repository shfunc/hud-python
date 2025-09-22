FROM hudpython/novnc-base:latest

# Set working directory
WORKDIR /app

# Install git for dependency installation
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Layer 1: Install controller dependencies
COPY pyproject.toml .
COPY controller/ ./controller/
RUN uv pip install --system --break-system-packages -e .

# Install Playwright browsers
RUN playwright install chromium --with-deps

# Layer 2: Install environment server dependencies
COPY environment/pyproject.toml ./environment/
RUN cd environment && uv pip install --system --break-system-packages -e .

# Layer 3: Copy the rest of environment folder (source code)
COPY environment/ /app/environment/

# Auto-discover and install/build all frontend apps
RUN set -e; \
    for pkg in $(find /app/environment -type f -path '*/frontend/package.json'); do \
        app_dir=$(dirname "$pkg"); \
        echo "Installing dependencies in $app_dir"; \
        if [ -f "$app_dir/package-lock.json" ]; then \
            (cd "$app_dir" && npm ci --no-audit --no-fund); \
        else \
            (cd "$app_dir" && npm install --no-audit --no-fund); \
        fi; \
    done && \
    for pkg in $(find /app/environment -type f -path '*/frontend/package.json'); do \
        app_dir=$(dirname "$pkg"); \
        if [ -f "$app_dir/next.config.js" ]; then \
            echo "Building Next.js app in $app_dir"; \
            (cd "$app_dir" && npm run build); \
        fi; \
    done

# Make scripts executable
RUN chmod +x /app/environment/start.sh && \
    find /app/environment -name "launch.py" -type f -exec chmod +x {} \;

# Environment configuration
ENV MCP_TRANSPORT="stdio"
ENV HUD_LOG_STREAM="stderr"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONWARNINGS="ignore::SyntaxWarning:pyautogui"

# Expose ports
EXPOSE 8000 8080 3000-3200 5000-5200

# Simple startup: environment server in background, controller with hot-reload
CMD ["sh", "-c", "\
    cd /app/environment && ./start.sh >&2 & \
    sleep 5 && \
    cd /app && exec hud controller:mcp --reload\
"] 