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

# Expose all the ports we need
EXPOSE 8080 8040-8050 3000-3010 5000-5010

# Copy and use start script
COPY start.sh /app/start.sh
# Fix line endings in case they were converted by Git on Windows
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Run via start script to ensure X11 is ready
CMD ["/app/start.sh"] 