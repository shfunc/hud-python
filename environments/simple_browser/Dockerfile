FROM hudpython/novnc-base:latest

# Build-time configuration with sensible defaults
ARG DEFAULT_BROWSER_URL="https://google.com"
ARG DEFAULT_FRONTEND_PORT="3000"
ARG DEFAULT_BACKEND_PORT="5000"

# Runtime configuration via ENV (can be overridden)
ENV BROWSER_URL=${DEFAULT_BROWSER_URL}
ENV FRONTEND_PORT_START=${DEFAULT_FRONTEND_PORT}
ENV BACKEND_PORT_START=${DEFAULT_BACKEND_PORT}
ENV LAUNCH_APPS=""
ENV WAIT_FOR_APPS="true"
ENV PORT_ALLOCATION="auto"

# Install the package
WORKDIR /app
COPY pyproject.toml .
COPY src/ ./src/
RUN uv pip install --system --break-system-packages -e .

# Copy apps directory
COPY apps/ /app/apps/
RUN find /app/apps -name "*.py" -type f -exec chmod +x {} \;

# Pre-install npm dependencies and build frontend apps
RUN for app in /app/apps/*/frontend; do \
        if [ -f "$app/package.json" ]; then \
            echo "Installing npm dependencies for $app"; \
            cd "$app" && npm install; \
            echo "Building Next.js app in $app"; \
            npm run build || echo "Build failed for $app (might not be Next.js)"; \
            cd -; \
        fi \
    done

# Copy start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Set working directory
WORKDIR /

# Expose ports
EXPOSE 3000 5000 8040 8080

# Support both ENV-based config and CLI arguments
# ENV vars are used if no CLI args provided
ENTRYPOINT ["/start.sh"]
CMD [] 