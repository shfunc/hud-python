# Build argument to control dev vs production mode
ARG DEV_MODE=false

# Use our HUD base browser image with Playwright and uv pre-installed
FROM hudpython/base-browser:latest

# Create app-specific working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# Conditionally copy source or create stub for development
RUN if [ "$DEV_MODE" = "true" ]; then \
        echo "Dev mode: Creating stub for editable install" && \
        mkdir -p /app/src/hud_controller && \
        echo "# Stub for package installation" > /app/src/hud_controller/__init__.py; \
    else \
        echo "Production mode: Source will be copied"; \
    fi

# Copy source only in production mode (this is a no-op if DEV_MODE=true but doesn't hurt)
COPY src/ ./src/ 

# Install the package using the existing venv at /opt/venv
# The --python flag tells uv to use this specific Python instead of creating a new venv
RUN uv pip install --python /opt/venv -e .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

# Environment variables (PATH and VIRTUAL_ENV already set by base image)
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Google Cloud Platform Credentials for Sheets functionality (here for Windows builds)
ENV GCP_CREDENTIALS_JSON=""

# Run the command directly - it's installed in /opt/venv which is in PATH
CMD ["hud-remote-browser"]