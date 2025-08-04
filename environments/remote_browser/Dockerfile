# Use our HUD base browser image with Playwright and uv pre-installed
FROM hudpython/base-browser:latest

# Create app-specific working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package using the existing venv at /opt/venv
# The --python flag tells uv to use this specific Python instead of creating a new venv
RUN uv pip install --python /opt/venv -e .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

# Environment variables (PATH and VIRTUAL_ENV already set by base image)
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Run the command directly - it's installed in /opt/venv which is in PATH
CMD ["hud-remote-browser"]