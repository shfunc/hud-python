FROM python:3.11-slim

WORKDIR /app

# Install git for hud-python dependency
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Set logging to stderr
ENV HUD_LOG_STREAM=stderr

# Set Exa API key (r pass it as a -e argument to the docker run command)
# ENV EXA_API_KEY=""

# Start context server in background, then MCP server
CMD ["sh", "-c", "python -m hud_controller.context & sleep 1 && exec python -m hud_controller.server"]
