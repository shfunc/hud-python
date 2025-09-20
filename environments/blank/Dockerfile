FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Start context server in background, then MCP server
CMD ["sh", "-c", "python -m controller.env & sleep 1 && exec python -m controller.server"]
