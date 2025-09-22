FROM python:3.11-slim

WORKDIR /app

# Install runtime deps (curl for debugging, optional)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy environment sources
COPY pyproject.toml ./
COPY controller/ ./controller/
COPY environment/ ./environment/

# Install environment in editable mode
RUN pip install --no-cache-dir -e .

# Configure
ENV HUD_LOG_STREAM=stderr
ENV ENV_SERVER_URL=http://localhost:8000
# EXA_API_KEY should be provided at runtime: -e EXA_API_KEY=...

# Start environment HTTP server, then run controller with hot-reload
CMD ["sh", "-c", "uvicorn environment.server:app --host 0.0.0.0 --port 8000 --reload & sleep 0.5 && exec hud controller --reload"]
