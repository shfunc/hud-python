FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY pyproject.toml ./
COPY controller/ ./controller/
COPY environment/ ./environment/
RUN pip install --no-cache-dir -e .

ENV ENV_SERVER_PORT=8005

# Start context server in background, then run controller with hot-reload
CMD ["sh", "-c", "uvicorn environment.server:app --host 0.0.0.0 --port $ENV_SERVER_PORT --reload & sleep 0.5 && exec hud controller --reload"]
