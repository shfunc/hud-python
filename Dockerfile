FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY ./api /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen

# Specify the port variable
ARG PORT=8000
ENV PORT=${PORT}

# Expose the port
EXPOSE ${PORT}

CMD ["/app/.venv/bin/uvicorn", "modal-api.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "20"]