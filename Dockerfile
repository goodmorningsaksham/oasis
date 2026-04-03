# GlucoRL — Insulin Dosing RL Environment
# Uses openenv-base for OpenEnv spec compliance (FastAPI, WebSocket, health checks)

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Install git (needed for openenv-core git dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . /app/env

# Set PYTHONPATH so imports resolve correctly
ENV PYTHONPATH="/app/env:${PYTHONPATH}"

# Enable OpenEnv web interface
ENV ENABLE_WEB_INTERFACE=true

# Health check — matches OpenEnv convention
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
