# Stage 1: Builder
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  git \
  build-essential \
  software-properties-common \
  python3-dev \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x (only needed for frontend build)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python dependencies (no dev deps, clean cache)
COPY pyproject.toml uv.lock README.md .python-version LICENSE.md patches.pth .
RUN uv sync --frozen --no-dev && uv cache clean

# Build frontend
COPY frontend/ ./frontend/
RUN cd frontend && npm install && npm run build

# Stage 2: Runtime (no Node.js, no build tools, no caches)
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DAYDREAM_SCOPE_LOGS_DIR=/workspace/logs
ENV DAYDREAM_SCOPE_MODELS_DIR=/workspace/models

WORKDIR /app

# Runtime-only system dependencies (OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Copy uv binary
COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /root/.local/bin/uvx /root/.local/bin/uvx
ENV PATH="/root/.local/bin:$PATH"

# Copy Python virtual environment and project metadata
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /app/uv.lock /app/uv.lock
COPY --from=builder /app/.python-version /app/.python-version
COPY --from=builder /app/README.md /app/README.md
COPY --from=builder /app/LICENSE.md /app/LICENSE.md
COPY --from=builder /app/patches.pth /app/patches.pth

# Copy built frontend
COPY --from=builder /app/frontend/dist /app/frontend/dist

# Copy project files
COPY src/ /app/src/

# Expose port 8000 for RunPod HTTP proxy
EXPOSE 8000

# Default command to run the application
CMD ["uv", "run", "daydream-scope", "--host", "0.0.0.0", "--port", "8000"]
