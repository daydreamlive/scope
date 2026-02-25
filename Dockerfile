FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DAYDREAM_SCOPE_LOGS_DIR=/workspace/logs
ENV DAYDREAM_SCOPE_MODELS_DIR=/workspace/models

WORKDIR /app

# Install system dependencies
# Note: build-essential and python3-dev needed for some pip packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  git \
  build-essential \
  software-properties-common \
  # Dependencies required for OpenCV
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  python3-dev \
  # Cleanup
  && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager) to system-wide location
ENV UV_INSTALL_DIR="/usr/local/bin"
RUN curl -LsSf https://astral.sh/uv/0.9.11/install.sh | sh

# Install Python dependencies
# UV_PYTHON_INSTALL_DIR ensures Python is installed to a shared location accessible by all users
ENV UV_PYTHON_INSTALL_DIR="/opt/uv/python"
COPY pyproject.toml uv.lock README.md .python-version LICENSE.md patches.pth .
RUN uv sync --frozen && chmod -R a+rX /opt/uv

# Build frontend
COPY frontend/ ./frontend/
RUN cd frontend \
  && npm install \
  && npm run build \
  && rm -rf node_modules \
  && npm cache clean --force

# Copy project files
COPY src/ /app/src/

# Expose port 8000 for RunPod HTTP proxy
EXPOSE 8000

# Default command to run the application
CMD ["uv", "run", "daydream-scope", "--host", "0.0.0.0", "--port", "8000"]
