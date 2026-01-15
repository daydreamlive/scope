FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DAYDREAM_SCOPE_LOGS_DIR=/workspace/logs
ENV DAYDREAM_SCOPE_MODELS_DIR=/workspace/models

WORKDIR /app

RUN apt-get update && apt-get install -y \
  # System dependencies
  curl \
  git \
  build-essential \
  software-properties-common \
  # Dependencies required for OpenCV
  libgl1-mesa-glx \
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
  && apt-get install -y nodejs

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml uv.lock README.md .python-version LICENSE.md patches.pth .
RUN uv sync --frozen

# Build frontend
COPY frontend/ ./frontend/
RUN cd frontend && npm install && npm run build

# Copy project files
COPY src/ /app/src/

# Expose port 8000 for RunPod HTTP proxy
EXPOSE 8000

# Default command to run the application
CMD ["uv", "run", "daydream-scope", "--host", "0.0.0.0", "--port", "8000"]
