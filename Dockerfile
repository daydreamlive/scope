# ---------- build stage (has nvcc for compiling CUDA extensions) ----------
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
  curl \
  git \
  build-essential \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python dependencies (sageattn3 compiles CUDA extensions here)
# TORCH_CUDA_ARCH_LIST tells setup.py which GPU architectures to compile for
# without needing a physical GPU present during docker build.
# 8.0=A100, 8.9=L40/L4/RTX4090, 9.0=H100/H200, 10.0=B200, 12.0=RTX5090
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;10.0;12.0"
COPY pyproject.toml uv.lock README.md .python-version LICENSE.md patches.pth .
RUN uv sync --frozen

# ---------- runtime stage (smaller, no compiler toolchain) ----------
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DAYDREAM_SCOPE_LOGS_DIR=/workspace/logs
ENV DAYDREAM_SCOPE_MODELS_DIR=/workspace/models
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y \
  curl \
  git \
  software-properties-common \
  # Dependencies required for OpenCV
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  # Cleanup
  && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y nodejs

# Copy uv and the pre-built virtual environment from builder
COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /root/.local/bin/uvx /root/.local/bin/uvx
COPY --from=builder /app /app

# Build frontend
COPY frontend/ ./frontend/
RUN cd frontend && npm install && npm run build

# Copy project files
COPY src/ /app/src/

# Expose port 8000 for RunPod HTTP proxy
EXPOSE 8000

# Default command to run the application
CMD ["uv", "run", "--no-sync", "daydream-scope", "--host", "0.0.0.0", "--port", "8000"]
