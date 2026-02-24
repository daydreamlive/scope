# ---------- build stage (has nvcc for compiling CUDA extensions) ----------
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS builder

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

# Install only dependencies (not the project itself) into the venv.
# sageattn3 is skipped because it requires a GPU to compile.
# The project is installed as a proper wheel in the runtime stage
# after the source is copied, avoiding broken editable install links.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;10.0;12.0"
COPY pyproject.toml uv.lock README.md .python-version LICENSE.md patches.pth .
RUN uv sync --frozen --no-install-project --no-install-package sageattn3

# ---------- runtime stage (needs devel for sageattn3 compilation) ----------
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

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
  build-essential \
  python3-dev \
  # Dependencies required for OpenCV
  libgl1 \
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

# Copy uv, its managed Python, and the pre-built virtual environment from builder
COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /root/.local/bin/uvx /root/.local/bin/uvx
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
COPY --from=builder /app /app

# Build frontend
COPY frontend/ ./frontend/
RUN cd frontend && npm install && npm run build

# Copy project source and install as a proper wheel (not editable).
# This goes into site-packages so scope is always importable.
COPY src/ /app/src/
RUN uv pip install --no-deps --no-build-isolation .

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port 8000 for RunPod HTTP proxy
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uv", "run", "--no-sync", "daydream-scope", "--host", "0.0.0.0", "--port", "8000"]
