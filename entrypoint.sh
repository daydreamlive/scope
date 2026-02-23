#!/bin/bash
set -e

# Detect GPU compute capability (major.minor)
GPU_CC=$(python -c "import torch; cc = torch.cuda.get_device_capability(0); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "0")

if [ "$GPU_CC" -ge 100 ]; then
  # Blackwell (SM 100+): use sageattn3, remove SA2 if present
  echo "Blackwell GPU detected (SM $GPU_CC) — using sageattn3"

  if uv pip show sageattention > /dev/null 2>&1; then
    echo "Removing SageAttention 2 (replaced by sageattn3 on Blackwell)..."
    uv pip uninstall sageattention
  fi

  if ! uv pip show sageattn3 > /dev/null 2>&1; then
    echo "Installing sageattn3 (first boot, compiling CUDA extensions)..."
    uv pip install --no-build-isolation --no-deps "sageattn3 @ git+https://github.com/thu-ml/SageAttention#subdirectory=sageattention3_blackwell"
    echo "sageattn3 installed successfully."
  fi
else
  # Hopper/Ada/Ampere: keep SageAttention 2 (sageattn3 is Blackwell-only)
  echo "Non-Blackwell GPU detected (SM $GPU_CC) — keeping SageAttention 2"
fi

exec "$@"
