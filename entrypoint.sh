#!/bin/bash
set -e

# Remove stale SageAttention 2 if present (replaced by sageattn3)
if uv pip show sageattention > /dev/null 2>&1; then
  echo "Removing stale sageattention (SA2)..."
  uv pip uninstall sageattention
fi

# Compile and install sageattn3 on first boot (requires GPU).
# sageattn3's setup.py calls torch.cuda.get_device_capability() which
# needs an actual NVIDIA GPU, so it cannot be built during docker build.
if ! uv pip show sageattn3 > /dev/null 2>&1; then
  echo "Installing sageattn3 (first boot, compiling CUDA extensions)..."
  uv pip install --no-deps "sageattn3 @ git+https://github.com/thu-ml/SageAttention#subdirectory=sageattention3_blackwell"
  echo "sageattn3 installed successfully."
fi

exec "$@"
