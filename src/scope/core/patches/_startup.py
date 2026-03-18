"""
Auto-patch at Python startup. Imported by patches.pth.

This module is imported at every Python startup before any user code.
It silently applies patches needed for the current platform and sets
environment variables that must be present before third-party imports.
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        from .cudnn import patch_torch_cudnn
        from .static_cuda_launcher import patch_torch_static_cuda_launcher

        patch_torch_cudnn(silent=True)
        patch_torch_static_cuda_launcher(silent=True)
    except Exception:
        # Never crash Python startup - fail silently
        pass

# Set torch.compile cache directories before torch is imported.
# The inductor cache (FX graph -> Triton source) defaults to the OS temp
# directory, which gets cleaned between reboots. Redirecting to a stable
# location under ~/.daydream-scope/ lets compiled kernels persist across
# sessions, turning a multi-minute first-time compilation into a seconds-long
# cache load on subsequent runs.
try:
    _compile_cache_base = Path("~/.daydream-scope/cache").expanduser().resolve()
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        _inductor_dir = _compile_cache_base / "inductor"
        _inductor_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_inductor_dir)
    if not os.environ.get("TRITON_CACHE_DIR"):
        _triton_dir = _compile_cache_base / "triton"
        _triton_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(_triton_dir)
except Exception:
    pass
