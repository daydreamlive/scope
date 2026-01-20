"""
Auto-patch at Python startup. Imported by patches.pth.

This module is imported at every Python startup before any user code.
It silently applies patches needed for the current platform.
"""

import sys

if sys.platform == "win32":
    try:
        from .cudnn import patch_torch_cudnn
        from .static_cuda_launcher import patch_torch_static_cuda_launcher

        patch_torch_cudnn(silent=True)
        patch_torch_static_cuda_launcher(silent=True)
    except Exception:
        # Never crash Python startup - fail silently
        pass
