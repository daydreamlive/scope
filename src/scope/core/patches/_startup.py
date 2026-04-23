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

# Sync bundled/user plugins before any server module (uvicorn, fastapi,
# starlette) is imported. If this ran later, `uv pip install` could replace
# packages already cached in sys.modules, producing cross-version splits
# where a class loaded pre-sync is used by a lazily-imported module loaded
# post-sync.
try:
    from scope.core.plugins import ensure_plugins_installed

    ensure_plugins_installed()
except Exception:
    pass
