"""
Patch PyTorch's bundled cuDNN with a newer version from nvidia-cudnn-cu12.

On Windows, PyTorch bundles cuDNN in torch/lib and loads it directly by path,
ignoring PATH and os.add_dll_directory(). To use a newer cuDNN version, we
must copy the DLLs from nvidia-cudnn-cu12 to torch/lib.

This fixes the PyTorch 2.9.1 Conv3D bf16 performance regression.
See: https://github.com/pytorch/pytorch/issues/168167

This can be removed when a new PyTorch with the correct cuDNN version is released.
"""

import glob
import os
import shutil
import sys


def patch_torch_cudnn(silent: bool = False):
    """Copy cuDNN DLLs from nvidia-cudnn-cu12 to torch/lib on Windows.

    This patches PyTorch to use the newer cuDNN version.
    Idempotent: skips files that are already the correct size.

    Args:
        silent: If True, suppress all output (for use at Python startup).
    """
    if sys.platform != "win32":
        if not silent:
            print("Not on Windows, skipping cuDNN patch")
        return

    try:
        import nvidia.cudnn
        import torch
    except ImportError as e:
        if not silent:
            print(f"Required packages not installed: {e}")
        return

    cudnn_src = os.path.join(nvidia.cudnn.__path__[0], "bin")
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

    if not os.path.isdir(cudnn_src):
        if not silent:
            print(f"cuDNN source not found: {cudnn_src}")
        return

    if not os.path.isdir(torch_lib):
        if not silent:
            print(f"torch lib not found: {torch_lib}")
        return

    # Find cuDNN DLLs to copy
    cudnn_dlls = glob.glob(os.path.join(cudnn_src, "cudnn*.dll"))

    if not cudnn_dlls:
        if not silent:
            print(f"No cuDNN DLLs found in {cudnn_src}")
        return

    if not silent:
        print(f"Patching torch cuDNN: {cudnn_src} -> {torch_lib}")

    for src_dll in cudnn_dlls:
        dll_name = os.path.basename(src_dll)
        dst_dll = os.path.join(torch_lib, dll_name)

        if os.path.exists(dst_dll):
            src_size = os.path.getsize(src_dll)
            dst_size = os.path.getsize(dst_dll)
            if src_size == dst_size:
                if not silent:
                    print(f"  {dll_name}: already patched (same size)")
                continue

        if not silent:
            print(f"  {dll_name}: copying...")
        try:
            # Try to make the destination writable if it exists
            if os.path.exists(dst_dll):
                os.chmod(dst_dll, 0o666)
            shutil.copy2(src_dll, dst_dll)
        except PermissionError:
            if not silent:
                print(
                    "    ERROR: Permission denied. Close any Python/torch processes and retry."
                )
                print(f"    Or manually copy: {src_dll} -> {dst_dll}")
            continue

    if not silent:
        print("Done. Restart Python to use new cuDNN.")


def main():
    """Entry point for manual patching."""
    patch_torch_cudnn(silent=False)


if __name__ == "__main__":
    main()
