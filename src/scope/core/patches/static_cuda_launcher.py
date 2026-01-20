"""
Binary patch for torch_python.dll to fix StaticCudaLauncher overflow.

On Windows, torch.compile with reduce-overhead mode can cause an OverflowError
when CUDA stream values exceed the range of a signed long integer. The fix
changes a format specifier from 'l' (signed long) to 'K' (unsigned long long).

Fixes: https://github.com/pytorch/pytorch/issues/162430
Commit: https://github.com/pytorch/pytorch/commit/7d1bcd9aea8f48733ea46d496e945b7f2592a585

This can be removed when upgrading to a PyTorch version with the fix (2.10.0+).
"""

import os
import sys
import tempfile

from ._utils import find_package_path

# Byte sequences for detection and patching
UNPATCHED_BYTES = b"KiiiiisOl"
PATCHED_BYTES = b"KiiiiisOK"


def patch_torch_static_cuda_launcher(silent: bool = False):
    """Binary patch torch_python.dll to fix StaticCudaLauncher overflow.

    Searches for the unpatched byte sequence and replaces it with the fixed version.
    Idempotent: skips if already patched.

    IMPORTANT: This function does NOT import torch, so it can safely
    modify the DLL before it is loaded.

    Args:
        silent: If True, suppress all output (for use at Python startup).
    """
    if sys.platform != "win32":
        if not silent:
            print("Not on Windows, skipping static_cuda_launcher patch")
        return

    # Find torch package path WITHOUT importing it
    torch_path = find_package_path("torch")

    if not torch_path:
        if not silent:
            print("torch package not found")
        return

    dll_path = os.path.join(torch_path, "lib", "torch_python.dll")

    if not os.path.isfile(dll_path):
        if not silent:
            print(f"torch_python.dll not found: {dll_path}")
        return

    # Read the DLL contents
    try:
        with open(dll_path, "rb") as f:
            content = f.read()
    except PermissionError:
        if not silent:
            print(f"Permission denied reading: {dll_path}")
            print("Close any Python/torch processes and retry.")
        return

    # Check if already patched
    if PATCHED_BYTES in content:
        if not silent:
            print("torch_python.dll: already patched (StaticCudaLauncher fix)")
        return

    # Check if patch is needed
    if UNPATCHED_BYTES not in content:
        if not silent:
            print("torch_python.dll: byte sequence not found (different version?)")
        return

    # Count occurrences to ensure we only patch once
    count = content.count(UNPATCHED_BYTES)
    if count != 1:
        if not silent:
            print(
                f"torch_python.dll: found {count} occurrences of target sequence, expected 1"
            )
        return

    if not silent:
        print(f"Patching torch_python.dll: {dll_path}")

    # Apply the patch
    patched_content = content.replace(UNPATCHED_BYTES, PATCHED_BYTES, 1)

    # Write to temp file first, then rename for atomic operation
    try:
        # Create temp file in same directory for same-filesystem rename
        dll_dir = os.path.dirname(dll_path)
        fd, temp_path = tempfile.mkstemp(dir=dll_dir, suffix=".dll.tmp")
        try:
            os.write(fd, patched_content)
        finally:
            os.close(fd)

        # Make original writable if needed
        if os.path.exists(dll_path):
            os.chmod(dll_path, 0o666)

        # Atomic rename (on Windows, need to remove destination first)
        if os.path.exists(dll_path):
            os.remove(dll_path)
        os.rename(temp_path, dll_path)

        if not silent:
            print("Done. Restart Python to use patched torch_python.dll.")

    except PermissionError:
        if not silent:
            print("Permission denied. Close any Python/torch processes and retry.")
            print(f"Or manually patch: {dll_path}")
        # Clean up temp file if it exists
        if "temp_path" in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def main():
    """Entry point for manual patching."""
    patch_torch_static_cuda_launcher(silent=False)


if __name__ == "__main__":
    main()
