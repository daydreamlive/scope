"""Test sageattn3 by running our Docker image on Modal with a GPU."""

import modal

# Point this to your pushed Docker image
IMAGE_TAG = "zmiccer/scope_persona:latest"

image = modal.Image.from_registry(
    IMAGE_TAG,
    add_python="3.12",
).env({
    # Let Modal's Python find packages from the image's uv venv
    "PYTHONPATH": "/app/.venv/lib/python3.12/site-packages",
    "PATH": "/root/.local/bin:/app/.venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
})

app = modal.App("test-sageattn3")


@app.function(image=image, gpu="H100", timeout=600)
def test_sageattn3():
    import os
    import subprocess
    import sys

    # Debug: check what's at /app
    print("=== Files in /app/ ===")
    for f in sorted(os.listdir("/app")):
        print(f"  {f}")
    print(f"entrypoint.sh exists: {os.path.exists('/app/entrypoint.sh')}")
    print()

    # Run the entrypoint logic to compile sageattn3
    result = subprocess.run(
        ["bash", "/app/entrypoint.sh", "echo", "entrypoint done"],
        capture_output=True,
        text=True,
        cwd="/app",
        env={
            "PATH": "/root/.local/bin:/app/.venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "VIRTUAL_ENV": "/app/.venv",
            "HOME": "/root",
            "CUDA_HOME": "/usr/local/cuda",
        },
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"entrypoint.sh failed with exit code {result.returncode}")
        sys.exit(1)

    # Reload site-packages so newly installed sageattn3 is visible
    import importlib
    import site
    importlib.invalidate_caches()
    site.addsitedir("/app/.venv/lib/python3.12/site-packages")

    import torch

    print(f"torch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    try:
        from sageattention import sageattn

        B, H, S, D = 1, 8, 256, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        out = sageattn(q, k, v)
        print(f"sageattn output: {out.shape}")
        print("SUCCESS: sageattn3 works!")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)


@app.local_entrypoint()
def main():
    test_sageattn3.remote()
