"""Test sageattn3 by running our Docker image on Modal with a GPU."""

import modal

IMAGE_TAG = "zmiccer/scope_persona:latest"

image = (
    modal.Image.from_registry(IMAGE_TAG)
    # Modal needs `python` and `pip` on PATH to detect the Python version.
    # Symlink from the uv-managed venv so we don't need add_python (which
    # clobbers the CUDA devel toolkit from the nvidia/cuda base image).
    .run_commands(
        "ln -sf /app/.venv/bin/python3.12 /usr/local/bin/python",
        "ln -sf /app/.venv/bin/pip /usr/local/bin/pip",
    )
    # Modal strips /usr/local/cuda/bin (no nvcc). Reinstall the CUDA 12.8
    # compiler toolkit so sageattn3 can compile its CUDA extensions.
    .run_commands(
        "apt-get update",
        "apt-get install -y --no-install-recommends wget",
        "wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb",
        "apt-get update",
        "apt-get install -y --no-install-recommends cuda-nvcc-12-8",
        "rm -rf /var/lib/apt/lists/*",
        # Verify nvcc is available
        "/usr/local/cuda-12.8/bin/nvcc --version",
        # Symlink so it's found at the standard path
        "ln -sf /usr/local/cuda-12.8 /usr/local/cuda || true",
    )
    # Re-add entrypoint.sh since Modal's overlays clobber Docker COPY paths
    .add_local_file("entrypoint.sh", "/opt/entrypoint.sh", copy=True)
    .run_commands("chmod +x /opt/entrypoint.sh")
)

app = modal.App("test-sageattn3")


@app.function(image=image, gpu="H100", timeout=600)
def test_sageattn3():
    import subprocess
    import sys

    # Run entrypoint.sh exactly as RunPod would
    result = subprocess.run(
        ["bash", "/opt/entrypoint.sh", "echo", "entrypoint done"],
        capture_output=True,
        text=True,
        cwd="/app",
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
