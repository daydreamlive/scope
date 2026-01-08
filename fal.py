import fal
from fal.container import ContainerImage

# Configuration
DOCKER_IMAGE = "daydreamlive/scope:038798f"

# Create a Dockerfile that uses your existing image as base
dockerfile_str = f"""
FROM {DOCKER_IMAGE}
"""

custom_image = ContainerImage.from_dockerfile_str(
    dockerfile_str,
    # registries=REGISTRY_CONFIG,  # Uncomment if using private registry
)

class ScopeBenchmark(fal.App, keep_alive=300):
    image = custom_image

    # GPU configuration
    machine_type = "GPU-H100"  # Will use GPU, fal will assign appropriate type

    # Additional requirements needed for the setup code
    requirements = [
        "requests",  # For health check
    ]

    @fal.endpoint("/benchmark")
    def benchmark(self, request_data: dict) -> dict:
        """
        Proxy endpoint for live video-to-video requests.
        Forwards the request to the local ai-runner server.
        """
        import requests
        import logging
        import sys
        import subprocess
        import os
        import time
        import threading

        logger = logging.getLogger(__name__)

        try:
            output_dir = "/data/benchmark_results"
            os.makedirs(output_dir, exist_ok=True)

            subprocess.run(
                ["uv", "sync", "--group", "benchmark"],
                cwd="/app",
                check=True,
            )

            # Build command with arguments from request
            cmd = ["uv", "run", "benchmark.py"]

            # Extract args from request_data
            if "args" in request_data:
                cmd.extend(request_data["args"])

            if "--output" not in cmd:
                timestamp = time.strftime("%Y%m%d_%H%M")
                output_path = f"{output_dir}/benchmark_{timestamp}.json"
                cmd.extend(["--output", output_path])

            logger.info(f"Running command: {' '.join(cmd)}")

            subprocess.run(
                cmd,
                cwd="/app",
                check=True,
            )
            return {"message": "Benchmark completed"}

        except requests.exceptions.RequestException as e:
            logger.error(f"Error running benchmark: {e}")
            return {
                "error": str(e),
                "message": "Failed"
            }
