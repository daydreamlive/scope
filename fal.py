import fal
from fal.container import ContainerImage

# Configuration
DOCKER_IMAGE = "daydreamlive/scope:main@sha256:69eb6cbf81b3899283486459c01a756844c99f4f5e42a724649dee6ec7b535ef"

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
    machine_type = "GPU"  # Will use GPU, fal will assign appropriate type

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

        logger = logging.getLogger(__name__)

        try:
            import sys
                # The container has uv and the project installed
            subprocess.run(
                ["uv", "run", "benchmark.py", "blah"],
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
