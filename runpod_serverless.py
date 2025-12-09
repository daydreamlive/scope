import json
import os
import subprocess
import time

import runpod


def start_scope_server():
    subprocess.Popen(
        [
            "uv",
            "run",
            "daydream-scope",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--no-browser",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def streaming_handler(job):
    # Start the Scope API server in the background
    start_scope_server()

    # Get RunPod environment variables
    public_ip = os.getenv("RUNPOD_PUBLIC_IP", "")
    tcp_port = os.getenv("RUNPOD_TCP_PORT_8000", "8000")

    # Continuously yield the URL every 60 seconds
    while True:
        url_data = {"url": f"{public_ip}:{tcp_port}"}
        yield json.dumps(url_data)
        time.sleep(60)


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": streaming_handler,
            "return_aggregate_stream": True,
        }
    )
