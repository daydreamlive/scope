# Running Scope on RunPod Serverless

## Introduction

Scope can be run on RunPod Serverless with a few modifications to the codebase and Docker configuration. This guide walks you through the necessary changes and deployment steps.

## Video Tutorial

[![Running Scope on RunPod Serverless](https://img.youtube.com/vi/WrwgfcKpmVs/0.jpg)](https://www.youtube.com/watch?v=WrwgfcKpmVs)

## Code Changes

To run Scope on RunPod Serverless, you need to make the following changes:

### 1. Create RunPod Serverless Handler

Create a new file `runpod_serverless.py` in the project root:

```python
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
```

### 2. Modify Dockerfile

Update your `Dockerfile` to include RunPod dependencies and the serverless handler:

Add these lines before the final `CMD` instruction:

```dockerfile
# Expose port 8000 for RunPod HTTP proxy
EXPOSE 8000

# Install runpod for serverless handler
RUN uv pip install runpod==1.7.0

# Copy RunPod serverless handler
COPY runpod_serverless.py /app/

# Start RunPod serverless handler
CMD ["uv", "run", "python", "runpod_serverless.py"]
```

### 3. Build and Push Docker Image

After making these changes, rebuild your Docker image and push it to Docker Hub:

```bash
# Build the image
docker build -t your-dockerhub-username/scope:runpod-serverless .

# Push to Docker Hub
docker push your-dockerhub-username/scope:runpod-serverless
```

## Steps to Run on RunPod Serverless

### 1. Create New Endpoint

1. Log in to [RunPod](https://www.runpod.io/)
2. Navigate to **Serverless**
3. Click **New Endpoint**
4. Select **Import from Docker Registry** and enter your Docker image name (e.g., `your-dockerhub-username/scope:runpod-serverless`) as the Container Image

### 2. Configure Endpoint Settings

1. **Endpoint Type**: Queue
2. **Worker Type**: GPU
3. **GPU Configuration**: Select **32GB** as the GPU memory specification
4. **Container Configuration**: Specify **8000** in the **"Expose TCP Ports"** field
5. **Environment Variables**: Add `HF_TOKEN` with your HuggingFace token
6. Click **Deploy Endpoint**

## Using the Serverless Endpoint

### Setup

Set the following environment variables:

```bash
export RUNPOD_API_KEY='<your-runpod-api-key>'
export RUNPOD_SERVERLESS_ID='<your-serverless-deployment_id>'
```

You can find your API key in the RunPod dashboard under **Settings** → **API Keys**.
The Serverless ID is the endpoint ID shown in the RunPod dashboard after deployment.

### Start Scope

Start a new job to launch Scope:

```bash
curl -X POST https://api.runpod.ai/v2/${RUNPOD_SERVERLESS_ID}/run \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -d '{"input":{"prompt":"N/A"}}'
```

This will return a `JOB_ID` that you'll need for subsequent operations. Save it:

```bash
export JOB_ID='<job-id-from-response>'
```

### Check Public URL

Stream the job output to get the public URL where Scope is accessible:

```bash
curl -X GET https://api.runpod.ai/v2/${RUNPOD_SERVERLESS_ID}/stream/${JOB_ID} \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

The response will include a `url` field with the public IP and port (e.g., `"url": "123.45.67.89:8000"`). Access Scope at `http://<url>` (e.g., `http://123.45.67.89:8000`).

### Stop the Worker

To stop the worker and free up resources:

```bash
curl -X POST https://api.runpod.ai/v2/${RUNPOD_SERVERLESS_ID}/cancel/${JOB_ID} \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

## Optimizing Model Downloads

To avoid downloading models every time a worker starts, you can use one of the following approaches:

- **Network Volume**: Mount a persistent network volume to store model files. Configure this in the RunPod endpoint settings under **Volumes**.
- **RunPod Model Caching**: Use RunPod's built-in model caching feature to cache Hugging Face models. Enable this in your endpoint configuration.

## Notes

- Workers automatically scale up and down based on demand
- The default timeout for serverless workers is 600 seconds, but this can be configured in the RunPod endpoint settings
- The HTTP curl requests can be integrated into your frontend application to create a seamless user experience
- Keep your `JOB_ID` handy, as you'll need it to stream the URL and cancel the job when finished
