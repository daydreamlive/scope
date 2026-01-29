"""
fal.ai deployment for Scope.

This runs the Scope backend and proxies WebRTC signaling + API calls through
a single WebSocket connection to avoid fal spawning new runners for each request.

Based on:
- https://docs.fal.ai/examples/serverless/deploy-models-with-custom-containers
- https://github.com/fal-ai-community/fal-demos/blob/main/fal_demos/video/yolo_webcam_webrtc/yolo.py
"""

import fal
from fal.container import ContainerImage
from fastapi import WebSocket

# Configuration
DOCKER_IMAGE = "daydreamlive/scope:d117be7"

# Create a Dockerfile that uses your existing image as base
dockerfile_str = f"""
FROM {DOCKER_IMAGE}

"""

# Create container image from Dockerfile string
custom_image = ContainerImage.from_dockerfile_str(
    dockerfile_str,
)


class ScopeApp(fal.App, keep_alive=1800):
    """
    Scope server on fal.ai.

    This runs the Scope backend as a subprocess and exposes a WebSocket endpoint
    that handles:
    1. WebRTC signaling (SDP offer/answer, ICE candidates)
    2. REST API calls (proxied through WebSocket to avoid new runner instances)

    The actual WebRTC video stream flows directly between browser and this runner
    once the signaling is complete.

    """

    # Set custom Docker image
    image = custom_image

    # GPU configuration
    machine_type = "GPU-H100"

    # Additional requirements needed for the setup code
    requirements = [
        "requests",
        "httpx",  # For async HTTP requests
    ]

    # Public auth mode (no JWT token required)
    auth_mode = "public"

    def setup(self):
        """
        Start the Scope backend server as a background process.
        """
        import logging
        import os
        import subprocess
        import threading
        import time

        logger = logging.getLogger(__name__)
        print("Starting Scope container setup...")

        # Verify GPU is available
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            print(f"GPU Status:\n{result.stdout}")
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            raise

        # Environment for scope
        scope_env = os.environ.copy()
        # Use fal's /data directory for persistent storage (survives restarts)
        scope_env["DAYDREAM_SCOPE_MODELS_DIR"] = "/data/models"
        scope_env["DAYDREAM_SCOPE_LOGS_DIR"] = "/data/logs"
        scope_env["DAYDREAM_SCOPE_ASSETS_DIR"] = "/data/assets"

        # Start the scope server in a background thread
        def start_server():
            print("Starting Scope server...")
            try:
                subprocess.run(
                    ["uv", "run", "daydream-scope", "--no-browser", "--host", "0.0.0.0", "--port", "8000"],
                    check=True,
                    env=scope_env,
                )
            except Exception as e:
                logger.error(f"Failed to start Scope server: {e}")
                raise

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        # Wait for the server to be ready
        print("Waiting for Scope server to start...")
        max_wait = 120  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                import requests

                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Scope server is running on port 8000")
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            logger.warning(
                f"Scope server health check timed out after {max_wait}s, continuing anyway..."
            )

        print("Scope container setup complete")

    @fal.endpoint("/ws", is_websocket=True)
    async def websocket_handler(self, ws: WebSocket) -> None:
        """
        Main WebSocket endpoint that handles:
        1. WebRTC signaling (offer/answer, ICE candidates)
        2. REST API call proxying

        Protocol:
        - All messages are JSON with a "type" field
        - WebRTC signaling types: "get_ice_servers", "offer", "icecandidate"
        - API proxy type: "api" with "method", "path", "body" fields

        This keeps a persistent connection to prevent fal from spawning new runners.
        """
        import asyncio
        import json
        import logging

        import httpx
        from starlette.websockets import WebSocketDisconnect, WebSocketState

        logger = logging.getLogger(__name__)
        SCOPE_BASE_URL = "http://localhost:8000"

        await ws.accept()
        print("✅ WebSocket connection accepted")

        # Send ready message
        await ws.send_json({"type": "ready"})

        # Track WebRTC session ID for ICE candidate routing
        session_id = None

        async def safe_send_json(payload: dict):
            """Send JSON, handling connection errors gracefully."""
            try:
                if (
                    ws.client_state != WebSocketState.CONNECTED
                    or ws.application_state != WebSocketState.CONNECTED
                ):
                    return
                await ws.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                pass

        async def handle_get_ice_servers():
            """Proxy GET /api/v1/webrtc/ice-servers"""
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SCOPE_BASE_URL}/api/v1/webrtc/ice-servers"
                )
                return {
                    "type": "ice_servers",
                    "data": response.json(),
                    "status": response.status_code,
                }

        async def handle_offer(payload: dict):
            """Proxy POST /api/v1/webrtc/offer

            The backend's offer endpoint handles pipeline loading:
            - If pipeline not loaded, it auto-loads from initialParameters
            - The load is awaited, so it blocks until complete
            - No pre-loading needed here
            """
            nonlocal session_id
            request_id = payload.get("request_id")

            async with httpx.AsyncClient() as client:
                # Just proxy the offer - backend handles pipeline loading and waits for it
                response = await client.post(
                    f"{SCOPE_BASE_URL}/api/v1/webrtc/offer",
                    json={
                        "sdp": payload.get("sdp"),
                        "type": payload.get("sdp_type", "offer"),
                        "initialParameters": payload.get("initialParameters"),
                    },
                    timeout=180.0,  # Long timeout - backend may need to load pipeline first
                )

                if response.status_code == 200:
                    data = response.json()
                    session_id = data.get("sessionId")
                    return {
                        "type": "answer",
                        "request_id": request_id,
                        "sdp": data.get("sdp"),
                        "sdp_type": data.get("type"),
                        "sessionId": session_id,
                    }
                else:
                    return {
                        "type": "error",
                        "request_id": request_id,
                        "error": f"Offer failed: {response.status_code}",
                        "detail": response.text,
                    }

        async def handle_icecandidate(payload: dict):
            """Proxy PATCH /api/v1/webrtc/offer/{session_id} for ICE candidates"""
            nonlocal session_id
            request_id = payload.get("request_id")

            candidate = payload.get("candidate")
            target_session = payload.get("sessionId") or session_id

            if not target_session:
                return {
                    "type": "error",
                    "request_id": request_id,
                    "error": "No session ID available for ICE candidate",
                }

            if candidate is None:
                # End of candidates signal
                return {"type": "icecandidate_ack", "request_id": request_id, "status": "end_of_candidates"}

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{SCOPE_BASE_URL}/api/v1/webrtc/offer/{target_session}",
                    json={
                        "candidates": [
                            {
                                "candidate": candidate.get("candidate"),
                                "sdpMid": candidate.get("sdpMid"),
                                "sdpMLineIndex": candidate.get("sdpMLineIndex"),
                            }
                        ]
                    },
                    timeout=10.0,
                )

                if response.status_code == 204:
                    return {"type": "icecandidate_ack", "request_id": request_id, "status": "ok"}
                else:
                    return {
                        "type": "error",
                        "request_id": request_id,
                        "error": f"ICE candidate failed: {response.status_code}",
                        "detail": response.text,
                    }

        async def handle_api_request(payload: dict):
            """
            Proxy arbitrary API requests to Scope backend.

            Expected payload:
            {
                "type": "api",
                "method": "GET" | "POST" | "PATCH" | "DELETE",
                "path": "/api/v1/...",
                "body": {...}  # optional, for POST/PATCH
                "request_id": "..."  # optional, for correlating responses
            }

            Special handling for file uploads:
            If body contains "_base64_content", it's decoded and sent as binary.
            """
            import base64

            method = payload.get("method", "GET").upper()
            path = payload.get("path", "")
            body = payload.get("body")
            request_id = payload.get("request_id")

            async with httpx.AsyncClient() as client:
                try:
                    # Check if this is a base64-encoded file upload
                    is_binary_upload = (
                        body
                        and isinstance(body, dict)
                        and "_base64_content" in body
                    )

                    if method == "GET":
                        response = await client.get(
                            f"{SCOPE_BASE_URL}{path}", timeout=30.0
                        )
                    elif method == "POST":
                        if is_binary_upload:
                            # Decode base64 and send as binary
                            binary_content = base64.b64decode(body["_base64_content"])
                            content_type = body.get(
                                "_content_type", "application/octet-stream"
                            )
                            response = await client.post(
                                f"{SCOPE_BASE_URL}{path}",
                                content=binary_content,
                                headers={"Content-Type": content_type},
                                timeout=60.0,  # Longer timeout for uploads
                            )
                        else:
                            response = await client.post(
                                f"{SCOPE_BASE_URL}{path}", json=body, timeout=30.0
                            )
                    elif method == "PATCH":
                        response = await client.patch(
                            f"{SCOPE_BASE_URL}{path}", json=body, timeout=30.0
                        )
                    elif method == "DELETE":
                        response = await client.delete(
                            f"{SCOPE_BASE_URL}{path}", timeout=30.0
                        )
                    else:
                        return {
                            "type": "api_response",
                            "request_id": request_id,
                            "status": 400,
                            "error": f"Unsupported method: {method}",
                        }

                    # Try to parse JSON response
                    try:
                        data = response.json()
                    except Exception:
                        data = response.text

                    return {
                        "type": "api_response",
                        "request_id": request_id,
                        "status": response.status_code,
                        "data": data,
                    }

                except httpx.TimeoutException:
                    return {
                        "type": "api_response",
                        "request_id": request_id,
                        "status": 504,
                        "error": "Request timeout",
                    }
                except Exception as e:
                    return {
                        "type": "api_response",
                        "request_id": request_id,
                        "status": 500,
                        "error": str(e),
                    }

        async def handle_message(payload: dict) -> dict | None:
            """Route message to appropriate handler based on type."""
            msg_type = payload.get("type")
            request_id = payload.get("request_id")

            if msg_type == "get_ice_servers":
                return await handle_get_ice_servers()
            elif msg_type == "offer":
                return await handle_offer(payload)
            elif msg_type == "icecandidate":
                return await handle_icecandidate(payload)
            elif msg_type == "api":
                return await handle_api_request(payload)
            elif msg_type == "ping":
                return {"type": "pong", "request_id": request_id}
            else:
                return {"type": "error", "request_id": request_id, "error": f"Unknown message type: {msg_type}"}

        # Main message loop
        try:
            while True:
                try:
                    message = await ws.receive_text()
                except RuntimeError:
                    break

                try:
                    payload = json.loads(message)
                except json.JSONDecodeError as e:
                    await safe_send_json(
                        {"type": "error", "error": f"Invalid JSON: {e}"}
                    )
                    continue

                # Handle the message
                response = await handle_message(payload)
                if response:
                    await safe_send_json(response)

        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await safe_send_json({"type": "error", "error": str(e)})
        finally:
            print("WebSocket connection closed")


# Deployment:
#   1. Run: fal run fal_app.py (for local testing)
#   2. Run: fal deploy fal_app.py (to deploy to fal.ai)
#   3. fal.ai will provide you with a WebSocket URL
#
# Client usage:
#   1. Connect to wss://<fal-url>/ws
#   2. Wait for {"type": "ready"}
#   3. Send {"type": "get_ice_servers"} to get ICE servers
#   4. Send {"type": "offer", "sdp": "...", "sdp_type": "offer"} for WebRTC offer
#   5. Receive {"type": "answer", "sdp": "...", "sessionId": "..."}
#   6. Exchange ICE candidates via {"type": "icecandidate", "candidate": {...}}
#   7. For API calls: {"type": "api", "method": "GET", "path": "/api/v1/pipeline/status"}
