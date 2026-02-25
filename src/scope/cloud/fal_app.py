"""
fal.ai deployment for Scope.

This runs the Scope backend and proxies WebRTC signaling + API calls through
a single WebSocket connection to avoid fal spawning new runners for each request.

Based on:
- https://docs.fal.ai/examples/serverless/deploy-models-with-custom-containers
- https://github.com/fal-ai-community/fal-demos/blob/main/fal_demos/video/yolo_webcam_webrtc/yolo.py
"""

import asyncio
import json
import os
import shutil
import subprocess as _subprocess
import time
import uuid
from typing import Any

import fal
from fal.container import ContainerImage
from fastapi import WebSocket


async def validate_user_access(user_id: str) -> tuple[bool, str]:
    """
    Validate that a user has access to cloud mode.

    Returns (is_valid, reason) tuple.
    """
    if not user_id:
        return False, "No user ID provided"
    return True, "Access granted"


class KafkaPublisher:
    """Async Kafka event publisher for fal.ai websocket events."""

    def __init__(self):
        self._producer = None
        self._started = False
        self._topic = None

    async def start(self) -> bool:
        """Start the Kafka producer."""
        # Read env vars at runtime (they may not be available at module load time on fal.ai)
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self._topic = os.getenv("KAFKA_TOPIC", "network_events")
        sasl_username = os.getenv("KAFKA_SASL_USERNAME")
        sasl_password = os.getenv("KAFKA_SASL_PASSWORD")

        print(
            f"[Kafka] Starting publisher (KAFKA_BOOTSTRAP_SERVERS={bootstrap_servers})"
        )
        if not bootstrap_servers:
            print("[Kafka] Not configured, event publishing disabled")
            return False

        try:
            from aiokafka import AIOKafkaProducer

            config = {
                "bootstrap_servers": bootstrap_servers,
                "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
                "key_serializer": lambda k: k.encode("utf-8") if k else None,
            }

            if sasl_username and sasl_password:
                import ssl

                ssl_context = ssl.create_default_context()
                config.update(
                    {
                        "security_protocol": "SASL_SSL",
                        "sasl_mechanism": "PLAIN",
                        "sasl_plain_username": sasl_username,
                        "sasl_plain_password": sasl_password,
                        "ssl_context": ssl_context,
                    }
                )

            self._producer = AIOKafkaProducer(**config)
            await self._producer.start()
            self._started = True
            print(f"[Kafka] ✅ Publisher started, topic: {self._topic}")
            return True

        except ImportError:
            print("[Kafka] ⚠️ aiokafka not installed, Kafka disabled")
            return False
        except Exception as e:
            print(f"[Kafka] ❌ Failed to start producer: {e}")
            return False

    async def stop(self):
        """Stop the Kafka producer."""
        if self._producer and self._started:
            try:
                await self._producer.stop()
                print("[Kafka] Publisher stopped")
            except Exception as e:
                print(f"[Kafka] Error stopping producer: {e}")
            finally:
                self._started = False
                self._producer = None

    async def publish(self, event_type: str, data: dict[str, Any]) -> bool:
        """Publish an event to Kafka."""
        if not self._started or not self._producer:
            return False

        event_id = str(uuid.uuid4())
        timestamp_ms = str(int(time.time() * 1000))

        event = {
            "id": event_id,
            "type": "stream_trace",
            "timestamp": timestamp_ms,
            "data": {
                "type": event_type,
                "client_source": "scope",
                "timestamp": timestamp_ms,
                **data,
            },
        }

        try:
            await self._producer.send_and_wait(self._topic, value=event, key=event_id)
            print(f"[Kafka] ✅ Published event: {event_type}")
            return True
        except Exception as e:
            print(f"[Kafka] ❌ Failed to publish event {event_type}: {e}")
            return False

    @property
    def is_running(self) -> bool:
        return self._started


# Global Kafka publisher instance
kafka_publisher: KafkaPublisher | None = None


ASSETS_DIR_PATH = "~/.daydream-scope/assets"
# Connection timeout settings
MAX_CONNECTION_DURATION_SECONDS = (
    3600  # Close connection after 60 minutes regardless of activity
)
TIMEOUT_CHECK_INTERVAL_SECONDS = 60  # Check for timeout every 60 seconds


def cleanup_session_data():
    """Clean up session-specific data when WebSocket disconnects.

    This prevents data leakage between users on fal.ai by clearing:
    - Assets directory (uploaded images, videos)
    - Recording files in temp directory
    """
    from pathlib import Path

    try:
        # Clean assets directory (matches DAYDREAM_SCOPE_ASSETS_DIR set in setup)
        assets_dir = Path(ASSETS_DIR_PATH).expanduser()
        if assets_dir.exists():
            for item in assets_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"Warning: Failed to delete {item}: {e}")
            print(f"Cleaned up assets directory: {assets_dir}")

    except Exception as e:
        print(f"Warning: Session cleanup failed: {e}")


# To deploy:
# 1. Ensure the docker image for your current git SHA has been built
#    (check https://github.com/daydreamlive/scope/actions for the docker-build workflow)
# 2. switch to python 3.10 to match the scope image
# 3. pip install fal
# 4. fal auth login
# 5. fal deploy --env (main/staging/prod) (--app-name X) fal_app.py --auth public

# Get git SHA at deploy time (this runs when the file is loaded during fal deploy)


def _get_git_sha() -> str:
    """Get the deploy tag from env var SCOPE_DEPLOY_TAG, or fall back to git SHA."""
    # Check for explicit deploy tag first
    deploy_tag = os.environ.get("SCOPE_DEPLOY_TAG")
    if deploy_tag:
        return deploy_tag

    # Fall back to git SHA
    try:
        result = _subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Warning: Could not get git SHA: {e}")
        return "unknown"


GIT_SHA = _get_git_sha()

# Configuration - uses git SHA from current checkout
DOCKER_IMAGE = f"daydreamlive/scope:{GIT_SHA}"

# Create a Dockerfile that uses your existing image as base
dockerfile_str = f"""
FROM {DOCKER_IMAGE}

"""

# Create container image from Dockerfile string
custom_image = ContainerImage.from_dockerfile_str(
    dockerfile_str,
)


class ScopeApp(fal.App, keep_alive=300):
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
        "aiokafka",  # For Kafka event publishing
    ]

    auth_mode = "public"

    def setup(self):
        """
        Start the Scope backend server as a background process.
        """
        import os
        import subprocess
        import threading
        import time

        print(f"Starting Scope container setup... (version: {GIT_SHA})")

        # Verify GPU is available
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            print(f"GPU Status:\n{result.stdout}")
        except Exception as e:
            print(f"GPU check failed: {e}")
            raise

        # Environment for scope
        scope_env = os.environ.copy()
        # Add any scope-specific environment variables here
        # scope_env["PIPELINE"] = "some-default-pipeline"
        # Use fal's /data directory for persistent storage
        scope_env["DAYDREAM_SCOPE_MODELS_DIR"] = "/data/models"
        scope_env["DAYDREAM_SCOPE_LOGS_DIR"] = "/data/logs"
        # not shared between users
        scope_env["DAYDREAM_SCOPE_ASSETS_DIR"] = ASSETS_DIR_PATH
        scope_env["DAYDREAM_SCOPE_LORA_DIR"] = ASSETS_DIR_PATH + "/lora"

        # Install kafka extra dependencies
        print("Installing daydream-scope[kafka]...")
        try:
            subprocess.run(
                ["uv", "pip", "install", "daydream-scope[kafka]"],
                check=True,
                env=scope_env,
            )
            print("✅ daydream-scope[kafka] installed")
        except Exception as e:
            print(f"Failed to install daydream-scope[kafka]: {e}")

        # Start the scope server in a background thread
        def start_server():
            print("Starting Scope server...")
            try:
                result = subprocess.run(
                    [
                        "uv",
                        "run",
                        "daydream-scope",
                        "--no-browser",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "8000",
                    ],
                    env=scope_env,
                )
                # Log when process exits (regardless of exit code)
                if result.returncode == 0:
                    print("Scope server process exited normally (exit code 0)")
                else:
                    print(
                        f"❌ Scope server process exited with code {result.returncode}"
                    )
            except Exception as e:
                print(f"❌ Failed to start Scope server: {e}")
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
            print(
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
        import json
        import uuid

        import httpx
        from starlette.websockets import WebSocketDisconnect, WebSocketState

        SCOPE_BASE_URL = "http://localhost:8000"

        # Initialize Kafka publisher if not already done
        global kafka_publisher
        if kafka_publisher is None:
            kafka_publisher = KafkaPublisher()
            await kafka_publisher.start()

        await ws.accept()

        # Generate a unique connection ID for this WebSocket session
        connection_id = str(uuid.uuid4())[:8]  # Short ID for readability in logs
        # User ID for log correlation (set via set_user_id message)
        user_id = None

        def log_prefix() -> str:
            """Get log prefix - uses user_id if set, otherwise connection_id."""
            if user_id:
                return f"{user_id}:{connection_id}"
            return connection_id

        print(f"[{log_prefix()}] ✅ WebSocket connection accepted")

        # Send ready message with connection_id
        await ws.send_json({"type": "ready", "connection_id": connection_id})

        # Track WebRTC session ID for ICE candidate routing
        session_id = None

        # Track connection start time for max duration timeout
        connection_start_time = time.time()

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

        async def check_max_duration_exceeded() -> bool:
            """Check if connection has exceeded max duration. Returns True if should close."""
            elapsed_seconds = time.time() - connection_start_time
            if elapsed_seconds >= MAX_CONNECTION_DURATION_SECONDS:
                print(
                    f"[{log_prefix()}] Closing due to max duration ({elapsed_seconds:.0f}s)"
                )
                await safe_send_json(
                    {
                        "type": "closing",
                        "reason": "max_duration",
                        "elapsed_seconds": elapsed_seconds,
                    }
                )
                return True
            return False

        async def handle_get_ice_servers(payload: dict):
            """Proxy GET /api/v1/webrtc/ice-servers"""
            request_id = payload.get("request_id")
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SCOPE_BASE_URL}/api/v1/webrtc/ice-servers"
                )
                return {
                    "type": "ice_servers",
                    "request_id": request_id,
                    "data": response.json(),
                    "status": response.status_code,
                }

        # Parse fal_log_labels as JSON if possible, otherwise use raw string
        fal_log_labels_raw = os.getenv("FAL_LOG_LABELS", "unknown")
        try:
            fal_log_labels = json.loads(fal_log_labels_raw)
        except (json.JSONDecodeError, TypeError):
            fal_log_labels = fal_log_labels_raw

        # Build connection_info with GPU type and any available infrastructure info
        connection_info = {
            "gpu_type": ScopeApp.machine_type,
            "fal_region": os.getenv("NOMAD_DC", "unknown"),
            "fal_runner_id": os.getenv(
                "FAL_JOB_ID", os.getenv("FAL_RUNNER_ID", "unknown")
            ),
            "fal_log_labels": fal_log_labels,
        }

        async def handle_offer(payload: dict):
            """Proxy POST /api/v1/webrtc/offer"""
            nonlocal session_id
            request_id = payload.get("request_id")

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{SCOPE_BASE_URL}/api/v1/webrtc/offer",
                        json={
                            "sdp": payload.get("sdp"),
                            "type": payload.get("sdp_type", "offer"),
                            "initialParameters": payload.get("initialParameters"),
                            "user_id": payload.get("user_id"),
                            "connection_id": connection_id,
                            "connection_info": connection_info,
                        },
                        timeout=30.0,
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
            except (httpx.TimeoutException, TimeoutError):
                return {
                    "type": "error",
                    "request_id": request_id,
                    "error": "WebRTC offer timeout - Scope server may be overloaded",
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
                return {
                    "type": "icecandidate_ack",
                    "request_id": request_id,
                    "status": "end_of_candidates",
                }

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
                    return {
                        "type": "icecandidate_ack",
                        "request_id": request_id,
                        "status": "ok",
                    }
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

            # Block plugin installation in cloud mode (security: prevent arbitrary code execution)
            if method == "POST" and path == "/api/v1/plugins":
                return {
                    "type": "api_response",
                    "request_id": request_id,
                    "status": 403,
                    "error": "Plugin installation is not available in cloud mode",
                }

            # Inject connection_id into pipeline load requests for event correlation
            if (
                method == "POST"
                and path == "/api/v1/pipeline/load"
                and isinstance(body, dict)
            ):
                body["connection_id"] = connection_id
                body["connection_info"] = connection_info
                body["user_id"] = user_id

            async with httpx.AsyncClient() as client:
                try:
                    # Check if this is a base64-encoded file upload
                    is_binary_upload = (
                        body and isinstance(body, dict) and "_base64_content" in body
                    )

                    if method == "GET":
                        # Use longer timeout for potential binary downloads (recordings)
                        timeout = 120.0 if "/recordings/" in path else 30.0
                        response = await client.get(
                            f"{SCOPE_BASE_URL}{path}", timeout=timeout
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
                            # Use longer timeout for LoRA installs
                            post_timeout = 300.0 if path == "/api/v1/loras" else 30.0
                            response = await client.post(
                                f"{SCOPE_BASE_URL}{path}",
                                json=body,
                                timeout=post_timeout,
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

                    # Check if response is binary (e.g., video/mp4 download)
                    content_type = response.headers.get("content-type", "")
                    is_binary_response = any(
                        ct in content_type
                        for ct in [
                            "video/",
                            "audio/",
                            "application/octet-stream",
                            "image/",
                        ]
                    )

                    if is_binary_response and response.status_code == 200:
                        # Base64 encode binary content for JSON transport
                        binary_content = response.content
                        encoded = base64.b64encode(binary_content).decode("utf-8")
                        return {
                            "type": "api_response",
                            "request_id": request_id,
                            "status": response.status_code,
                            "_base64_content": encoded,
                            "_content_type": content_type,
                            "_content_length": len(binary_content),
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
            nonlocal user_id
            msg_type = payload.get("type")
            request_id = payload.get("request_id")

            # Reject all messages until user_id is set (except set_user_id itself)
            if user_id is None and msg_type != "set_user_id":
                print(
                    f"[{connection_id}] Rejecting message type '{msg_type}' - user_id not set yet"
                )
                return None

            if msg_type == "set_user_id":
                requested_user_id = payload.get("user_id")

                # Validate user has access to cloud mode
                is_valid, reason = await validate_user_access(requested_user_id)
                if not is_valid:
                    print(f"[{log_prefix()}] Access denied: {reason}")
                    await safe_send_json(
                        {
                            "type": "error",
                            "error": f"Access denied: {reason}",
                            "code": "ACCESS_DENIED",
                        }
                    )
                    await ws.close(code=4003, reason="Access denied")
                    return None

                user_id = requested_user_id
                print(f"[{log_prefix()}] User ID set, access granted")
                # Publish websocket connected event with user_id
                if kafka_publisher and kafka_publisher.is_running:
                    await kafka_publisher.publish(
                        "websocket_connected",
                        {
                            "user_id": user_id,
                            "connection_id": connection_id,
                            "connection_info": connection_info,
                        },
                    )
                return {"type": "user_id_set", "user_id": user_id}
            elif msg_type == "get_ice_servers":
                return await handle_get_ice_servers(payload)
            elif msg_type == "offer":
                return await handle_offer(payload)
            elif msg_type == "icecandidate":
                return await handle_icecandidate(payload)
            elif msg_type == "api":
                return await handle_api_request(payload)
            elif msg_type == "ping":
                return {"type": "pong", "request_id": request_id}
            else:
                return {
                    "type": "error",
                    "request_id": request_id,
                    "error": f"Unknown message type: {msg_type}",
                }

        # Main message loop
        try:
            while True:
                try:
                    # Use timeout on receive to periodically check connection duration
                    message = await asyncio.wait_for(
                        ws.receive_text(), timeout=TIMEOUT_CHECK_INTERVAL_SECONDS
                    )
                except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
                    if await check_max_duration_exceeded():
                        break
                    continue
                except RuntimeError:
                    break

                # Check duration on each message as well (in case of constant activity)
                if await check_max_duration_exceeded():
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
            print(f"[{log_prefix()}] WebSocket disconnected")
        except Exception as e:
            print(f"[{log_prefix()}] WebSocket error ({type(e).__name__}): {e}")
            await safe_send_json({"type": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            # Publish websocket disconnected event
            if kafka_publisher and kafka_publisher.is_running:
                end_time = time.time()
                elapsed_ms = int((end_time - connection_start_time) * 1000)
                await kafka_publisher.publish(
                    "websocket_disconnected",
                    {
                        "user_id": user_id,
                        "connection_id": connection_id,
                        "connection_info": connection_info,
                        "duration_ms": elapsed_ms,
                        "session_start_time_ms": int(connection_start_time * 1000),
                        "session_end_time_ms": int(end_time * 1000),
                    },
                )
            # Clean up session data to prevent data leakage between users
            cleanup_session_data()
            print(
                f"[{log_prefix()}] WebSocket connection closed, session data cleaned up"
            )


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
