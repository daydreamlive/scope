"""
fal.ai deployment for Scope.

This runs the Scope backend and proxies WebRTC signaling + API calls through
a single WebSocket connection to avoid fal spawning new runners for each request.

Uses fal's @realtime decorator for structured WebSocket handling with Pydantic models.

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
from typing import Annotated, Any, AsyncIterator, Literal

import fal
from fal.container import ContainerImage
from pydantic import BaseModel, Field, RootModel


# =============================================================================
# Pydantic Models for WebSocket Protocol
# =============================================================================


class SetUserIdInput(BaseModel):
    """Client sends this to authenticate and set their user ID."""

    type: Literal["set_user_id"]
    user_id: str
    request_id: str | None = None


class GetIceServersInput(BaseModel):
    """Request ICE servers for WebRTC connection."""

    type: Literal["get_ice_servers"]
    request_id: str | None = None


class OfferInput(BaseModel):
    """WebRTC SDP offer from client."""

    type: Literal["offer"]
    sdp: str
    sdp_type: str = "offer"
    initialParameters: dict | None = None
    user_id: str | None = None
    request_id: str | None = None


class IceCandidateData(BaseModel):
    """ICE candidate data structure."""

    candidate: str | None = None
    sdpMid: str | None = None
    sdpMLineIndex: int | None = None


class IceCandidateInput(BaseModel):
    """ICE candidate from client."""

    type: Literal["icecandidate"]
    candidate: IceCandidateData | None = None
    sessionId: str | None = None
    request_id: str | None = None


class ApiRequestInput(BaseModel):
    """Proxied API request."""

    type: Literal["api"]
    method: str = "GET"
    path: str
    body: dict | None = None
    request_id: str | None = None


class PingInput(BaseModel):
    """Keepalive ping."""

    type: Literal["ping"]
    request_id: str | None = None


# Discriminated union of all input message types
RealtimeInputMessage = Annotated[
    SetUserIdInput
    | GetIceServersInput
    | OfferInput
    | IceCandidateInput
    | ApiRequestInput
    | PingInput,
    Field(discriminator="type"),
]


class RealtimeInput(RootModel):
    """Root model for incoming WebSocket messages."""

    root: RealtimeInputMessage


# Output message types


class ReadyOutput(BaseModel):
    """Sent immediately after connection is established."""

    type: Literal["ready"]
    connection_id: str


class UserIdSetOutput(BaseModel):
    """Confirms user ID was set successfully."""

    type: Literal["user_id_set"]
    user_id: str


class IceServersOutput(BaseModel):
    """ICE servers response."""

    type: Literal["ice_servers"]
    request_id: str | None = None
    data: dict
    status: int


class AnswerOutput(BaseModel):
    """WebRTC SDP answer."""

    type: Literal["answer"]
    request_id: str | None = None
    sdp: str
    sdp_type: str
    sessionId: str | None = None


class IceCandidateAckOutput(BaseModel):
    """ICE candidate acknowledgment."""

    type: Literal["icecandidate_ack"]
    request_id: str | None = None
    status: str


class ApiResponseOutput(BaseModel):
    """API response for proxied requests."""

    type: Literal["api_response"]
    request_id: str | None = None
    status: int
    data: Any | None = None
    error: str | None = None
    # For binary responses
    _base64_content: str | None = Field(None, alias="_base64_content")
    _content_type: str | None = Field(None, alias="_content_type")
    _content_length: int | None = Field(None, alias="_content_length")


class PongOutput(BaseModel):
    """Keepalive pong response."""

    type: Literal["pong"]
    request_id: str | None = None


class ErrorOutput(BaseModel):
    """Error response."""

    type: Literal["error"]
    error: str
    code: str | None = None
    request_id: str | None = None
    detail: str | None = None


RealtimeOutputMessage = Annotated[
    ReadyOutput
    | UserIdSetOutput
    | IceServersOutput
    | AnswerOutput
    | IceCandidateAckOutput
    | ApiResponseOutput
    | PongOutput
    | ErrorOutput,
    Field(discriminator="type"),
]


class RealtimeOutput(RootModel):
    """Root model for outgoing WebSocket messages."""

    root: RealtimeOutputMessage


# =============================================================================
# User Validation
# =============================================================================


async def validate_user_access(user_id: str) -> tuple[bool, str]:
    """
    Validate that a user has access to cloud mode.

    Returns (is_valid, reason) tuple.
    """
    import urllib.error
    import urllib.request

    if not user_id:
        return False, "No user ID provided"

    url = f"{os.getenv('DAYDREAM_API_BASE', 'https://api.daydream.live')}/v1/users/{user_id}"
    print(f"Validating user access for {user_id} via {url}")

    def fetch_user():
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    try:
        # Run synchronous urllib in thread pool to not block event loop
        await asyncio.get_event_loop().run_in_executor(None, fetch_user)
        return True, "Access granted"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False, "User not found"
        return False, f"Failed to fetch user: {e.code}"
    except Exception as e:
        return False, f"Error validating user: {e}"


# =============================================================================
# Kafka Publisher
# =============================================================================


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


# =============================================================================
# Configuration
# =============================================================================

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
        return result.stdout.strip()[:7]
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


# =============================================================================
# Main Application
# =============================================================================


class ScopeApp(fal.App, keep_alive=300):
    """
    Scope server on fal.ai.

    This runs the Scope backend as a subprocess and exposes a realtime WebSocket
    endpoint that handles:
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
        import subprocess
        import threading
        import time as time_module

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
        start_time = time_module.time()

        while time_module.time() - start_time < max_wait:
            try:
                import requests

                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Scope server is running on port 8000")
                    break
            except Exception:
                pass
            time_module.sleep(2)
        else:
            print(
                f"Scope server health check timed out after {max_wait}s, continuing anyway..."
            )

        print("Scope container setup complete")

    @fal.realtime("/ws", buffering="none")
    async def websocket_handler(
        self, inputs: AsyncIterator[RealtimeInput]
    ) -> AsyncIterator[RealtimeOutput]:
        """
        Main WebSocket endpoint using fal's realtime decorator.

        Handles:
        1. WebRTC signaling (offer/answer, ICE candidates)
        2. REST API call proxying

        Protocol:
        - All messages are JSON with a "type" field (enforced via Pydantic models)
        - WebRTC signaling types: "get_ice_servers", "offer", "icecandidate"
        - API proxy type: "api" with "method", "path", "body" fields

        This keeps a persistent connection to prevent fal from spawning new runners.
        """
        import base64

        import httpx

        SCOPE_BASE_URL = "http://localhost:8000"

        # Initialize Kafka publisher if not already done
        global kafka_publisher
        if kafka_publisher is None:
            kafka_publisher = KafkaPublisher()
            await kafka_publisher.start()

        # Generate a unique connection ID for this WebSocket session
        connection_id = str(uuid.uuid4())[:8]  # Short ID for readability in logs
        # User ID for log correlation (set via set_user_id message)
        user_id: str | None = None
        # Track WebRTC session ID for ICE candidate routing
        session_id: str | None = None
        # Track connection start time for max duration timeout
        connection_start_time = time.time()

        def log_prefix() -> str:
            """Get log prefix - uses user_id if set, otherwise connection_id."""
            if user_id:
                return f"{user_id}:{connection_id}"
            return connection_id

        print(f"[{log_prefix()}] ✅ WebSocket connection accepted (realtime)")

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

        # Send ready message immediately
        yield RealtimeOutput(
            root=ReadyOutput(type="ready", connection_id=connection_id)
        )

        # Helper functions for handling different message types

        async def handle_get_ice_servers(
            payload: GetIceServersInput,
        ) -> RealtimeOutputMessage:
            """Proxy GET /api/v1/webrtc/ice-servers"""
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SCOPE_BASE_URL}/api/v1/webrtc/ice-servers"
                )
                return IceServersOutput(
                    type="ice_servers",
                    request_id=payload.request_id,
                    data=response.json(),
                    status=response.status_code,
                )

        async def handle_offer(payload: OfferInput) -> RealtimeOutputMessage:
            """Proxy POST /api/v1/webrtc/offer"""
            nonlocal session_id
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{SCOPE_BASE_URL}/api/v1/webrtc/offer",
                        json={
                            "sdp": payload.sdp,
                            "type": payload.sdp_type,
                            "initialParameters": payload.initialParameters,
                            "user_id": payload.user_id,
                            "connection_id": connection_id,
                            "connection_info": connection_info,
                        },
                        timeout=30.0,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        session_id = data.get("sessionId")
                        return AnswerOutput(
                            type="answer",
                            request_id=payload.request_id,
                            sdp=data.get("sdp", ""),
                            sdp_type=data.get("type", "answer"),
                            sessionId=session_id,
                        )
                    else:
                        return ErrorOutput(
                            type="error",
                            request_id=payload.request_id,
                            error=f"Offer failed: {response.status_code}",
                            detail=response.text,
                        )
            except (httpx.TimeoutException, TimeoutError):
                return ErrorOutput(
                    type="error",
                    request_id=payload.request_id,
                    error="WebRTC offer timeout - Scope server may be overloaded",
                )

        async def handle_icecandidate(
            payload: IceCandidateInput,
        ) -> RealtimeOutputMessage:
            """Proxy PATCH /api/v1/webrtc/offer/{session_id} for ICE candidates"""
            target_session = payload.sessionId or session_id

            if not target_session:
                return ErrorOutput(
                    type="error",
                    request_id=payload.request_id,
                    error="No session ID available for ICE candidate",
                )

            if payload.candidate is None:
                # End of candidates signal
                return IceCandidateAckOutput(
                    type="icecandidate_ack",
                    request_id=payload.request_id,
                    status="end_of_candidates",
                )

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{SCOPE_BASE_URL}/api/v1/webrtc/offer/{target_session}",
                    json={
                        "candidates": [
                            {
                                "candidate": payload.candidate.candidate,
                                "sdpMid": payload.candidate.sdpMid,
                                "sdpMLineIndex": payload.candidate.sdpMLineIndex,
                            }
                        ]
                    },
                    timeout=10.0,
                )

                if response.status_code == 204:
                    return IceCandidateAckOutput(
                        type="icecandidate_ack",
                        request_id=payload.request_id,
                        status="ok",
                    )
                else:
                    return ErrorOutput(
                        type="error",
                        request_id=payload.request_id,
                        error=f"ICE candidate failed: {response.status_code}",
                        detail=response.text,
                    )

        async def handle_api_request(payload: ApiRequestInput) -> RealtimeOutputMessage:
            """
            Proxy arbitrary API requests to Scope backend.

            Special handling for file uploads:
            If body contains "_base64_content", it's decoded and sent as binary.
            """
            method = payload.method.upper()
            path = payload.path
            body = payload.body

            # Block plugin installation in cloud mode (security: prevent arbitrary code execution)
            if method == "POST" and path == "/api/v1/plugins":
                return ApiResponseOutput(
                    type="api_response",
                    request_id=payload.request_id,
                    status=403,
                    error="Plugin installation is not available in cloud mode",
                )

            # Inject connection_id into pipeline load requests for event correlation
            if method == "POST" and path == "/api/v1/pipeline/load" and body:
                body = dict(body)  # Make a copy
                body["connection_id"] = connection_id
                body["connection_info"] = connection_info
                body["user_id"] = user_id

            async with httpx.AsyncClient() as client:
                try:
                    # Check if this is a base64-encoded file upload
                    is_binary_upload = body and "_base64_content" in body

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
                        return ApiResponseOutput(
                            type="api_response",
                            request_id=payload.request_id,
                            status=400,
                            error=f"Unsupported method: {method}",
                        )

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
                        # Return as dict to preserve underscore-prefixed fields
                        return ApiResponseOutput(
                            type="api_response",
                            request_id=payload.request_id,
                            status=response.status_code,
                            data={
                                "_base64_content": encoded,
                                "_content_type": content_type,
                                "_content_length": len(binary_content),
                            },
                        )

                    # Try to parse JSON response
                    try:
                        data = response.json()
                    except Exception:
                        data = response.text

                    return ApiResponseOutput(
                        type="api_response",
                        request_id=payload.request_id,
                        status=response.status_code,
                        data=data,
                    )

                except httpx.TimeoutException:
                    return ApiResponseOutput(
                        type="api_response",
                        request_id=payload.request_id,
                        status=504,
                        error="Request timeout",
                    )
                except Exception as e:
                    return ApiResponseOutput(
                        type="api_response",
                        request_id=payload.request_id,
                        status=500,
                        error=str(e),
                    )

        # Main message processing loop
        try:
            async for input_msg in inputs:
                # Check max duration
                elapsed_seconds = time.time() - connection_start_time
                if elapsed_seconds >= MAX_CONNECTION_DURATION_SECONDS:
                    print(
                        f"[{log_prefix()}] Closing due to max duration ({elapsed_seconds:.0f}s)"
                    )
                    yield RealtimeOutput(
                        root=ErrorOutput(
                            type="error",
                            error="Max duration exceeded",
                            code="MAX_DURATION_EXCEEDED",
                        )
                    )
                    break

                # Get the actual message from the RootModel
                payload = input_msg.root

                # Reject all messages until user_id is set (except set_user_id itself)
                if user_id is None and not isinstance(payload, SetUserIdInput):
                    print(
                        f"[{connection_id}] Rejecting message type '{payload.type}' - user_id not set yet"
                    )
                    continue

                # Handle message based on type
                if isinstance(payload, SetUserIdInput):
                    # Validate user has access to cloud mode
                    is_valid, reason = await validate_user_access(payload.user_id)
                    if not is_valid:
                        print(f"[{log_prefix()}] Access denied: {reason}")
                        yield RealtimeOutput(
                            root=ErrorOutput(
                                type="error",
                                error="Access denied",
                                code="ACCESS_DENIED",
                            )
                        )
                        # Note: With @fal.realtime, we can't close with a custom code
                        # The connection will close when we stop yielding
                        break

                    user_id = payload.user_id
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

                    yield RealtimeOutput(
                        root=UserIdSetOutput(type="user_id_set", user_id=user_id)
                    )

                elif isinstance(payload, GetIceServersInput):
                    result = await handle_get_ice_servers(payload)
                    yield RealtimeOutput(root=result)

                elif isinstance(payload, OfferInput):
                    result = await handle_offer(payload)
                    yield RealtimeOutput(root=result)

                elif isinstance(payload, IceCandidateInput):
                    result = await handle_icecandidate(payload)
                    yield RealtimeOutput(root=result)

                elif isinstance(payload, ApiRequestInput):
                    result = await handle_api_request(payload)
                    yield RealtimeOutput(root=result)

                elif isinstance(payload, PingInput):
                    yield RealtimeOutput(
                        root=PongOutput(type="pong", request_id=payload.request_id)
                    )

                else:
                    yield RealtimeOutput(
                        root=ErrorOutput(
                            type="error",
                            error=f"Unknown message type: {getattr(payload, 'type', 'unknown')}",
                        )
                    )

        except Exception as e:
            print(f"[{log_prefix()}] WebSocket error ({type(e).__name__}): {e}")
            yield RealtimeOutput(
                root=ErrorOutput(type="error", error=f"{type(e).__name__}: {e}")
            )

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
#   3. Send {"type": "set_user_id", "user_id": "..."} to authenticate
#   4. Send {"type": "get_ice_servers"} to get ICE servers
#   5. Send {"type": "offer", "sdp": "...", "sdp_type": "offer"} for WebRTC offer
#   6. Receive {"type": "answer", "sdp": "...", "sessionId": "..."}
#   7. Exchange ICE candidates via {"type": "icecandidate", "candidate": {...}}
#   8. For API calls: {"type": "api", "method": "GET", "path": "/api/v1/pipeline/status"}
