"""CloudConnectionManager - manages WebSocket connection to cloud.ai at app level.

This module handles the connection to cloud.ai cloud for remote GPU inference.
The connection is established when cloud mode is toggled ON and stays open
until toggled OFF, allowing:
1. API calls to be proxied to the cloud-hosted scope backend
2. WebRTC media relay - video flows through the backend to cloud.ai
3. The cloud runner to stay warm and ready for requests

Architecture:
- CloudConnectionManager is instantiated once at app startup
- When connect() is called, it opens a WebSocket to cloud and waits for "ready"
- When start_webrtc() is called, it establishes a WebRTC connection to cloud.ai
- Video frames flow: Browser/Spout → Backend → fal.ai → Backend → Browser/Spout
- When disconnect() is called, both WebSocket and WebRTC are closed
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import aiohttp
import numpy as np
from av import VideoFrame

if TYPE_CHECKING:
    from .cloud_webrtc_client import CloudWebRTCClient

logger = logging.getLogger(__name__)

TOKEN_EXPIRATION_SECONDS = 120


class CloudConnectionManager:
    """Manages the WebSocket connection to cloud.ai cloud.

    This is a singleton-style manager that handles:
    - WebSocket connection lifecycle (connect on cloud mode ON, disconnect on OFF)
    - API request proxying through the WebSocket
    - Request/response correlation via request_id
    """

    def __init__(self):
        self.app_id: str | None = None
        self.api_key: str | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._receive_task: asyncio.Task | None = None
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._connected = False
        self._stop_event = asyncio.Event()
        # Connection ID from cloud.ai for log correlation
        self._connection_id: str | None = None
        # User ID for log correlation (from fal token)
        self._user_id: str | None = None

        # WebRTC client for media relay
        self._webrtc_client: CloudWebRTCClient | None = None
        self._frame_callbacks: list[Callable[[VideoFrame], None]] = []

        # Stats tracking
        self._stats = {
            "webrtc_offers_sent": 0,
            "webrtc_offers_successful": 0,
            "webrtc_ice_candidates_sent": 0,
            "api_requests_sent": 0,
            "api_requests_successful": 0,
            "connected_at": None,
            "last_activity_at": None,
            "frames_sent_to_fal": 0,
            "frames_received_from_fal": 0,
        }

    @property
    def is_connected(self) -> bool:
        """Check if connected to cloud."""
        return self._connected and self.ws is not None and not self.ws.closed

    async def connect(
        self, app_id: str, api_key: str, user_id: str | None = None
    ) -> None:
        """Connect to cloud.

        Args:
            app_id: The cloud app ID (e.g., "username/scope-app")
            api_key: The cloud API key
            user_id: Optional user ID for log correlation

        Raises:
            RuntimeError: If connection fails or times out
        """
        if self.is_connected:
            logger.info("Already connected to cloud, disconnecting first")
            await self.disconnect()

        self.app_id = app_id
        self.api_key = api_key
        self._user_id = user_id
        self._stop_event.clear()

        # Get temporary token
        # token = await self._get_temporary_token()
        token = "foo"

        # Build WebSocket URL
        ws_url = self._build_ws_url(token)
        logger.info(f"Connecting to cloud WebSocket: {ws_url.split('?')[0]}...")

        # Create session and connect
        self._session = aiohttp.ClientSession()
        try:
            self.ws = await asyncio.wait_for(
                self._session.ws_connect(ws_url),
                timeout=30.0,
            )
        except TimeoutError:
            await self._cleanup_session()
            raise RuntimeError(
                f"Timeout connecting to cloud WebSocket. Check that app_id '{app_id}' "
                "is correct and the cloud app is deployed."
            ) from None
        except Exception as e:
            await self._cleanup_session()
            raise RuntimeError(f"Failed to connect to cloud WebSocket: {e}") from e

        # Wait for "ready" message
        try:
            msg = await asyncio.wait_for(self.ws.receive(), timeout=180.0)
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") != "ready":
                    raise RuntimeError(f"Expected 'ready' message, got: {data}")
                # Extract connection_id for log correlation
                self._connection_id = data.get("connection_id")
            else:
                raise RuntimeError(f"Unexpected message type: {msg.type}")
        except TimeoutError:
            await self._cleanup()
            raise RuntimeError(
                "Timeout waiting for 'ready' from cloud server. "
                "The cloud runner may be starting up (cold start can take 1-2 minutes)."
            ) from None

        logger.info(f"Cloud server ready (connection_id: {self._connection_id})")
        self._connected = True
        self._stats["connected_at"] = time.time()
        self._stats["last_activity_at"] = time.time()

        # Send user_id to cloud for log correlation
        if self._user_id:
            await self.ws.send_json({"type": "set_user_id", "user_id": self._user_id})
            logger.info(f"Sent user_id to cloud: {self._user_id}")

        # Reset stats on new connection
        self._stats["webrtc_offers_sent"] = 0
        self._stats["webrtc_offers_successful"] = 0
        self._stats["webrtc_ice_candidates_sent"] = 0
        self._stats["api_requests_sent"] = 0
        self._stats["api_requests_successful"] = 0

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _get_temporary_token(self) -> str:
        """Get temporary JWT token from cloud API."""
        if not self.api_key or not self.app_id:
            raise RuntimeError("API key and app_id must be set before getting token")

        # Extract alias from app_id (e.g., "owner/app-name" -> "app-name")
        parts = self.app_id.split("/")
        alias = parts[1] if len(parts) >= 2 else self.app_id

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://rest.alpha.fal.ai/tokens/",
                headers={
                    "Authorization": f"Key {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "allowed_apps": [alias],
                    "token_expiration": TOKEN_EXPIRATION_SECONDS,
                },
            ) as resp:
                if not resp.ok:
                    error_body = await resp.text()
                    raise RuntimeError(
                        f"Token request failed: {resp.status} {error_body}"
                    )
                token_data = await resp.json()
                # Handle both string and object responses
                if isinstance(token_data, dict) and "detail" in token_data:
                    return token_data["detail"]
                return token_data

    def _build_ws_url(self, token: str) -> str:
        """Build WebSocket URL with JWT token."""
        app_id = self.app_id.strip("/") if self.app_id else ""
        # Ensure we're connecting to the /ws endpoint
        if not app_id.endswith("/ws"):
            app_id = f"{app_id}/ws"
        return f"wss://fal.run/{app_id}"

    async def _receive_loop(self) -> None:
        """Receive and route WebSocket messages."""
        if self.ws is None:
            return

        try:
            while not self._stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(
                        self.ws.receive(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_message(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON message from cloud: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("Cloud WebSocket closed")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Cloud WebSocket error: {self.ws.exception()}")
                    break

        except Exception as e:
            logger.error(f"Receive loop error: {e}")
        finally:
            self._connected = False

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming WebSocket message."""
        request_id = data.get("request_id")

        if request_id and request_id in self._pending_requests:
            # This is a response to a pending request
            future = self._pending_requests.pop(request_id)
            if not future.done():
                future.set_result(data)
        else:
            # Unsolicited message (e.g., notifications)
            msg_type = data.get("type")
            logger.debug(f"Received unsolicited message: {msg_type}")

    async def send_and_wait(
        self,
        message: dict,
        timeout: float = 30.0,
    ) -> dict:
        """Send a message and wait for the correlated response.

        Args:
            message: The message to send (request_id will be added)
            timeout: Timeout in seconds

        Returns:
            The response message

        Raises:
            RuntimeError: If not connected or request times out
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to cloud")

        # Add request_id for correlation
        request_id = str(uuid.uuid4())
        message["request_id"] = request_id

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            # Send message
            await self.ws.send_json(message)

            # Wait for response
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise RuntimeError(f"Request timeout after {timeout}s") from None
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise RuntimeError(f"Request failed: {e}") from e

    async def api_request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: float = 30.0,
    ) -> dict:
        """Make an API request through the Cloud WebSocket proxy.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            path: API path (e.g., "/api/v1/pipeline/load")
            body: Request body for POST/PATCH
            timeout: Timeout in seconds

        Returns:
            Response dict with "status" and "data" or "error"
        """
        self._stats["api_requests_sent"] += 1
        self._stats["last_activity_at"] = time.time()
        logger.info(f"[CLOUD] API request: {method} {path}")

        message = {
            "type": "api",
            "method": method.upper(),
            "path": path,
        }
        if body is not None:
            message["body"] = body

        response = await self.send_and_wait(message, timeout=timeout)

        # Check for error in response
        if response.get("type") == "error":
            logger.error(f"[CLOUD] API request failed: {response.get('error')}")
            raise RuntimeError(response.get("error", "Unknown error"))

        self._stats["api_requests_successful"] += 1
        status = response.get("status", 200)
        logger.info(f"[CLOUD] API response: {status} for {method} {path}")

        return response

    async def webrtc_get_ice_servers(self) -> dict:
        """Get ICE servers from cloud-hosted scope backend."""
        logger.info("[CLOUD] Fetching ICE servers from cloud")
        self._stats["last_activity_at"] = time.time()
        response = await self.send_and_wait({"type": "get_ice_servers"})
        ice_servers = response.get("data", {})
        logger.info(
            f"[CLOUD] Got {len(ice_servers.get('iceServers', []))} ICE servers from cloud"
        )
        return ice_servers

    async def webrtc_offer(
        self,
        sdp: str,
        sdp_type: str = "offer",
        initial_parameters: dict | None = None,
    ) -> dict:
        """Send WebRTC offer to cloud-hosted scope backend.

        Returns:
            Dict with "sdp", "sdp_type", and "sessionId"
        """
        self._stats["webrtc_offers_sent"] += 1
        self._stats["last_activity_at"] = time.time()
        logger.info(
            f"[CLOUD] Sending WebRTC offer to cloud (offer #{self._stats['webrtc_offers_sent']})"
        )

        message: dict[str, Any] = {
            "type": "offer",
            "sdp": sdp,
            "sdp_type": sdp_type,
        }
        if initial_parameters:
            message["initialParameters"] = initial_parameters
            logger.info(
                f"[CLOUD] Offer includes initial parameters: {list(initial_parameters.keys())}"
            )

        response = await self.send_and_wait(message, timeout=30.0)

        if response.get("type") == "error":
            logger.error(f"[CLOUD] WebRTC offer failed: {response.get('error')}")
            raise RuntimeError(response.get("error", "Offer failed"))

        self._stats["webrtc_offers_successful"] += 1
        session_id = response.get("sessionId")
        logger.info(f"[CLOUD] WebRTC offer successful! Session ID: {session_id}")
        logger.info(
            f"[CLOUD] Stats: {self._stats['webrtc_offers_successful']}/{self._stats['webrtc_offers_sent']} offers successful"
        )

        return {
            "sdp": response.get("sdp"),
            "type": response.get("sdp_type"),
            "sessionId": session_id,
        }

    async def webrtc_ice_candidate(
        self,
        session_id: str,
        candidate: dict | None,
    ) -> None:
        """Send ICE candidate to cloud-hosted scope backend."""
        self._stats["webrtc_ice_candidates_sent"] += 1
        self._stats["last_activity_at"] = time.time()

        if candidate:
            logger.debug(
                f"[CLOUD] Sending ICE candidate to cloud for session {session_id}"
            )
        else:
            logger.info(
                f"[CLOUD] Sending end-of-candidates signal for session {session_id}"
            )

        message = {
            "type": "icecandidate",
            "sessionId": session_id,
            "candidate": candidate,
        }
        await self.send_and_wait(message, timeout=10.0)

        if self._stats["webrtc_ice_candidates_sent"] % 5 == 0:
            logger.info(
                f"[CLOUD] Sent {self._stats['webrtc_ice_candidates_sent']} ICE candidates total"
            )

    async def disconnect(self) -> None:
        """Disconnect from cloud.ai cloud."""
        self._stop_event.set()
        self._connected = False
        self._connection_id = None

        # Stop WebRTC client first
        await self.stop_webrtc()

        # Cancel pending requests
        for request_id, future in self._pending_requests.items():
            if not future.done():
                future.set_exception(RuntimeError("Disconnected from cloud"))
        self._pending_requests.clear()

        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        await self._cleanup()
        logger.info("Disconnected from cloud")

    # =========================================================================
    # WebRTC Media Relay - Video flows through backend to cloud.ai
    # =========================================================================

    @property
    def webrtc_connected(self) -> bool:
        """Check if WebRTC connection to cloud.ai is established."""
        return self._webrtc_client is not None and self._webrtc_client.is_connected

    async def start_webrtc(self, initial_parameters: dict | None = None) -> None:
        """Start WebRTC connection to cloud.ai for media relay.

        This establishes a WebRTC peer connection FROM the backend TO fal.ai,
        allowing video frames to flow through the backend.

        Args:
            initial_parameters: Initial pipeline parameters (prompts, etc.)
        """
        if not self.is_connected:
            raise RuntimeError("Must be connected to cloud.ai WebSocket first")

        if self._webrtc_client is not None:
            logger.info("[CLOUD-RTC] WebRTC already active, stopping first")
            await self.stop_webrtc()

        # Import here to avoid circular imports
        from .cloud_webrtc_client import CloudWebRTCClient

        logger.info("[CLOUD-RTC] Starting WebRTC connection to cloud.ai...")
        self._webrtc_client = CloudWebRTCClient(self)

        # Register frame callback to update stats and forward to subscribers
        self._webrtc_client.output_handler.add_callback(self._on_frame_from_fal)

        try:
            await self._webrtc_client.connect(initial_parameters)
            logger.info("[CLOUD-RTC] WebRTC connection established")
        except Exception as e:
            logger.error(f"[CLOUD-RTC] Failed to start WebRTC: {e}")
            self._webrtc_client = None
            raise

    async def stop_webrtc(self) -> None:
        """Stop the WebRTC connection to cloud.ai."""
        if self._webrtc_client is not None:
            logger.info("[CLOUD-RTC] Stopping WebRTC connection...")
            await self._webrtc_client.disconnect()
            self._webrtc_client = None
            logger.info("[CLOUD-RTC] WebRTC connection stopped")

    def send_frame_to_fal(self, frame: VideoFrame | np.ndarray) -> bool:
        """Send a video frame to cloud.ai for processing.

        Args:
            frame: VideoFrame or numpy array (RGB24 format)

        Returns:
            True if frame was queued, False if not connected or queue full
        """
        if self._webrtc_client is None or not self._webrtc_client.is_connected:
            return False

        success = self._webrtc_client.send_frame(frame)
        if success:
            self._stats["frames_sent_to_fal"] += 1
        return success

    def send_parameters_to_fal(self, params: dict) -> None:
        """Send parameter update to cloud.ai via WebRTC data channel.

        Args:
            params: Parameters to send (prompts, noise_scale, etc.)
        """
        if self._webrtc_client is not None and self._webrtc_client.is_connected:
            self._webrtc_client.send_parameters(params)
        else:
            logger.warning("[CLOUD-RTC] Cannot send parameters - WebRTC not connected")

    def add_frame_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        """Register a callback to receive processed frames from cloud.ai.

        Args:
            callback: Function to call with each processed VideoFrame
        """
        self._frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        """Remove a frame callback."""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)

    def _on_frame_from_fal(self, frame: VideoFrame) -> None:
        """Handle frames received from cloud.ai."""
        self._stats["frames_received_from_fal"] += 1

        # Forward to all registered callbacks
        for callback in self._frame_callbacks:
            try:
                callback(frame)
            except Exception as e:
                logger.error(f"[CLOUD-RTC] Error in frame callback: {e}")

    @property
    def fal_session_id(self) -> str | None:
        """Get the current fal.ai WebRTC session ID."""
        if self._webrtc_client is not None:
            return self._webrtc_client.session_id
        return None

    async def download_recording(self, session_id: str | None = None) -> bytes | None:
        """Download a recording from cloud.ai.

        Args:
            session_id: The fal.ai session ID. If None, uses the current session.

        Returns:
            The recording file bytes, or None if not available.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to cloud.ai")

        # Use provided session ID or fall back to current
        target_session_id = session_id or self.fal_session_id
        if not target_session_id:
            raise RuntimeError("No fal.ai session ID available")

        logger.info(f"[CLOUD] Downloading recording for session: {target_session_id}")

        # Request recording via WebSocket - fal_app will base64 encode the response
        response = await self.api_request(
            method="GET",
            path=f"/api/v1/recordings/{target_session_id}",
            timeout=120.0,  # Longer timeout for large files
        )

        # Check if we got binary data (base64 encoded)
        if response.get("_base64_content"):
            import base64

            content = base64.b64decode(response["_base64_content"])
            logger.info(f"[CLOUD] Downloaded recording: {len(content)} bytes")
            return content

        # Check for error
        if response.get("status") != 200:
            error = response.get("error", "Unknown error")
            logger.error(f"[CLOUD] Recording download failed: {error}")
            return None

        # Unexpected response format
        logger.warning(
            f"[CLOUD] Unexpected recording response format: {list(response.keys())}"
        )
        return None

    async def _cleanup(self) -> None:
        """Clean up WebSocket and session."""
        if self.ws:
            await self.ws.close()
            self.ws = None
        await self._cleanup_session()

    async def _cleanup_session(self) -> None:
        """Clean up aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    def get_status(self) -> dict:
        """Get current connection status."""
        status = {
            "connected": self.is_connected,
            "app_id": self.app_id if self.is_connected else None,
            "connection_id": self._connection_id if self.is_connected else None,
            "webrtc_connected": self.webrtc_connected,
        }

        # Include stats if connected
        if self.is_connected:
            uptime = None
            if self._stats["connected_at"]:
                uptime = time.time() - self._stats["connected_at"]

            status["stats"] = {
                "uptime_seconds": uptime,
                "webrtc_offers_sent": self._stats["webrtc_offers_sent"],
                "webrtc_offers_successful": self._stats["webrtc_offers_successful"],
                "webrtc_ice_candidates_sent": self._stats["webrtc_ice_candidates_sent"],
                "api_requests_sent": self._stats["api_requests_sent"],
                "api_requests_successful": self._stats["api_requests_successful"],
                "frames_sent_to_fal": self._stats["frames_sent_to_fal"],
                "frames_received_from_fal": self._stats["frames_received_from_fal"],
            }

            # Include WebRTC client stats if available
            if self._webrtc_client is not None:
                status["webrtc_stats"] = self._webrtc_client.get_stats()

        return status

    def print_stats(self) -> None:
        """Print current stats to logger."""
        if not self.is_connected:
            logger.info("[CLOUD] Not connected to cloud")
            return

        uptime = 0
        if self._stats["connected_at"]:
            uptime = time.time() - self._stats["connected_at"]

        logger.info("=" * 50)
        logger.info("[CLOUD] Cloud Connection Stats")
        logger.info("=" * 50)
        logger.info(f"  App ID: {self.app_id}")
        logger.info(f"  Uptime: {uptime:.1f}s")
        logger.info("  WebSocket: connected")
        logger.info(
            f"  WebRTC Media: {'connected' if self.webrtc_connected else 'not connected'}"
        )
        logger.info("")
        logger.info("  Signaling Stats:")
        logger.info(
            f"    WebRTC offers: {self._stats['webrtc_offers_successful']}/{self._stats['webrtc_offers_sent']}"
        )
        logger.info(
            f"    ICE candidates sent: {self._stats['webrtc_ice_candidates_sent']}"
        )
        logger.info(
            f"    API requests: {self._stats['api_requests_successful']}/{self._stats['api_requests_sent']}"
        )
        logger.info("")
        logger.info("  Media Stats:")
        logger.info(f"    Frames sent to cloud: {self._stats['frames_sent_to_fal']}")
        logger.info(
            f"    Frames received from cloud: {self._stats['frames_received_from_fal']}"
        )

        if self._webrtc_client is not None:
            rtc_stats = self._webrtc_client.get_stats()
            logger.info(
                f"    WebRTC state: {rtc_stats.get('connection_state', 'unknown')}"
            )

        logger.info("=" * 50)
