"""WebSocket + WebRTC client for connecting to fal.ai cloud.

Based on fal-demos/yolo_webcam_webrtc reference implementation.
Scope acts as WebRTC client (creates offers), fal.ai acts as server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import aiohttp
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp

if TYPE_CHECKING:
    from av import VideoFrame

    from scope.server.fal_tracks import FalOutputTrack

logger = logging.getLogger(__name__)

TOKEN_EXPIRATION_SECONDS = 120


class FalClient:
    """WebSocket + WebRTC client for connecting to fal.ai cloud.

    Based on fal-demos/yolo_webcam_webrtc reference implementation.
    Scope acts as WebRTC client (creates offers), fal.ai acts as server.
    """

    def __init__(
        self,
        app_id: str,
        api_key: str,
        on_frame_received: Callable[[VideoFrame], None] | None = None,
    ):
        self.app_id = app_id  # e.g., "owner/app-name/webrtc"
        self.api_key = api_key
        self.on_frame_received = on_frame_received

        self.ws: websockets.WebSocketClientProtocol | None = None
        self.pc: RTCPeerConnection | None = None
        self.output_track: FalOutputTrack | None = None
        self.stop_event = asyncio.Event()
        self._receive_task: asyncio.Task | None = None

    async def _get_temporary_token(self) -> str:
        """Get temporary JWT token from fal API (mirrors frontend pattern)."""
        # Extract alias from app_id (e.g., "owner/app-name/webrtc" -> "app-name")
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
                token = await resp.json()
                # Handle both string and object responses
                if isinstance(token, dict) and "detail" in token:
                    return token["detail"]
                return token

    def _build_ws_url(self, token: str) -> str:
        """Build WebSocket URL with JWT token (mirrors frontend pattern)."""
        app_id = self.app_id.strip("/")
        return f"wss://fal.run/{app_id}?fal_jwt_token={token}"

    async def connect(self) -> None:
        """Connect to fal WebSocket and establish WebRTC connection."""
        # Get temporary token
        token = await self._get_temporary_token()
        ws_url = self._build_ws_url(token)

        logger.info(f"Connecting to fal WebSocket: {ws_url[:50]}...")
        self.ws = await websockets.connect(ws_url)

        # Wait for "ready" message from server
        ready_msg = await self.ws.recv()
        ready_data = json.loads(ready_msg)
        if ready_data.get("type") != "ready":
            raise RuntimeError(f"Expected 'ready' message, got: {ready_data}")
        logger.info("fal server ready")

        # Create peer connection
        self.pc = RTCPeerConnection(
            configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
        )

        # Set up event handlers
        self._setup_pc_handlers()

        # Add output track (for sending frames to fal)
        from scope.server.fal_tracks import FalOutputTrack

        self.output_track = FalOutputTrack()
        self.pc.addTrack(self.output_track)

        # Create and send offer (we are the client)
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await self.ws.send(
            json.dumps(
                {
                    "type": "offer",
                    "sdp": self.pc.localDescription.sdp,
                }
            )
        )
        logger.info("Sent WebRTC offer")

        # Start message receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

    def _setup_pc_handlers(self) -> None:
        """Set up RTCPeerConnection event handlers."""
        if self.pc is None:
            return

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if self.ws is None:
                return
            if candidate is None:
                await self.ws.send(
                    json.dumps(
                        {
                            "type": "icecandidate",
                            "candidate": None,
                        }
                    )
                )
            else:
                await self.ws.send(
                    json.dumps(
                        {
                            "type": "icecandidate",
                            "candidate": {
                                "candidate": candidate.candidate,
                                "sdpMid": candidate.sdpMid,
                                "sdpMLineIndex": candidate.sdpMLineIndex,
                            },
                        }
                    )
                )

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if self.pc is None:
                return
            logger.info(f"Connection state: {self.pc.connectionState}")
            if self.pc.connectionState in ("failed", "closed", "disconnected"):
                self.stop_event.set()

        @self.pc.on("track")
        def on_track(track):
            """Handle incoming track (processed frames from fal)."""
            if track.kind == "video":
                logger.info("Received video track from fal")
                asyncio.create_task(self._consume_track(track))

    async def _consume_track(self, track) -> None:
        """Consume frames from the incoming track."""
        while not self.stop_event.is_set():
            try:
                frame = await track.recv()
                if self.on_frame_received:
                    self.on_frame_received(frame)
            except Exception as e:
                logger.error(f"Error receiving frame: {e}")
                break

    async def _receive_loop(self) -> None:
        """Receive and handle WebSocket messages."""
        if self.ws is None or self.pc is None:
            return

        try:
            while not self.stop_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        self.ws.recv(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message: {message}")
                    continue

                msg_type = data.get("type")

                if msg_type == "answer":
                    # Set remote description from server's answer
                    answer = RTCSessionDescription(
                        sdp=data["sdp"],
                        type="answer",
                    )
                    await self.pc.setRemoteDescription(answer)
                    logger.info("Set remote description from answer")

                elif msg_type == "icecandidate":
                    candidate_data = data.get("candidate")
                    if candidate_data is None:
                        await self.pc.addIceCandidate(None)
                    else:
                        candidate = candidate_from_sdp(
                            candidate_data.get("candidate", "")
                        )
                        candidate.sdpMid = candidate_data.get("sdpMid")
                        candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex")
                        await self.pc.addIceCandidate(candidate)

                elif msg_type == "error":
                    logger.error(f"Server error: {data.get('error')}")

                else:
                    logger.debug(f"Unknown message type: {msg_type}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
        finally:
            self.stop_event.set()

    async def send_frame(self, frame: VideoFrame) -> None:
        """Send a frame to fal for processing."""
        if self.output_track:
            await self.output_track.put_frame(frame)

    async def disconnect(self) -> None:
        """Close WebRTC and WebSocket connections."""
        self.stop_event.set()

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self.pc:
            await self.pc.close()
            self.pc = None

        if self.ws:
            await self.ws.close()
            self.ws = None

        logger.info("Disconnected from fal")
