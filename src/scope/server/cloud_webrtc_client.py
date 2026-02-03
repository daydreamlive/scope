"""CloudWebRTCClient - WebRTC client that connects to cloud.ai as a peer.

This module creates a WebRTC connection FROM the local backend TO fal.ai,
allowing video frames to flow through the backend:

    Browser/Spout → Local Backend → fal.ai → Local Backend → Browser/Spout

This enables:
1. Spout input to be forwarded to cloud.ai for processing
2. Full control over the video pipeline on the backend
3. Ability to record/manipulate frames before/after fal processing
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import time
import uuid
from typing import TYPE_CHECKING, Callable

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamTrack, VIDEO_TIME_BASE
from av import VideoFrame

if TYPE_CHECKING:
    from .cloud_connection import CloudConnectionManager

logger = logging.getLogger(__name__)


class FrameInputTrack(MediaStreamTrack):
    """A MediaStreamTrack that receives frames from a queue/callback.

    This track is used to send frames TO fal.ai. Frames can come from:
    - Browser WebRTC connection (relayed through backend)
    - Spout receiver
    - Any other frame source
    """

    kind = "video"

    def __init__(self, fps: int = 30):
        super().__init__()
        self._queue: asyncio.Queue[VideoFrame | None] = asyncio.Queue(maxsize=2)
        self._fps = fps
        self._frame_count = 0
        self._start_time: float | None = None
        self._last_pts = 0

    async def recv(self) -> VideoFrame:
        """Get the next frame to send to cloud.ai."""
        if self._start_time is None:
            self._start_time = time.time()

        # Wait for a frame with timeout
        try:
            frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Return a black frame if no input
            frame = self._create_black_frame()

        if frame is None:
            # End of stream signal
            raise StopAsyncIteration

        # Set proper timestamps
        self._frame_count += 1
        pts = int((time.time() - self._start_time) * 90000)  # 90kHz clock
        frame.pts = pts
        frame.time_base = VIDEO_TIME_BASE  # fractions.Fraction(1, 90000)

        return frame

    def put_frame(self, frame: VideoFrame | np.ndarray) -> bool:
        """Add a frame to be sent to cloud.ai.

        Args:
            frame: VideoFrame or numpy array (RGB24 format)

        Returns:
            True if frame was queued, False if queue is full
        """
        if isinstance(frame, np.ndarray):
            frame = VideoFrame.from_ndarray(frame, format="rgb24")

        try:
            self._queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            return False

    def _create_black_frame(self) -> VideoFrame:
        """Create a black frame for when no input is available."""
        black = np.zeros((512, 512, 3), dtype=np.uint8)
        return VideoFrame.from_ndarray(black, format="rgb24")


class FrameOutputHandler:
    """Handles frames received FROM fal.ai.

    Processed frames from cloud.ai are passed to registered callbacks,
    which can send them to:
    - Browser WebRTC connection
    - Spout sender
    - Recording/storage
    """

    def __init__(self):
        self._callbacks: list[Callable[[VideoFrame], None]] = []
        self._frame_count = 0
        self._last_frame: VideoFrame | None = None

    def add_callback(self, callback: Callable[[VideoFrame], None]):
        """Register a callback to receive processed frames."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[VideoFrame], None]):
        """Remove a frame callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def handle_frame(self, frame: VideoFrame):
        """Called when a frame is received from cloud.ai."""
        self._frame_count += 1
        self._last_frame = frame

        for callback in self._callbacks:
            try:
                callback(frame)
            except Exception as e:
                logger.error(f"Error in frame callback: {e}")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def last_frame(self) -> VideoFrame | None:
        return self._last_frame


class CloudWebRTCClient:
    """WebRTC client that connects to cloud.ai for remote processing.

    This establishes a WebRTC peer connection to the fal.ai runner,
    allowing video frames to be sent for processing and received back.

    Usage:
        client = CloudWebRTCClient(cloud_connection_manager)
        await client.connect()

        # Send frames to cloud.ai
        client.input_track.put_frame(frame)

        # Receive processed frames
        client.output_handler.add_callback(my_callback)
    """

    def __init__(self, cloud_manager: "CloudConnectionManager"):
        self.cloud_manager = cloud_manager
        self.pc: RTCPeerConnection | None = None
        self.input_track: FrameInputTrack | None = None
        self.output_handler = FrameOutputHandler()
        self._data_channel = None
        self._session_id: str | None = None
        self._connected = False
        self._receive_task: asyncio.Task | None = None

        # Stats
        self._stats = {
            "frames_sent": 0,
            "frames_received": 0,
            "connected_at": None,
            "connection_state": "new",
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and self.pc is not None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def connect(self, initial_parameters: dict | None = None) -> None:
        """Establish WebRTC connection to cloud.ai.

        Args:
            initial_parameters: Initial pipeline parameters to send with the offer
        """
        if not self.cloud_manager.is_connected:
            raise RuntimeError("CloudConnectionManager not connected to cloud.ai")

        if self.is_connected:
            logger.info("Already connected, disconnecting first")
            await self.disconnect()

        logger.info("[FAL-RTC] Creating WebRTC connection to cloud.ai...")

        # Get ICE servers from cloud
        ice_response = await self.cloud_manager.webrtc_get_ice_servers()
        ice_servers = ice_response.get("data", {}).get("iceServers", [])

        # Create peer connection
        config = {"iceServers": ice_servers} if ice_servers else {}
        self.pc = RTCPeerConnection(config)

        # Create input track for sending frames to cloud
        self.input_track = FrameInputTrack(fps=30)
        self.pc.addTrack(self.input_track)

        # Create data channel for parameter updates
        self._data_channel = self.pc.createDataChannel("parameters", ordered=True)

        @self._data_channel.on("open")
        def on_dc_open():
            logger.info("[FAL-RTC] Data channel opened")

        @self._data_channel.on("message")
        def on_dc_message(message):
            logger.debug(f"[FAL-RTC] Data channel message: {message}")

        # Handle incoming track (processed frames from cloud)
        @self.pc.on("track")
        async def on_track(track: MediaStreamTrack):
            logger.info(f"[FAL-RTC] Received track: {track.kind}")
            if track.kind == "video":
                self._receive_task = asyncio.create_task(
                    self._receive_frames(track)
                )
                # Request keyframe immediately to avoid VP8 decode errors
                # PLI (Picture Loss Indication) tells remote to send an I-frame
                asyncio.create_task(self._request_keyframe())

        # Monitor connection state
        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            state = self.pc.connectionState
            logger.info(f"[FAL-RTC] Connection state: {state}")
            self._stats["connection_state"] = state

            if state == "connected":
                self._connected = True
                self._stats["connected_at"] = time.time()
                logger.info("[FAL-RTC] WebRTC connected to cloud.ai")
            elif state in ("disconnected", "failed", "closed"):
                self._connected = False

        @self.pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                logger.debug(f"[FAL-RTC] Local ICE candidate: {candidate.candidate}")
                # Send to cloud via WebSocket
                if self._session_id:
                    try:
                        await self.cloud_manager.webrtc_ice_candidate(
                            self._session_id,
                            {
                                "candidate": candidate.candidate,
                                "sdpMid": candidate.sdpMid,
                                "sdpMLineIndex": candidate.sdpMLineIndex,
                            }
                        )
                    except Exception as e:
                        logger.error(f"[FAL-RTC] Failed to send ICE candidate: {e}")

        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        logger.info("[FAL-RTC] Sending offer to cloud.ai...")

        # Send offer through WebSocket
        response = await self.cloud_manager.webrtc_offer(
            sdp=self.pc.localDescription.sdp,
            sdp_type=self.pc.localDescription.type,
            initial_parameters=initial_parameters,
        )

        if "error" in response:
            raise RuntimeError(f"Offer failed: {response.get('error')}")

        self._session_id = response.get("sessionId")
        answer_sdp = response.get("sdp")
        answer_type = response.get("sdp_type", "answer")

        logger.info(f"[FAL-RTC] Received answer, session: {self._session_id}")

        # Set remote description
        answer = RTCSessionDescription(sdp=answer_sdp, type=answer_type)
        await self.pc.setRemoteDescription(answer)

        # Wait for connection with timeout
        timeout = 30.0
        start = time.time()
        while not self._connected and time.time() - start < timeout:
            await asyncio.sleep(0.1)

        if not self._connected:
            raise RuntimeError(f"WebRTC connection to cloud.ai timed out after {timeout}s")

        logger.info("[FAL-RTC] Connection established successfully")

    async def _receive_frames(self, track: MediaStreamTrack):
        """Background task to receive frames from cloud.ai."""
        logger.info("[FAL-RTC] Starting frame receive loop")

        try:
            while True:
                try:
                    frame = await track.recv()
                    self._stats["frames_received"] += 1

                    if self._stats["frames_received"] % 100 == 0:
                        logger.debug(
                            f"[FAL-RTC] Received {self._stats['frames_received']} frames"
                        )

                    # Pass to output handler
                    self.output_handler.handle_frame(frame)

                except Exception as e:
                    if "MediaStreamError" in str(type(e)):
                        logger.info("[FAL-RTC] Track ended")
                        break
                    logger.error(f"[FAL-RTC] Error receiving frame: {e}")
                    break

        except asyncio.CancelledError:
            logger.info("[FAL-RTC] Frame receive loop cancelled")
        finally:
            logger.info(
                f"[FAL-RTC] Frame receive loop ended, "
                f"total frames: {self._stats['frames_received']}"
            )

    async def _request_keyframe(self):
        """Request a keyframe via PLI after short delay for receiver setup.

        VP8/VP9 decoders need a keyframe (I-frame) to start decoding.
        After a new WebRTC connection, we may receive P-frames first,
        causing decode errors. Sending PLI (Picture Loss Indication)
        requests the remote end to send a keyframe.
        """
        await asyncio.sleep(0.1)  # Allow receiver to initialize
        for receiver in self.pc.getReceivers():
            if receiver.track and receiver.track.kind == "video":
                try:
                    # Access internal PLI method from aiortc
                    await receiver._send_rtcp_pli()
                    logger.info("[FAL-RTC] Sent PLI (keyframe request)")
                except Exception as e:
                    logger.debug(f"[FAL-RTC] Could not send PLI: {e}")

    def send_frame(self, frame: VideoFrame | np.ndarray) -> bool:
        """Send a frame to cloud.ai for processing.

        Args:
            frame: VideoFrame or numpy array (RGB24)

        Returns:
            True if frame was queued, False if queue is full
        """
        if not self.is_connected or self.input_track is None:
            return False

        success = self.input_track.put_frame(frame)
        if success:
            self._stats["frames_sent"] += 1
        return success

    def send_parameters(self, params: dict):
        """Send parameter update to cloud.ai via data channel."""
        if self._data_channel and self._data_channel.readyState == "open":
            import json
            self._data_channel.send(json.dumps(params))
            logger.debug(f"[FAL-RTC] Sent parameters: {params}")
        else:
            logger.warning("[FAL-RTC] Data channel not ready for parameters")

    async def disconnect(self):
        """Close the WebRTC connection to cloud.ai."""
        logger.info("[FAL-RTC] Disconnecting from cloud.ai...")

        self._connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self.pc:
            await self.pc.close()
            self.pc = None

        self.input_track = None
        self._data_channel = None
        self._session_id = None

        logger.info("[FAL-RTC] Disconnected")

    def get_stats(self) -> dict:
        """Get connection statistics."""
        stats = dict(self._stats)
        if stats["connected_at"]:
            stats["uptime_seconds"] = time.time() - stats["connected_at"]
        return stats
