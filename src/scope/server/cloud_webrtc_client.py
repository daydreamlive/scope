"""CloudWebRTCClient - WebRTC client that connects to cloud as a peer.

This module creates a WebRTC connection FROM the local backend TO cloud,
allowing video frames to flow through the backend:

    Browser/Spout → Local Backend → Cloud → Local Backend → Browser/Spout

This enables:
1. Spout input to be forwarded to cloud for processing
2. Full control over the video pipeline on the backend
3. Ability to record/manipulate frames before/after cloud processing
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import VIDEO_TIME_BASE, MediaStreamTrack
from av import VideoFrame

if TYPE_CHECKING:
    from .cloud_connection import CloudConnectionManager

logger = logging.getLogger(__name__)


class FrameInputTrack(MediaStreamTrack):
    """A MediaStreamTrack that receives frames from a queue/callback.

    This track is used to send frames TO cloud. Frames can come from:
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
        """Get the next frame to send to cloud."""
        if self._start_time is None:
            self._start_time = time.time()

        # Wait for a frame with timeout
        try:
            frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except TimeoutError:
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
        """Add a frame to be sent to cloud.

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
    """Handles frames received FROM cloud.

    Processed frames from cloud are passed to registered callbacks,
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
        """Called when a frame is received from cloud."""
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
    """WebRTC client that connects to cloud for remote processing.

    This establishes a WebRTC peer connection to the cloud runner,
    allowing video frames to be sent for processing and received back.

    Usage:
        client = CloudWebRTCClient(cloud_connection_manager)
        await client.connect()

        # Send frames to cloud
        client.input_track.put_frame(frame)

        # Receive processed frames
        client.output_handler.add_callback(my_callback)
    """

    def __init__(self, cloud_manager: CloudConnectionManager):
        self.cloud_manager = cloud_manager
        self.pc: RTCPeerConnection | None = None
        self.input_track: FrameInputTrack | None = None
        self.output_handler = FrameOutputHandler()
        self._data_channel = None
        self._session_id: str | None = None
        self._connected = False
        self._receive_task: asyncio.Task | None = None
        # Flag to track intentional disconnects (user-initiated via stop_webrtc)
        # When True, connection loss won't trigger error/reconnect
        self._intentional_disconnect = False

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
        """Establish WebRTC connection to cloud.

        Args:
            initial_parameters: Initial pipeline parameters to send with the offer
        """
        if not self.cloud_manager.is_connected:
            raise RuntimeError("CloudConnectionManager not connected to cloud")

        if self.is_connected:
            logger.info("Already connected, disconnecting first")
            await self.disconnect()

        logger.info("Creating WebRTC connection to cloud...")

        # Reset intentional disconnect flag for new connection
        self._intentional_disconnect = False

        # Get ICE servers from cloud
        ice_response = await self.cloud_manager.webrtc_get_ice_servers()

        from .webrtc import credentials_to_rtc_ice_servers

        rtc_ice_servers = credentials_to_rtc_ice_servers(ice_response)
        config = (
            RTCConfiguration(iceServers=rtc_ice_servers)
            if rtc_ice_servers
            else RTCConfiguration()
        )
        self.pc = RTCPeerConnection(config)

        # Create input track for sending frames to cloud
        self.input_track = FrameInputTrack(fps=30)
        self.pc.addTrack(self.input_track)

        # Create data channel for parameter updates
        self._data_channel = self.pc.createDataChannel("parameters", ordered=True)

        @self._data_channel.on("open")
        def on_dc_open():
            logger.info("Data channel opened")

        @self._data_channel.on("message")
        def on_dc_message(message):
            logger.debug(f"Data channel message: {message}")

        # Handle incoming track (processed frames from cloud)
        @self.pc.on("track")
        async def on_track(track: MediaStreamTrack):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "video":
                self._receive_task = asyncio.create_task(self._receive_frames(track))
                # Request keyframe immediately to avoid VP8 decode errors
                # PLI (Picture Loss Indication) tells remote to send an I-frame
                asyncio.create_task(self._request_keyframe())

        # Monitor connection state
        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            state = self.pc.connectionState
            logger.info(f"Connection state: {state}")
            self._stats["connection_state"] = state

            if state == "connected":
                self._connected = True
                self._stats["connected_at"] = time.time()
                logger.info("WebRTC connected to cloud")
            elif state in ("disconnected", "failed", "closed"):
                was_connected = self._connected
                self._connected = False
                # Notify cloud manager to surface error and attempt reconnection
                # Only notify if:
                # 1. We were previously connected (not on initial failure)
                # 2. This wasn't an intentional disconnect (user-initiated)
                if was_connected and not self._intentional_disconnect:
                    self.cloud_manager.on_webrtc_connection_lost(
                        reason=f"WebRTC connection state: {state}"
                    )

        @self.pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                logger.debug(f"Local ICE candidate: {candidate.candidate}")
                # Send to cloud via WebSocket
                if self._session_id:
                    try:
                        await self.cloud_manager.webrtc_ice_candidate(
                            self._session_id,
                            {
                                "candidate": candidate.candidate,
                                "sdpMid": candidate.sdpMid,
                                "sdpMLineIndex": candidate.sdpMLineIndex,
                            },
                        )
                    except Exception as e:
                        logger.error(f"Failed to send ICE candidate: {e}")

        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        logger.info("Sending offer to cloud...")

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

        logger.info(f"Received answer, session: {self._session_id}")

        # Set remote description
        answer = RTCSessionDescription(sdp=answer_sdp, type=answer_type)
        await self.pc.setRemoteDescription(answer)

        # Wait for connection with timeout
        timeout = 30.0
        start = time.time()
        while not self._connected and time.time() - start < timeout:
            await asyncio.sleep(0.1)

        if not self._connected:
            raise RuntimeError(f"WebRTC connection to cloud timed out after {timeout}s")

        logger.info("Connection established successfully")

    async def _receive_frames(self, track: MediaStreamTrack):
        """Background task to receive frames from cloud."""
        from aiortc.mediastreams import MediaStreamError

        logger.info("Starting frame receive loop")
        consecutive_errors = 0
        max_consecutive_errors = 10

        try:
            while True:
                try:
                    frame = await track.recv()
                    consecutive_errors = 0
                    self._stats["frames_received"] += 1

                    if self._stats["frames_received"] == 1:
                        logger.info(
                            "First frame received from cloud "
                            "(keyframe decoded successfully)"
                        )
                    elif self._stats["frames_received"] % 100 == 0:
                        logger.debug(
                            f"Received {self._stats['frames_received']} frames"
                        )

                    # Pass to output handler
                    self.output_handler.handle_frame(frame)

                except MediaStreamError:
                    logger.info("Track ended")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"Error receiving frame, stopping after "
                            f"{consecutive_errors} consecutive errors: {e}"
                        )
                        break
                    logger.warning(
                        f"Transient error receiving frame "
                        f"({consecutive_errors}/{max_consecutive_errors}): {e}"
                    )
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info("Frame receive loop cancelled")
        finally:
            logger.info(
                f"Frame receive loop ended, "
                f"total frames: {self._stats['frames_received']}"
            )

    async def _request_keyframe(self):
        """Request a keyframe via PLI once the receiver has remote streams.

        VP8/VP9 decoders need a keyframe (I-frame) to start decoding.
        After a new WebRTC connection, we may receive P-frames first,
        causing decode errors. Sending PLI (Picture Loss Indication)
        requests the remote end to send a keyframe.
        """
        # Poll until remote_streams is populated (RTP packets have arrived)
        timeout = 5.0
        poll_interval = 0.1
        elapsed = 0.0
        receiver = None
        remote_streams = None

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            for r in self.pc.getReceivers():
                if r.track and r.track.kind == "video":
                    streams = r._RTCRtpReceiver__remote_streams
                    if streams:
                        receiver = r
                        remote_streams = streams
                        break
            if remote_streams:
                break

        if not remote_streams or not receiver:
            logger.debug("No remote streams after %.1fs, skipping PLI", timeout)
            return

        try:
            media_ssrc = next(iter(remote_streams))
            await receiver._send_rtcp_pli(media_ssrc)
            logger.info(f"Sent PLI keyframe request (media_ssrc={media_ssrc})")
        except Exception as e:
            logger.debug(f"Could not send PLI: {e}")

    def send_frame(self, frame: VideoFrame | np.ndarray) -> bool:
        """Send a frame to cloud for processing.

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
        """Send parameter update to cloud via data channel."""
        if self._data_channel and self._data_channel.readyState == "open":
            import json

            self._data_channel.send(json.dumps(params))
            logger.debug(f"Sent parameters: {params}")
        else:
            logger.warning("Data channel not ready for parameters")

    async def disconnect(self):
        """Close the WebRTC connection to cloud."""
        logger.info("Disconnecting from cloud...")

        # Mark as intentional disconnect to prevent error/reconnect triggers
        self._intentional_disconnect = True
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

        logger.info("Disconnected")

    def get_stats(self) -> dict:
        """Get connection statistics."""
        stats = dict(self._stats)
        if stats["connected_at"]:
            stats["uptime_seconds"] = time.time() - stats["connected_at"]
        return stats
