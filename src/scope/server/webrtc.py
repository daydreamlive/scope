import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.codecs import h264, vpx

from .credentials import get_turn_credentials
from .pipeline_manager import PipelineManager
from .schema import WebRTCOfferRequest
from .tracks import VideoProcessingTrack

logger = logging.getLogger(__name__)

# TODO: Fix bitrate
# Monkey patching these values in aiortc don't seem to work as expected
# The expected behavior is for the bitrate calculations to set a bitrate based on the ceiling, floor and defaults
# For now, these values were set kind of arbitrarily to increase the bitrate
h264.MAX_FRAME_RATE = 8
h264.DEFAULT_BITRATE = 7000000
h264.MIN_BITRATE = 5000000
h264.MAX_BITRATE = 10000000

vpx.MAX_FRAME_RATE = 8
vpx.DEFAULT_BITRATE = 7000000
vpx.MIN_BITRATE = 5000000
vpx.MAX_BITRATE = 10000000

# Session removal debounce (seconds) - short to allow quick refresh recovery
SESSION_REMOVAL_DEBOUNCE_SECONDS = 0.5


class Session:
    """WebRTC Session containing peer connection and associated video track."""

    def __init__(
        self,
        pc: RTCPeerConnection,
        video_track: MediaStreamTrack | None = None,
        data_channel: RTCDataChannel | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.pc = pc
        self.video_track = video_track
        self.data_channel = data_channel
        self._removal_task: asyncio.Task | None = None
        self._close_start_time: float | None = None

    async def close(self):
        """Close this session and cleanup resources."""
        self._close_start_time = time.time()
        try:
            # Cancel any pending removal task
            if self._removal_task and not self._removal_task.done():
                self._removal_task.cancel()

            # Stop video track first to properly cleanup FrameProcessor
            if self.video_track is not None:
                track_close_start = time.time()
                if hasattr(self.video_track, "aclose"):
                    await self.video_track.aclose()
                else:
                    self.video_track.stop()
                logger.debug(
                    f"[TIMING] Track close took {(time.time() - track_close_start) * 1000:.1f}ms"
                )

            if self.pc.connectionState not in ["closed", "failed"]:
                pc_close_start = time.time()
                await self.pc.close()
                logger.debug(
                    f"[TIMING] PC close took {(time.time() - pc_close_start) * 1000:.1f}ms"
                )

            total_time = (time.time() - self._close_start_time) * 1000
            logger.info(f"Session {self.id} closed in {total_time:.1f}ms")
        except Exception as e:
            logger.error(f"Error closing session {self.id}: {e}")

    def __str__(self):
        return f"Session({self.id}, state={self.pc.connectionState})"


class NotificationSender:
    """
    Handles sending notifications from backend to frontend using WebRTC data channels for a single session.
    """

    def __init__(self):
        self.data_channel = None
        self.pending_notifications: list[dict] = []

        # Store reference to the event loop for thread-safe notifications
        self.event_loop = asyncio.get_running_loop()

    def set_data_channel(self, data_channel: RTCDataChannel):
        """Set the data channel and flush any pending notifications."""
        self.data_channel = data_channel
        self.flush_pending_notifications()

    def call(self, message: dict):
        """Send a message to the frontend via data channel."""
        if self.data_channel and self.data_channel.readyState == "open":
            self._send_message_threadsafe(message)
        else:
            logger.debug(f"Data channel not ready, queuing message: {message}")
            self.pending_notifications.append(message)

    def _send_message_threadsafe(self, message: dict):
        """Send a message via data channel in a thread-safe manner"""
        try:
            message_str = json.dumps(message)
            # Use thread-safe method to send message
            if self.event_loop and self.event_loop.is_running():
                # Schedule the send operation in the main event loop
                def send_sync():
                    try:
                        if self.data_channel and self.data_channel.readyState == "open":
                            self.data_channel.send(message_str)
                            logger.debug(f"Sent notification to frontend: {message}")
                    except Exception as e:
                        logger.error(f"Failed to send notification: {e}")

                # Schedule the sync function to run in the main event loop
                self.event_loop.call_soon_threadsafe(send_sync)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def flush_pending_notifications(self):
        """Send all pending notifications when data channel becomes available"""
        if not self.pending_notifications:
            return

        logger.debug(
            f"Flushing {len(self.pending_notifications)} pending notifications"
        )
        for message in self.pending_notifications:
            self._send_message_threadsafe(message)
        self.pending_notifications.clear()


class WebRTCManager:
    """
    Manages multiple WebRTC peer connections using sessions.
    """

    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.rtc_config = create_rtc_config()

    async def handle_offer(
        self, request: WebRTCOfferRequest, pipeline_manager: PipelineManager
    ) -> dict[str, Any]:
        """
        Handle an incoming WebRTC offer and return an answer.

        Args:
            offer_data: Dictionary containing SDP offer
            pipeline_manager: The pipeline manager instance

        Returns:
            Dictionary containing SDP answer
        """
        offer_start_time = time.time()

        try:
            # Extract initial parameters from offer
            initial_parameters = {}
            if request.initialParameters:
                # Convert Pydantic model to dict, excluding None values
                initial_parameters = request.initialParameters.model_dump(
                    exclude_none=True
                )
            logger.info(
                f"[TIMING] Received offer, initial params: {list(initial_parameters.keys())}"
            )

            # Close any existing sessions to free resources quickly
            # This helps with tab refresh scenarios
            if self.sessions:
                logger.info(
                    f"[TIMING] Closing {len(self.sessions)} existing sessions before new offer"
                )
                close_start = time.time()
                # Close sessions in background - don't wait
                for session in list(self.sessions.values()):
                    asyncio.create_task(self._close_session_background(session))
                self.sessions.clear()
                logger.debug(
                    f"[TIMING] Session cleanup initiated in {(time.time() - close_start) * 1000:.1f}ms"
                )

            # Create new RTCPeerConnection with configuration
            pc_create_start = time.time()
            pc = RTCPeerConnection(self.rtc_config)
            logger.debug(
                f"[TIMING] PC created in {(time.time() - pc_create_start) * 1000:.1f}ms"
            )

            notification_sender = NotificationSender()
            session = Session(pc)
            self.sessions[session.id] = session

            video_track = VideoProcessingTrack(
                pipeline_manager,
                initial_parameters=initial_parameters,
                notification_callback=notification_sender.call,
            )
            session.video_track = video_track
            pc.addTrack(video_track)

            logger.info(f"Created new session: {session}")

            @pc.on("track")
            def on_track(track: MediaStreamTrack):
                track_recv_time = time.time()
                logger.info(
                    f"[TIMING] Track received: {track.kind} for session {session.id} "
                    f"({(track_recv_time - offer_start_time) * 1000:.1f}ms since offer)"
                )
                if track.kind == "video":
                    video_track.initialize_input_processing(track)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                state = pc.connectionState
                logger.info(
                    f"Connection state changed to: {state} for session {session.id}"
                )
                # Only remove on closed/failed, with short debounce for quick reconnects
                if state in ["closed", "failed"]:
                    await self._debounced_remove(
                        session, delay_seconds=SESSION_REMOVAL_DEBOUNCE_SECONDS
                    )

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                ice_state = pc.iceConnectionState
                logger.info(
                    f"ICE connection state: {ice_state} for session {session.id}"
                )
                # Log when ICE is connected - this is when media can flow
                if ice_state == "connected":
                    logger.info(
                        f"[TIMING] ICE connected ({(time.time() - offer_start_time) * 1000:.1f}ms since offer)"
                    )

            @pc.on("icegatheringstatechange")
            async def on_icegatheringstatechange():
                logger.debug(
                    f"ICE gathering state: {pc.iceGatheringState} for session {session.id}"
                )

            @pc.on("icecandidate")
            def on_icecandidate(candidate):
                if candidate:
                    logger.debug(f"ICE candidate for session {session.id}")

            # Handle incoming data channel from frontend
            @pc.on("datachannel")
            def on_data_channel(data_channel: RTCDataChannel):
                dc_time = time.time()
                logger.info(
                    f"[TIMING] Data channel received: {data_channel.label} "
                    f"({(dc_time - offer_start_time) * 1000:.1f}ms since offer)"
                )
                session.data_channel = data_channel
                notification_sender.set_data_channel(data_channel)

                @data_channel.on("open")
                def on_data_channel_open():
                    logger.info(
                        f"[TIMING] Data channel opened ({(time.time() - offer_start_time) * 1000:.1f}ms since offer)"
                    )
                    notification_sender.flush_pending_notifications()

                @data_channel.on("message")
                def on_data_channel_message(message):
                    try:
                        # Parse the JSON message
                        data = json.loads(message)
                        # Only log non-ping messages at info level
                        if data.get("type") != "ping":
                            logger.info(f"Received parameter update: {data}")
                        else:
                            logger.debug("Received ping from frontend")

                        # Respond to pings with pong (frontend keepalive)
                        if data.get("type") == "ping":
                            notification_sender.call(
                                {"type": "pong", "ts": time.time()}
                            )
                            return

                        # Check for paused parameter and call pause() method on video track
                        if "paused" in data and session.video_track:
                            session.video_track.pause(data["paused"])

                        # Send parameters to the frame processor
                        if session.video_track and hasattr(
                            session.video_track, "frame_processor"
                        ):
                            session.video_track.frame_processor.update_parameters(data)
                        else:
                            logger.warning(
                                "No frame processor available for parameter update"
                            )

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse parameter update message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling parameter update: {e}")

            # Set remote description (the offer)
            set_remote_start = time.time()
            offer_sdp = RTCSessionDescription(sdp=request.sdp, type=request.type)
            await pc.setRemoteDescription(offer_sdp)
            logger.debug(
                f"[TIMING] setRemoteDescription took {(time.time() - set_remote_start) * 1000:.1f}ms"
            )

            # Create answer
            create_answer_start = time.time()
            answer = await pc.createAnswer()
            logger.debug(
                f"[TIMING] createAnswer took {(time.time() - create_answer_start) * 1000:.1f}ms"
            )

            set_local_start = time.time()
            await pc.setLocalDescription(answer)
            logger.debug(
                f"[TIMING] setLocalDescription took {(time.time() - set_local_start) * 1000:.1f}ms"
            )

            total_time = (time.time() - offer_start_time) * 1000
            logger.info(f"[TIMING] Offer handling complete in {total_time:.1f}ms")

            return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

        except Exception as e:
            logger.error(f"Error handling WebRTC offer: {e}")
            if "session" in locals():
                await self.remove_session(session.id)
            raise

    async def _close_session_background(self, session: Session):
        """Close a session in background without blocking."""
        try:
            await session.close()
        except Exception as e:
            logger.error(f"Error in background session close: {e}")

    async def remove_session(self, session_id: str):
        """Remove and cleanup a specific session."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            logger.info(f"Removing session: {session}")
            await session.close()
        else:
            logger.debug(f"Attempted to remove non-existent session: {session_id}")

    async def _debounced_remove(self, session: Session, delay_seconds: float = 0.5):
        """
        Debounce session removal to allow brief disconnects (e.g., tab refresh).

        If the connection recovers before the delay, the removal task is cancelled.
        """
        # Cancel any pending removal for this session
        if session._removal_task and not session._removal_task.done():
            session._removal_task.cancel()

        async def _remove_if_still_closed():
            try:
                await asyncio.sleep(delay_seconds)
                # If session disappeared during wait, nothing to do
                current = self.sessions.get(session.id)
                if current is None:
                    return
                # Remove only if still closed/failed
                if current.pc.connectionState in ["closed", "failed"]:
                    await self.remove_session(session.id)
            except asyncio.CancelledError:
                return

        session._removal_task = asyncio.create_task(_remove_if_still_closed())

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def list_sessions(self) -> dict[str, Session]:
        """Get all current sessions."""
        return self.sessions.copy()

    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(
            [
                s
                for s in self.sessions.values()
                if s.pc.connectionState not in ["closed", "failed"]
            ]
        )

    async def stop(self):
        """Close and cleanup all sessions."""
        # Close all sessions in parallel
        close_tasks = [session.close() for session in self.sessions.values()]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Clear the sessions dict
        self.sessions.clear()


def create_rtc_config() -> RTCConfiguration:
    """Setup RTCConfiguration with TURN credentials if available."""
    try:
        hf_token = os.getenv("HF_TOKEN")
        twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")

        turn_provider = None
        if hf_token:
            turn_provider = "cloudflare"
        elif twilio_account_sid and twilio_auth_token:
            turn_provider = "twilio"

        if turn_provider:
            turn_credentials = get_turn_credentials(method=turn_provider)

            ice_servers = credentials_to_rtc_ice_servers(turn_credentials)
            logger.info(
                f"RTCConfiguration created with {turn_provider} and {len(ice_servers)} ICE servers"
            )
            return RTCConfiguration(iceServers=ice_servers)
        else:
            logger.info(
                "No Twilio or HF_TOKEN credentials found, using default STUN server"
            )
            stun_server = RTCIceServer(urls=["stun:stun.l.google.com:19302"])
            return RTCConfiguration(iceServers=[stun_server])
    except Exception as e:
        logger.warning(f"Failed to get TURN credentials, using default STUN: {e}")
        stun_server = RTCIceServer(urls=["stun:stun.l.google.com:19302"])
        return RTCConfiguration(iceServers=[stun_server])


def credentials_to_rtc_ice_servers(credentials: dict[str, Any]) -> list[RTCIceServer]:
    ice_servers = []
    if "iceServers" in credentials:
        for server in credentials["iceServers"]:
            urls = server.get("urls", [])
            username = server.get("username")
            credential = server.get("credential")

            ice_server = RTCIceServer(
                urls=urls, username=username, credential=credential
            )
            ice_servers.append(ice_server)
    return ice_servers
