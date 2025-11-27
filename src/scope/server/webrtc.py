import asyncio
import json
import logging
import os
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
from aiortc.contrib.media import MediaRelay
from aiortc.sdp import candidate_from_sdp

from .credentials import get_turn_credentials
from .pipeline_manager import PipelineManager
from .recording import RecordingManager
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


class Session:
    """WebRTC Session containing peer connection and associated video track."""

    def __init__(
        self,
        pc: RTCPeerConnection,
        video_track: MediaStreamTrack | None = None,
        data_channel: RTCDataChannel | None = None,
        relay: MediaRelay | None = None,
        recording_manager: RecordingManager | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.pc = pc
        self.video_track = video_track
        self.data_channel = data_channel
        self.relay = relay
        self.recording_manager = recording_manager

    async def close(self):
        """Close this session and cleanup resources."""
        try:
            # Stop video track first to properly cleanup FrameProcessor
            if self.video_track is not None:
                await self.video_track.stop()

            if self.pc.connectionState not in ["closed", "failed"]:
                await self.pc.close()

            logger.info(f"Session {self.id} closed")
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
        self.pending_notifications = []

        # Store reference to the event loop for thread-safe notifications
        self.event_loop = asyncio.get_running_loop()

    def set_data_channel(self, data_channel):
        """Set the data channel and flush any pending notifications."""
        self.data_channel = data_channel
        self.flush_pending_notifications()

    def call(self, message: dict):
        """Send a message to the frontend via data channel."""
        if self.data_channel and self.data_channel.readyState == "open":
            self._send_message_threadsafe(message)
        else:
            logger.info(f"Data channel not ready, queuing message: {message}")
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
                        self.data_channel.send(message_str)
                        logger.info(f"Sent notification to frontend: {message}")
                    except Exception as e:
                        logger.error(f"Failed to send notification: {e}")

                # Schedule the sync function to run in the main event loop
                self.event_loop.call_soon_threadsafe(send_sync)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def flush_pending_notifications(self):
        """Send all pending notifications when data channel becomes available"""
        if not self.pending_notifications:
            logger.info("No pending notifications to flush")
            return

        logger.info(f"Flushing {len(self.pending_notifications)} pending notifications")
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
        self.is_first_track = True

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
        try:
            # Extract initial parameters from offer
            initial_parameters = {}
            if request.initialParameters:
                # Convert Pydantic model to dict, excluding None values
                initial_parameters = request.initialParameters.model_dump(
                    exclude_none=True
                )
            logger.info(f"Received initial parameters: {initial_parameters}")

            # Create new RTCPeerConnection with configuration
            pc = RTCPeerConnection(self.rtc_config)
            session = Session(pc)
            self.sessions[session.id] = session

            # Create NotificationSender for this session to send notifications to the frontend
            notification_sender = NotificationSender()

            video_track = VideoProcessingTrack(
                pipeline_manager,
                initial_parameters=initial_parameters,
                notification_callback=notification_sender.call,
            )
            session.video_track = video_track

            # Create a MediaRelay to allow multiple consumers (WebRTC and recording)
            relay = MediaRelay()
            relayed_track = relay.subscribe(video_track)

            # Create RecordingManager and store it in the session
            # Pass the original video_track - RecordingManager will subscribe to relay itself
            recording_manager = RecordingManager(video_track=video_track)
            session.recording_manager = recording_manager

            # Set the relay on the recording manager so it can create a recording track
            recording_manager.set_relay(relay)

            # Add the relayed track to WebRTC connection
            pc.addTrack(relayed_track)

            # Store relay for cleanup
            session.relay = relay

            # Start recording when ready
            # We start it in a background task to avoid blocking the offer/answer flow
            async def start_recording_when_ready():
                """Start recording when frames start flowing."""
                try:
                    # Wait a bit for the connection to establish and frames to start flowing
                    await asyncio.sleep(0.1)
                    # Try to start recording
                    await recording_manager.start_recording()
                except Exception as e:
                    logger.debug(f"Could not start recording yet: {e}")

            asyncio.create_task(start_recording_when_ready())

            logger.info(f"Created new session: {session}")

            @pc.on("track")
            def on_track(track: MediaStreamTrack):
                logger.info(f"Track received: {track.kind} for session {session.id}")
                if track.kind == "video":
                    video_track.initialize_input_processing(track)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(
                    f"Connection state changed to: {pc.connectionState} for session {session.id}"
                )
                if pc.connectionState in ["closed", "failed"]:
                    await self.remove_session(session.id)

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                logger.info(
                    f"ICE connection state changed to: {pc.iceConnectionState} for session {session.id}"
                )

            @pc.on("icegatheringstatechange")
            async def on_icegatheringstatechange():
                logger.info(
                    f"ICE gathering state changed to: {pc.iceGatheringState} for session {session.id}"
                )

            @pc.on("icecandidate")
            def on_icecandidate(candidate):
                logger.debug(f"ICE candidate for session {session.id}: {candidate}")

            # Handle incoming data channel from frontend
            @pc.on("datachannel")
            def on_data_channel(data_channel):
                logger.info(
                    f"Data channel received: {data_channel.label} for session {session.id}"
                )
                session.data_channel = data_channel
                notification_sender.set_data_channel(data_channel)

                @data_channel.on("open")
                def on_data_channel_open():
                    logger.info(f"Data channel opened for session {session.id}")
                    notification_sender.flush_pending_notifications()

                @data_channel.on("message")
                def on_data_channel_message(message):
                    try:
                        # Parse the JSON message
                        data = json.loads(message)
                        logger.info(f"Received parameter update: {data}")

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
            offer_sdp = RTCSessionDescription(sdp=request.sdp, type=request.type)
            await pc.setRemoteDescription(offer_sdp)

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionId": session.id,
            }

        except Exception as e:
            logger.error(f"Error handling WebRTC offer: {e}")
            if "session" in locals():
                await self.remove_session(session.id)
            raise

    async def remove_session(self, session_id: str):
        """Remove and cleanup a specific session."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            logger.info(f"Removing session: {session}")

            # Delete recording file when session ends
            if session.recording_manager:
                await session.recording_manager.delete_recording()

            await session.close()
        else:
            logger.warning(f"Attempted to remove non-existent session: {session_id}")

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

    async def add_ice_candidate(
        self,
        session_id: str,
        candidate: str,
        sdp_mid: str | None,
        sdp_mline_index: int | None,
    ) -> None:
        """Add an ICE candidate to an existing session.

        Args:
            session_id: ID of the session
            candidate: ICE candidate string
            sdp_mid: Media stream ID
            sdp_mline_index: Media line index

        Raises:
            ValueError: If session not found or candidate invalid
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.pc.connectionState in ["closed", "failed"]:
            raise ValueError(f"Session {session_id} is closed or failed")

        # Parse candidate string and create RTCIceCandidate
        # aiortc expects the candidate object to be created from the SDP string

        try:
            ice_candidate = candidate_from_sdp(candidate)
            ice_candidate.sdpMid = sdp_mid
            ice_candidate.sdpMLineIndex = sdp_mline_index

            await session.pc.addIceCandidate(ice_candidate)
            logger.debug(f"Added ICE candidate to session {session_id}: {candidate}")
        except Exception as e:
            logger.error(f"Failed to add ICE candidate to session {session_id}: {e}")
            raise ValueError(f"Invalid ICE candidate: {e}") from e

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
