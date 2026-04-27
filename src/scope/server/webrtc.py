import asyncio
import json
import logging
import os
import uuid

# Type checking imports
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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

from scope.core.nodes.registry import NodeRegistry

from .audio_track import AudioProcessingTrack
from .cloud_track import CloudTrack
from .credentials import get_turn_credentials
from .frame_processor import FrameProcessor
from .headless import HeadlessSession
from .kafka_publisher import publish_event
from .livepeer import LivepeerConnection
from .pipeline_manager import PipelineManager
from .recording import RecordingManager
from .schema import WebRTCOfferRequest
from .tracks import (
    SinkOutputTrack,
    SourceInputHandler,
    VideoProcessingTrack,
)

if TYPE_CHECKING:
    from .scope_cloud_types import ScopeCloudBackend
    from .tracks import NodeOutputTrack

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


def _build_extra_output_tracks(
    frame_processor: FrameProcessor,
    sink_node_ids: list[str],
    record_node_ids: list[str],
) -> list["NodeOutputTrack"]:
    """Build live WebRTC output tracks for graph outputs.

    Record-node queues are single-consumer recording queues. Attaching them as
    live WebRTC tracks races with RecordingCoordinator and can drain/corrupt
    recordings, so only sink nodes get live output tracks here.
    """
    if record_node_ids:
        logger.debug(
            "Skipping live WebRTC tracks for record nodes: %s",
            record_node_ids,
        )
    return [
        SinkOutputTrack(frame_processor=frame_processor, sink_node_id=sink_id)
        for sink_id in sink_node_ids[1:]
    ]


def _parse_graph_node_ids(
    initial_parameters: dict,
) -> tuple[list[str], list[str], list[str], list[str], list[str], bool]:
    """Extract sink, source, and record node IDs from graph initial parameters.

    Returns:
        (webrtc_sink_node_ids, webrtc_source_node_ids,
         all_sink_node_ids, all_source_node_ids,
         record_node_ids, has_non_webrtc_sources)

    ``webrtc_sink_node_ids`` excludes sink nodes that have a non-WebRTC
    ``sink_mode`` (spout/ndi/syphon) since those are handled by per-node
    output sinks in FrameProcessor.
    ``all_sink_node_ids`` includes every sink node regardless of mode.
    ``webrtc_source_node_ids`` excludes hardware sources (spout/ndi/syphon).
    ``all_source_node_ids`` includes every source node regardless of mode;
    used by the cloud's ``handle_offer`` where hardware-source frames arrive
    via WebRTC tracks relayed by the local instance.
    ``record_node_ids`` contains IDs of record-type nodes that need
    dedicated output tracks in cloud mode.
    """
    webrtc_sink_node_ids: list[str] = []
    all_sink_node_ids: list[str] = []
    webrtc_source_node_ids: list[str] = []
    all_source_node_ids: list[str] = []
    record_node_ids: list[str] = []
    has_non_webrtc_sources = False
    graph_data = initial_parameters.get("graph")
    if graph_data and isinstance(graph_data, dict):
        for node in graph_data.get("nodes", []):
            if node.get("type") == "sink":
                all_sink_node_ids.append(node["id"])
                sm = node.get("sink_mode")
                if sm not in ("spout", "ndi", "syphon"):
                    webrtc_sink_node_ids.append(node["id"])
            elif node.get("type") == "source":
                all_source_node_ids.append(node["id"])
                sm = node.get("source_mode", "video")
                if sm not in ("spout", "ndi", "syphon"):
                    webrtc_source_node_ids.append(node["id"])
                else:
                    has_non_webrtc_sources = True
            elif node.get("type") == "record":
                record_node_ids.append(node["id"])
    return (
        webrtc_sink_node_ids,
        webrtc_source_node_ids,
        all_sink_node_ids,
        all_source_node_ids,
        record_node_ids,
        has_non_webrtc_sources,
    )


class Session:
    """WebRTC Session containing peer connection and associated tracks."""

    def __init__(
        self,
        pc: RTCPeerConnection,
        video_track: MediaStreamTrack | None = None,
        audio_track: "AudioProcessingTrack | None" = None,
        frame_processor: "FrameProcessor | None" = None,
        data_channel: RTCDataChannel | None = None,
        relay: MediaRelay | None = None,
        recording_manager: RecordingManager | None = None,
        user_id: str | None = None,
        connection_id: str | None = None,
        connection_info: dict | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.pc = pc
        self.video_track = video_track
        self.audio_track = audio_track
        self.frame_processor = frame_processor
        self.data_channel = data_channel
        self.relay = relay
        self.recording_manager = recording_manager
        self.user_id = user_id
        self.connection_id = connection_id
        self.connection_info = connection_info
        self.notification_sender = None
        self.tempo_sync = None
        # Multi-sink/source support
        self.additional_tracks: list[NodeOutputTrack] = []
        self.input_handlers: list[SourceInputHandler] = []

    async def close(self):
        """Close this session and cleanup resources."""
        try:
            if self.tempo_sync is not None and self.notification_sender is not None:
                self.tempo_sync.unregister_notification_session(
                    self.notification_sender
                )

            # Stop additional sink tracks
            for track in self.additional_tracks:
                track.stop()

            # Stop additional input handlers
            for handler in self.input_handlers:
                await handler.stop()

            # Stop tracks first
            if self.video_track is not None:
                await self.video_track.stop()
            if self.audio_track is not None:
                self.audio_track.stop()

            # Stop frame processor (owned by session, shared between tracks)
            if self.frame_processor is not None:
                self.frame_processor.stop()

            if self.pc is not None and self.pc.connectionState not in [
                "closed",
                "failed",
            ]:
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
                        if message.get("type") != "tempo_update":
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


def _publish_connection_error(
    session_id: str | None,
    connection_id: str | None,
    user_id: str | None,
    connection_info: dict | None,
    error_message: str,
    exception_type: str,
    error_type: str,
    mode: str,
    phase: str | None = None,
) -> None:
    """Publish a connection error event to Kafka (fire-and-forget)."""
    metadata = {"mode": mode}
    if phase:
        metadata["phase"] = phase

    publish_event(
        event_type="error",
        session_id=session_id,
        connection_id=connection_id,
        user_id=user_id,
        error={
            "error_type": error_type,
            "message": error_message,
            "exception_type": exception_type,
            "recoverable": False,
        },
        metadata=metadata,
        connection_info=connection_info,
    )


class WebRTCManager:
    """
    Manages multiple WebRTC peer connections using sessions.
    """

    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.headless_session: HeadlessSession | None = None
        self.rtc_config = create_rtc_config()
        self.is_first_track = True

    async def handle_offer(
        self,
        request: WebRTCOfferRequest,
        pipeline_manager: PipelineManager,
        tempo_sync=None,
    ) -> dict[str, Any]:
        """
        Handle an incoming WebRTC offer and return an answer.

        Args:
            offer_data: Dictionary containing SDP offer
            pipeline_manager: The pipeline manager instance
            tempo_sync: Optional TempoSync instance for beat state injection

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
            session = Session(
                pc,
                user_id=request.user_id,
                connection_id=request.connection_id,
                connection_info=request.connection_info,
            )
            self.sessions[session.id] = session

            # Create NotificationSender for this session to send notifications to the frontend
            notification_sender = NotificationSender()

            # Determine media modalities from the local pipeline registry
            # (authoritative for local mode). initial_parameters values are not
            # used here because they may be stale from a previous pipeline load.
            pipeline_ids = initial_parameters.get("pipeline_ids", [])
            produces_video = NodeRegistry.chain_produces_video(pipeline_ids)

            # Parse graph from initial parameters to find sink/source/record node IDs
            (
                _,  # webrtc_sink_node_ids (unused — see below)
                webrtc_source_node_ids,
                sink_node_ids,
                _,  # all_source_node_ids (browser tracks map to webrtc sources only)
                record_node_ids,
                has_non_webrtc_sources,
            ) = _parse_graph_node_ids(initial_parameters)
            # Use all_sink_node_ids (not webrtc_sink_node_ids) so that
            # SinkOutputTracks are created for every sink, including
            # NDI/Spout/Syphon ones. In cloud relay mode the cloud sends
            # all sink output back via WebRTC; in browser mode, the extra
            # tracks are harmless (attached via addTrack fallback, ignored
            # by the browser).  The graph executor creates separate
            # hardware queues for NDI/Spout/Syphon output threads, so
            # SinkOutputTrack reads from the WebRTC queue without conflict.

            # In cloud relay mode the local instance sends WebRTC tracks for
            # ALL sources (including NDI/Syphon/Spout) and signals this via
            # ``source_track_order`` in initial_parameters. Use that list so
            # incoming tracks are mapped to the correct source nodes.
            #
            # In browser mode there is no ``source_track_order``; the browser
            # only sends tracks for file/camera sources. Use
            # webrtc_source_node_ids so track indices align with the reduced
            # set of non-hardware sources.
            all_source_node_ids_for_routing = initial_parameters.get(
                "source_track_order", webrtc_source_node_ids
            )

            # If the graph has pipeline nodes, ensure they are loaded keyed by
            # node_id so build_graph can find them via node.id.  The pipeline
            # may already be loaded under its pipeline_id (e.g. from the
            # load_pipeline API), so we re-register it under the node_id key.
            #
            # Snapshot original instances first: if a node_id matches an
            # already-loaded pipeline_id (e.g. node "passthrough" wants
            # pipeline_id "split-screen", but "passthrough" is also loaded),
            # a plain alias_pipeline would silently skip the override.  By
            # snapshotting we guarantee each node gets the correct instance
            # even when names collide.
            graph_data = initial_parameters.get("graph")
            if graph_data and isinstance(graph_data, dict):
                original_instances: dict[str, Any] = {}
                for node in graph_data.get("nodes", []):
                    if node.get("type") == "pipeline" and node.get("pipeline_id"):
                        pid = node["pipeline_id"]
                        if pid not in original_instances:
                            try:
                                original_instances[pid] = (
                                    pipeline_manager.get_pipeline_by_id(pid)
                                )
                            except Exception:
                                pass
                for node in graph_data.get("nodes", []):
                    if node.get("type") == "pipeline" and node.get("pipeline_id"):
                        pid = node["pipeline_id"]
                        if pid in original_instances:
                            pipeline_manager.set_pipeline_instance(
                                node["id"], original_instances[pid]
                            )
                            logger.info(
                                f"Re-keyed pipeline {pid} as {node['id']} for graph"
                            )
                # Plain custom nodes aren't pre-loaded via /pipeline/load,
                # but the graph executor still resolves them through the
                # PipelineManager — so instantiate and register them here.
                pipeline_manager.register_graph_nodes(graph_data)

            # Create FrameProcessor (owned by session, shared between tracks)
            frame_processor = FrameProcessor(
                pipeline_manager=pipeline_manager,
                initial_parameters=initial_parameters,
                notification_callback=notification_sender.call,
                session_id=session.id,
                user_id=request.user_id,
                connection_id=request.connection_id,
                connection_info=request.connection_info,
                tempo_sync=tempo_sync,
            )
            frame_processor.start()
            session.frame_processor = frame_processor

            video_track = None
            relay = None
            audio_track = None
            audio_relay = None

            if produces_video:
                video_track = VideoProcessingTrack(
                    pipeline_manager,
                    initial_parameters=initial_parameters,
                    notification_callback=notification_sender.call,
                    session_id=session.id,
                    user_id=request.user_id,
                    connection_id=request.connection_id,
                    connection_info=request.connection_info,
                    frame_processor=frame_processor,
                )
                session.video_track = video_track

                # Create a MediaRelay to allow multiple consumers (WebRTC and recording)
                relay = MediaRelay()

                # When graph sinks exist, use a SinkOutputTrack for the first
                # sink instead of VideoProcessingTrack — all sinks are treated
                # equally.  VideoProcessingTrack still handles input/lifecycle.
                if sink_node_ids:
                    first_sink_track = SinkOutputTrack(
                        frame_processor=frame_processor,
                        sink_node_id=sink_node_ids[0],
                    )
                    session.additional_tracks.append(first_sink_track)
                    relayed_track = relay.subscribe(first_sink_track)
                else:
                    relayed_track = relay.subscribe(video_track)

                # Add the relayed video track to WebRTC connection
                pc.addTrack(relayed_track)

                # Extra sink tracks are added AFTER setRemoteDescription (below)
                # so they can be placed on the correct recvonly transceivers.

                # Eagerly initialize frame processor when graph has sources,
                # sinks, or record nodes, so SourceInputHandler / SinkOutputTrack
                # and recording endpoints can reference it immediately.
                if (
                    webrtc_source_node_ids
                    or has_non_webrtc_sources
                    or sink_node_ids
                    or record_node_ids
                ):
                    video_track.initialize_output_processing()
                # When graph sources are handled externally (SourceInputHandler
                # for WebRTC multi-source, or _setup_multi_input_sources for
                # Syphon/NDI/Spout), signal recv() to keep running.
                if webrtc_source_node_ids or has_non_webrtc_sources:
                    video_track.has_external_input = True

                # Store relay for cleanup
                session.relay = relay
            else:
                logger.info(
                    f"Pipeline(s) {pipeline_ids} do not produce video, "
                    "skipping video track"
                )

            produces_audio = NodeRegistry.chain_produces_audio(pipeline_ids)
            if produces_audio:
                audio_track = AudioProcessingTrack(
                    frame_processor=frame_processor,
                )
                session.audio_track = audio_track

                # Create audio relay for recording (and potentially other consumers)
                audio_relay = MediaRelay()
            else:
                logger.info(
                    f"Pipeline(s) {pipeline_ids} do not produce audio, "
                    "skipping audio track"
                )

            # Recording setup (works for video-only, audio-only, or both)
            from .recording import RECORDING_ENABLED

            recording_param = initial_parameters.get("recording")
            recording_enabled = (
                recording_param if recording_param is not None else RECORDING_ENABLED
            )
            if recording_enabled and (
                video_track is not None or audio_track is not None
            ):
                # Graph record nodes use per-node RecordingCoordinator via
                # /api/v1/recordings/...?node_id= — a single session manager
                # would conflict with multiple stop/download cycles.
                if record_node_ids:
                    session.recording_manager = None
                else:
                    recording_manager = RecordingManager(
                        video_track=video_track,
                        audio_track=audio_track,
                    )
                    session.recording_manager = recording_manager
                    if relay is not None:
                        recording_manager.set_relay(relay)
                    if audio_relay is not None:
                        recording_manager.set_audio_relay(audio_relay)

                    async def start_recording_when_ready():
                        """Start recording when frames start flowing."""
                        try:
                            await asyncio.sleep(0.1)
                            await recording_manager.start_recording()
                        except Exception as e:
                            logger.debug(f"Could not start recording yet: {e}")

                    asyncio.create_task(start_recording_when_ready())
            else:
                session.recording_manager = None

            session.notification_sender = notification_sender
            session.tempo_sync = tempo_sync
            if tempo_sync is not None:
                tempo_sync.register_notification_session(notification_sender)

            logger.info(f"Created new session: {session}")

            # Counter for matching incoming video tracks to source nodes
            received_video_count = [0]

            @pc.on("track")
            def on_track(track: MediaStreamTrack):
                logger.info(f"Track received: {track.kind} for session {session.id}")
                if track.kind == "video" and video_track is not None:
                    if all_source_node_ids_for_routing:
                        # Multi-source: route each incoming browser track via
                        # SourceInputHandler (one track per file/camera source;
                        # hardware sources do not use this path).
                        idx = received_video_count[0]
                        received_video_count[0] += 1
                        if idx < len(all_source_node_ids_for_routing):
                            handler = SourceInputHandler(
                                frame_processor=video_track.frame_processor,
                                source_node_id=all_source_node_ids_for_routing[idx],
                            )
                            handler.start(track)
                            session.input_handlers.append(handler)
                            logger.info(
                                f"Added input handler for source node "
                                f"{all_source_node_ids_for_routing[idx]}"
                            )
                    else:
                        video_track.initialize_input_processing(track)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(
                    f"Connection state changed to: {pc.connectionState} for session {session.id}"
                )
                if pc.connectionState == "failed":
                    _publish_connection_error(
                        session.id,
                        session.connection_id,
                        session.user_id,
                        session.connection_info,
                        "WebRTC connection failed",
                        "WebRTCConnectionError",
                        error_type="webrtc_connection_failed",
                        mode="local",
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
                        logger.debug(f"Received parameter update: {data}")

                        # Always handle paused immediately (before quantized
                        # scheduling) so pause/unpause is never delayed.
                        if "paused" in data:
                            if session.video_track:
                                session.video_track.pause(data["paused"])
                            elif session.frame_processor:
                                session.frame_processor.paused = data["paused"]

                        # Check for quantized update flag
                        if data.pop("_quantized", False):
                            fp = session.frame_processor
                            if (
                                not fp
                                and session.video_track
                                and hasattr(session.video_track, "frame_processor")
                            ):
                                fp = session.video_track.frame_processor
                            if fp:
                                fp.schedule_quantized_update(data)
                            return

                        # Send parameters to the frame processor
                        if session.frame_processor:
                            session.frame_processor.update_parameters(data)
                        else:
                            logger.warning(
                                "No frame processor available for parameter update"
                            )

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse parameter update message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling parameter update: {e}")

            # Set remote description (the offer).
            # The browser's offer includes a recvonly audio m-line (from
            # addTransceiver("audio", {direction: "recvonly"})). aiortc will
            # create an audio transceiver for it during setRemoteDescription.
            offer_sdp = RTCSessionDescription(sdp=request.sdp, type=request.type)
            await pc.setRemoteDescription(offer_sdp)

            # Attach our audio track to the transceiver that aiortc created
            # from the browser's recvonly audio m-line. We find it by kind,
            # assign our track to its sender, and set direction to sendonly.
            # When an audio relay exists (for recording), use a relayed
            # subscription so the relay drives recv() and fans out to both
            # WebRTC and the RecordingManager.
            if audio_track is not None:
                audio_for_webrtc = (
                    audio_relay.subscribe(audio_track)
                    if audio_relay is not None
                    else audio_track
                )
                for t in pc.getTransceivers():
                    if t.kind == "audio":
                        t.sender.replaceTrack(audio_for_webrtc)
                        t.direction = "sendonly"
                        logger.info(
                            f"Audio track attached to transceiver (mid={t.mid})"
                        )
                        break

            # Attach extra sink output tracks to the recvonly transceivers that
            # the browser (or cloud client) created. Record nodes are not live
            # WebRTC outputs because their queues are consumed by recordings.
            # Must happen AFTER setRemoteDescription so the transceivers exist.
            extra_output_tracks: list[NodeOutputTrack] = []
            if video_track is not None and relay is not None:
                extra_output_tracks = _build_extra_output_tracks(
                    frame_processor=frame_processor,
                    sink_node_ids=sink_node_ids,
                    record_node_ids=record_node_ids,
                )

            if extra_output_tracks:
                recv_only_video = [
                    t
                    for t in pc.getTransceivers()
                    if t.kind == "video"
                    and t.sender.track is None
                    and t.receiver.track is None
                ]
                for i, extra_track in enumerate(extra_output_tracks):
                    session.additional_tracks.append(extra_track)
                    relayed = relay.subscribe(extra_track)
                    if i < len(recv_only_video):
                        t = recv_only_video[i]
                        t.sender.replaceTrack(relayed)
                        t.direction = "sendonly"
                        logger.info(
                            f"Attached extra output track on transceiver mid={t.mid}"
                        )
                    else:
                        pc.addTrack(relayed)
                        logger.info("Attached extra output track via addTrack fallback")

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            # Publish session_created event
            pipeline_ids = initial_parameters.get("pipeline_ids")
            publish_event(
                event_type="session_created",
                session_id=session.id,
                connection_id=request.connection_id,
                pipeline_ids=pipeline_ids if pipeline_ids else None,
                user_id=request.user_id,
                metadata={"mode": "local"},
                connection_info=request.connection_info,
            )

            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionId": session.id,
            }

        except Exception as e:
            logger.error(f"Error handling WebRTC offer: {e}", exc_info=True)
            _publish_connection_error(
                session.id if "session" in locals() else None,
                request.connection_id,
                request.user_id,
                request.connection_info,
                str(e),
                type(e).__name__,
                error_type="webrtc_offer_failed",
                mode="local",
                phase="offer_handling",
            )
            if "session" in locals():
                await self.remove_session(session.id)
            raise

    async def handle_offer_with_relay(
        self, request: WebRTCOfferRequest, cloud_manager: "ScopeCloudBackend"
    ) -> dict[str, Any]:
        """
        Handle WebRTC offer and relay video through cloud for processing.

        This creates a CloudTrack that:
        1. Receives video from the browser
        2. Sends it to cloud for processing
        3. Returns processed frames to the browser

        Args:
            request: WebRTC offer request
            cloud_manager: The active remote backend for relay connection

        Returns:
            Dictionary containing SDP answer
        """
        try:
            # Extract initial parameters from offer
            initial_parameters = {}
            if request.initialParameters:
                initial_parameters = request.initialParameters.model_dump(
                    exclude_none=True
                )
            logger.info(f"Received offer with parameters: {initial_parameters}")

            # Create new RTCPeerConnection with configuration
            pc = RTCPeerConnection(self.rtc_config)
            session = Session(
                pc,
                user_id=request.user_id,
                connection_id=request.connection_id,
                connection_info=request.connection_info,
            )
            self.sessions[session.id] = session

            # Parse graph from initial parameters for multi-source/sink/record
            (
                _,  # webrtc_sink_node_ids (unused — use all_sink_node_ids)
                webrtc_source_node_ids,
                sink_node_ids,
                _,  # all_source_node_ids
                record_node_ids,
                has_non_webrtc_sources,
            ) = _parse_graph_node_ids(initial_parameters)

            # Determine media modalities from initial_parameters. These are
            # set by the frontend from the pipeline/status endpoint, which is
            # proxied to the cloud backend — so they reflect the cloud
            # pipeline's actual capabilities (even for cloud-only pipelines
            # not registered locally).
            produces_video = initial_parameters.get("produces_video", True)
            produces_audio = initial_parameters.get("produces_audio", False)

            # Create FrameProcessor in cloud mode so it can be shared
            # between CloudTrack (video) and AudioProcessingTrack (audio)
            frame_processor = FrameProcessor(
                pipeline_manager=None,  # Not needed in cloud mode
                initial_parameters=initial_parameters,
                cloud_manager=cloud_manager,
                session_id=session.id,
                user_id=request.user_id,
                connection_id=request.connection_id,
                connection_info=request.connection_info,
            )
            session.frame_processor = frame_processor

            cloud_track = None
            audio_track = None

            if produces_video:
                cloud_track = CloudTrack(
                    cloud_manager=cloud_manager,
                    # Once incoming timestamps are good everywhere,remove this
                    preserve_output_timestamps=isinstance(
                        cloud_manager, LivepeerConnection
                    ),
                    initial_parameters=initial_parameters,
                    user_id=request.user_id,
                    connection_id=request.connection_id,
                    connection_info=request.connection_info,
                    session_id=session.id,
                    frame_processor=frame_processor,
                )
                session.video_track = cloud_track

                relay = MediaRelay()
                relayed_track = relay.subscribe(cloud_track)
                pc.addTrack(relayed_track)
                session.relay = relay
            else:
                logger.info(
                    "Audio-only cloud pipeline, skipping CloudTrack. "
                    "Starting cloud connection directly."
                )

                async def _start_cloud():
                    await cloud_manager.start_webrtc(initial_parameters)
                    frame_processor.start()

                asyncio.create_task(_start_cloud())

            audio_relay = None
            if produces_audio:
                audio_track = AudioProcessingTrack(
                    frame_processor=frame_processor,
                )
                session.audio_track = audio_track

            # Recording setup (local recording from relayed cloud frames)
            from .recording import RECORDING_ENABLED

            recording_param = initial_parameters.get("recording")
            recording_enabled = (
                recording_param if recording_param is not None else RECORDING_ENABLED
            )
            if recording_enabled and (
                cloud_track is not None or audio_track is not None
            ):
                if record_node_ids:
                    session.recording_manager = None
                else:
                    recording_manager = RecordingManager(
                        video_track=cloud_track,
                        audio_track=audio_track,
                    )
                    session.recording_manager = recording_manager
                    if relay is not None:
                        recording_manager.set_relay(relay)
                    if audio_track is not None:
                        audio_relay = MediaRelay()
                        recording_manager.set_audio_relay(audio_relay)

                    async def start_recording_when_ready():
                        try:
                            await asyncio.sleep(0.1)
                            await recording_manager.start_recording()
                        except Exception as e:
                            logger.debug(f"Could not start recording yet: {e}")

                    asyncio.create_task(start_recording_when_ready())
            else:
                session.recording_manager = None

            logger.info(f"Created session: {session.id}")

            video_track_index = [0]

            @pc.on("track")
            def on_track(track: MediaStreamTrack):
                logger.info(f"Track received: {track.kind} for session {session.id}")
                if track.kind == "video" and cloud_track is not None:
                    # When all sources are server-side hardware (Syphon/NDI/
                    # Spout), ignore the browser video track — it carries no
                    # useful data and would collide with hardware-source frames
                    # on the cloud input track, producing corrupt VP8 bitstreams.
                    if has_non_webrtc_sources and not webrtc_source_node_ids:
                        logger.info(
                            "Ignoring browser video track (all sources are "
                            "server-side hardware)"
                        )
                        return

                    idx = video_track_index[0]
                    video_track_index[0] += 1
                    if webrtc_source_node_ids:
                        # Graph with WebRTC source nodes: route ALL browser
                        # tracks (including the primary) through
                        # CloudSourceInputHandler so each one writes directly
                        # to the correct cloud input track.  Using
                        # set_source_track + _input_loop would funnel frames
                        # through FrameProcessor.put() → send_frame(generic
                        # track 0), colliding with hardware-source frames on
                        # the same track.
                        #
                        # Pass the source NODE ID (not the cloud track index)
                        # to CloudTrack. The cloud track index is resolved
                        # later inside CloudTrack._start, after the cloud
                        # relay's webrtc_client has been (re)connected with
                        # this graph and `source_node_to_track_index` is
                        # populated. Resolving the index here would either
                        # use a stale mapping from a prior session or fall
                        # back to the browser's receive order — both of
                        # which mis-route Camera frames into Syphon/NDI
                        # source pipelines for mixed-source graphs.
                        if idx < len(webrtc_source_node_ids):
                            node_id = webrtc_source_node_ids[idx]
                            cloud_track.add_extra_source_track(node_id, track)
                        else:
                            logger.warning(
                                f"Browser sent video track index {idx} but "
                                f"only {len(webrtc_source_node_ids)} WebRTC "
                                "source node(s) in graph; ignoring extra track"
                            )
                    else:
                        # No graph or non-graph perform mode — use the generic
                        # CloudTrack input path (single source, no routing).
                        cloud_track.set_source_track(track)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(
                    f"Connection state: {pc.connectionState} for session {session.id}"
                )
                if pc.connectionState == "failed":
                    _publish_connection_error(
                        session.id,
                        session.connection_id,
                        session.user_id,
                        session.connection_info,
                        "WebRTC connection failed",
                        "WebRTCConnectionError",
                        error_type="webrtc_connection_failed",
                        mode="relay",
                    )
                if pc.connectionState in ["closed", "failed"]:
                    if cloud_track is not None:
                        await cloud_track.stop()
                    await self.remove_session(session.id)

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                logger.info(
                    f"ICE state: {pc.iceConnectionState} for session {session.id}"
                )

            # Handle data channel for parameter updates
            @pc.on("datachannel")
            def on_data_channel(data_channel):
                logger.info(f"Data channel: {data_channel.label}")
                session.data_channel = data_channel

                @data_channel.on("message")
                def on_data_channel_message(message):
                    try:
                        data = json.loads(message)
                        logger.debug(f"Parameter update: {data}")

                        # Forward parameters to cloud and frame processor
                        if cloud_track is not None:
                            cloud_track.update_parameters(data)
                        else:
                            if frame_processor:
                                frame_processor.update_parameters(data)
                            cloud_manager.send_parameters(data)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")

            # Set remote description (the offer).
            offer_sdp = RTCSessionDescription(sdp=request.sdp, type=request.type)
            await pc.setRemoteDescription(offer_sdp)

            # Attach our audio track to the transceiver that aiortc
            # created from the browser's recvonly audio m-line (if present).
            # When an audio relay exists (for recording), use a relayed
            # subscription so the relay fans out to both WebRTC and the
            # RecordingManager.
            if audio_track is not None:
                audio_for_webrtc = (
                    audio_relay.subscribe(audio_track)
                    if audio_relay is not None
                    else audio_track
                )
                for t in pc.getTransceivers():
                    if t.kind == "audio":
                        t.sender.replaceTrack(audio_for_webrtc)
                        t.direction = "sendonly"
                        logger.info(
                            f"Audio track attached to transceiver (mid={t.mid})"
                        )
                        break

            # Attach extra sink video tracks to the browser's recvonly
            # transceivers (same pattern as the local-mode handler).
            if len(sink_node_ids) > 1 and cloud_track is not None and relay is not None:
                from .cloud_track import CloudSinkOutputTrack

                extra_sink_tracks: list[CloudSinkOutputTrack] = []
                recv_only_video = [
                    t
                    for t in pc.getTransceivers()
                    if t.kind == "video"
                    and t.sender.track is None
                    and t.receiver.track is None
                ]
                logger.info(
                    f"Cloud relay: {len(recv_only_video)} recv-only video "
                    f"transceivers for {len(sink_node_ids) - 1} extra sink(s)"
                )
                for i, sink_id in enumerate(sink_node_ids[1:]):
                    extra_track = CloudSinkOutputTrack()
                    extra_sink_tracks.append(extra_track)
                    session.additional_tracks.append(extra_track)
                    relayed = relay.subscribe(extra_track)
                    if i < len(recv_only_video):
                        tv = recv_only_video[i]
                        tv.sender.replaceTrack(relayed)
                        tv.direction = "sendonly"
                        logger.info(
                            f"Cloud relay: extra sink {sink_id} on mid={tv.mid}"
                        )
                    else:
                        pc.addTrack(relayed)
                        logger.info(f"Cloud relay: extra sink {sink_id} via addTrack")
                # Tell CloudTrack to wire these after cloud connection starts
                cloud_track.set_extra_sink_tracks(extra_sink_tracks)

            # Set up record node callbacks so cloud record frames are
            # received locally and fed into frame_processor record queues.
            # Record tracks are NOT relayed to the browser — they stay
            # server-side for local recording.
            if record_node_ids and cloud_track is not None:
                record_callbacks: list[tuple[str, Callable]] = []
                for rec_id in record_node_ids:

                    def _make_record_cb(node_id: str, fp: FrameProcessor) -> Callable:
                        def _cb(frame) -> None:
                            fp.sink_manager.put_to_record(node_id, frame)

                        return _cb

                    record_callbacks.append(
                        (rec_id, _make_record_cb(rec_id, frame_processor))
                    )
                cloud_track.set_record_callbacks(record_callbacks)
                logger.info(
                    f"Cloud relay: registered {len(record_callbacks)} "
                    f"record callback(s)"
                )

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            # Publish session_created event for relay mode
            pipeline_ids = initial_parameters.get("pipeline_ids")
            publish_event(
                event_type="session_created",
                session_id=session.id,
                connection_id=request.connection_id,
                pipeline_ids=pipeline_ids if pipeline_ids else None,
                user_id=request.user_id,
                metadata={"mode": "relay"},
                connection_info=request.connection_info,
            )

            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionId": session.id,
            }

        except Exception as e:
            logger.error(f"Error handling offer: {e}")
            _publish_connection_error(
                session.id if "session" in locals() else None,
                request.connection_id,
                request.user_id,
                request.connection_info,
                str(e),
                type(e).__name__,
                error_type="webrtc_offer_failed",
                mode="relay",
                phase="offer_handling",
            )
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

            # Publish session_closed event
            publish_event(
                event_type="session_closed",
                session_id=session_id,
                connection_id=session.connection_id,
                user_id=session.user_id,
                connection_info=session.connection_info,
            )
        else:
            logger.warning(f"Attempted to remove non-existent session: {session_id}")

    def get_session(self, session_id: str) -> Session | HeadlessSession | None:
        """Get a session by ID.

        Use session_id="headless" to retrieve the active headless session.
        """
        if session_id == "headless":
            return self.headless_session
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

    def add_headless_session(self, session: HeadlessSession) -> None:
        """Register the headless session (only one supported at a time)."""
        self.headless_session = session

    async def remove_headless_session(self) -> None:
        """Stop and remove the active headless session."""
        session = self.headless_session
        self.headless_session = None
        if session:
            await session.close()
        else:
            logger.warning("Attempted to remove non-existent headless session")

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

    def get_frame_processor(self) -> tuple[str, "FrameProcessor", bool] | None:
        """Return (session_id, frame_processor, is_headless) for the active session, or None."""
        for sid, session in self.sessions.items():
            if session.pc.connectionState in ("closed", "failed"):
                continue
            fp = None
            if (
                session.video_track
                and hasattr(session.video_track, "frame_processor")
                and session.video_track.frame_processor
            ):
                fp = session.video_track.frame_processor
            elif session.frame_processor:
                fp = session.frame_processor
            if fp:
                return sid, fp, False
        if self.headless_session and self.headless_session.frame_processor:
            return "headless", self.headless_session.frame_processor, True
        return None

    def get_last_frame(self, sink_node_id: str | None = None):
        """Return the most recent frame from the active session, or None.

        Args:
            sink_node_id: If provided, return the last frame from this specific
                sink node (multi-sink graph mode). If None, return the most
                recent frame from any sink.
        """
        if sink_node_id is None:
            for session in self.sessions.values():
                if session.video_track and hasattr(
                    session.video_track, "get_last_frame"
                ):
                    frame = session.video_track.get_last_frame()
                    if frame is not None:
                        return frame
        if self.headless_session:
            frame = self.headless_session.get_last_frame(sink_node_id=sink_node_id)
            if frame is not None:
                return frame
        return None

    def broadcast_parameter_update(self, parameters: dict) -> None:
        """Send a parameter update to all active sessions (e.g. from OSC or REST API)."""
        for session in self.sessions.values():
            if session.pc.connectionState in ("closed", "failed"):
                continue
            if "paused" in parameters:
                if session.video_track:
                    session.video_track.pause(parameters["paused"])
                elif session.frame_processor:
                    session.frame_processor.paused = parameters["paused"]
            if (
                session.video_track
                and hasattr(session.video_track, "frame_processor")
                and session.video_track.frame_processor
            ):
                session.video_track.frame_processor.update_parameters(parameters)
            elif session.frame_processor:
                session.frame_processor.update_parameters(parameters)
        if self.headless_session and self.headless_session.frame_processor:
            self.headless_session.frame_processor.update_parameters(parameters)

    def broadcast_notification(self, message: dict) -> None:
        """Send a notification to active sessions via their data channels.

        Safe to call from worker threads: sends are marshalled onto the
        aiortc event loop through each session's :class:`NotificationSender`,
        which also buffers messages until the data channel is open.
        """
        for session in self.sessions.values():
            if session.pc.connectionState in ("closed", "failed"):
                continue
            sender = session.notification_sender
            if sender is None:
                continue
            sender.call(message)

    async def stop(self):
        """Close and cleanup all sessions (WebRTC and headless)."""
        close_tasks = [session.close() for session in self.sessions.values()]
        if self.headless_session:
            close_tasks.append(self.headless_session.close())
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self.sessions.clear()
        self.headless_session = None


def create_rtc_config() -> RTCConfiguration:
    """Setup RTCConfiguration with TURN credentials if available."""
    try:
        from huggingface_hub import get_token

        hf_token = get_token()
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
