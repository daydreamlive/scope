"""CloudTrack - MediaStreamTrack that relays video through cloud.

This track receives frames from a source (browser WebRTC or Spout),
sends them to cloud for processing, and returns the processed frames.

Architecture:
    Browser/Spout → CloudTrack → FrameProcessor (cloud mode) → Cloud
                                                                  ↓
    Browser/Spout ← CloudTrack ← FrameProcessor (cloud mode) ← Cloud

Spout integration is handled by FrameProcessor (same code as local mode).
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import VideoFrame

from .media_packets import ensure_video_packet

if TYPE_CHECKING:
    from .frame_processor import FrameProcessor
    from .scope_cloud_types import ScopeCloudBackend

logger = logging.getLogger(__name__)


class CloudTrack(MediaStreamTrack):
    """MediaStreamTrack that relays video through cloud for processing.

    This track uses FrameProcessor in cloud mode, which handles:
    - Sending frames to cloud
    - Receiving processed frames from cloud
    - Spout input/output integration

    Usage:
        relay_track = CloudTrack(cloud_manager)
        relay_track.set_source_track(browser_video_track)
        # relay_track can now be used as a MediaStreamTrack
    """

    kind = "video"

    def __init__(
        self,
        cloud_manager: ScopeCloudBackend,
        fps: int = 30,
        preserve_output_timestamps: bool = False,
        initial_parameters: dict | None = None,
        notification_callback: Callable | None = None,
        user_id: str | None = None,
        connection_id: str | None = None,
        connection_info: dict | None = None,
        session_id: str | None = None,
        frame_processor: FrameProcessor | None = None,
    ):
        super().__init__()
        self.cloud_manager = cloud_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        self.user_id = user_id
        self.connection_id = connection_id
        self.connection_info = connection_info
        self.session_id = session_id

        # FPS control
        self.fps = fps
        self.frame_ptime = 1.0 / fps
        self.preserve_output_timestamps = preserve_output_timestamps

        # Source track for input frames (from browser)
        self._source_track: MediaStreamTrack | None = None
        self._input_task: asyncio.Task | None = None
        self._input_running = False

        # FrameProcessor handles relay to cloud and Spout integration
        self.frame_processor = frame_processor
        self._last_frame: VideoFrame | None = None
        self._started = False
        self._first_frame_emitted = False

        # Multi-source / multi-sink / record (wired up in _start after cloud connects)
        # Store (source_node_id, track). Resolving to a cloud input track index
        # is deferred until _start() runs, because the webrtc_client's
        # source_node_to_track_index mapping for *this* graph only exists
        # after `cloud_manager.start_webrtc(initial_parameters)` runs in _start.
        self._pending_extra_sources: list[tuple[str, MediaStreamTrack]] = []
        self._extra_sink_tracks: list[CloudSinkOutputTrack] = []
        self._extra_input_handlers: list[CloudSourceInputHandler] = []
        self._record_callbacks: list[tuple[str, Callable]] = []

    def set_source_track(self, track: MediaStreamTrack) -> None:
        """Set the source track for input frames (from browser)."""
        self._source_track = track
        logger.info("Source track set")

    async def _start(self) -> None:
        """Start the relay - called on first recv()."""
        if self._started:
            return

        self._started = True
        logger.info("Starting cloud relay...")

        # Start WebRTC connection to cloud with this session's parameters
        logger.info("Starting WebRTC connection to cloud...")
        await self.cloud_manager.start_webrtc(self.initial_parameters)

        # Create FrameProcessor in cloud mode (unless one was injected)
        if self.frame_processor is None:
            from .frame_processor import FrameProcessor

            self.frame_processor = FrameProcessor(
                pipeline_manager=None,  # Not needed in cloud mode
                initial_parameters=self.initial_parameters,
                notification_callback=self.notification_callback,
                cloud_manager=self.cloud_manager,  # Enable cloud mode
                session_id=self.session_id,
                user_id=self.user_id,
                connection_id=self.connection_id,
                connection_info=self.connection_info,
            )
        self.frame_processor.start()

        # Start input processing if we have a source track
        if self._source_track is not None:
            self._input_running = True
            self._input_task = asyncio.create_task(self._input_loop())

        # Wire up extra source tracks now that the cloud connection exists.
        # Resolve each pending source node ID to a cloud input track index
        # via the freshly populated source_node_to_track_index. Doing this
        # lookup here (rather than at add_extra_source_track time) is what
        # makes mixed browser+hardware source graphs route correctly: when
        # `pc.on("track")` fires for the browser camera, the webrtc_client
        # may not yet exist or may carry a stale mapping from a previous
        # session.
        #
        # TODO: This relies on duck-typed public members
        # (input_tracks/output_handlers) from get_webrtc_client().
        webrtc_client = self.cloud_manager.get_webrtc_client()
        if webrtc_client is not None:
            for source_node_id, track in self._pending_extra_sources:
                track_index = webrtc_client.source_node_to_track_index.get(
                    source_node_id
                )
                if track_index is None or track_index >= len(
                    webrtc_client.input_tracks
                ):
                    logger.warning(
                        "Could not resolve cloud input track for source node "
                        f"{source_node_id!r} (index={track_index}, "
                        f"have {len(webrtc_client.input_tracks)} input tracks)"
                    )
                    continue
                handler = CloudSourceInputHandler(
                    webrtc_client.input_tracks[track_index]
                )
                handler.start(track)
                self._extra_input_handlers.append(handler)
                logger.info(
                    f"Wired extra source track for node {source_node_id!r} "
                    f"to cloud input track {track_index}"
                )
            self._pending_extra_sources.clear()

            # Wire extra sink output callbacks (index 0 is primary, 1+ are extras)
            for i, sink_track in enumerate(self._extra_sink_tracks):
                sink_index = i + 1
                if sink_index < len(webrtc_client.output_handlers):
                    webrtc_client.output_handlers[sink_index].add_callback(
                        sink_track.put_frame
                    )
                    logger.info(f"Wired extra sink track {i} to cloud output")

            # Wire record node output callbacks (placed after sink tracks)
            num_extra_sinks = len(self._extra_sink_tracks)
            for i, (rec_id, callback) in enumerate(self._record_callbacks):
                handler_index = num_extra_sinks + 1 + i
                if handler_index < len(webrtc_client.output_handlers):
                    webrtc_client.output_handlers[handler_index].add_callback(callback)
                    logger.info(f"Wired record node {rec_id} to cloud output")

        logger.info("Relay started")

    async def _input_loop(self) -> None:
        """Background loop that receives frames from source and sends to cloud."""
        logger.info("Input loop started")
        consecutive_errors = 0
        max_consecutive_errors = 10

        try:
            while self._input_running and self._source_track is not None:
                try:
                    # Get frame from browser
                    frame = await self._source_track.recv()
                    consecutive_errors = 0

                    # Send through FrameProcessor (which relays to cloud)
                    if self.frame_processor:
                        self.frame_processor.put(frame)

                except MediaStreamError:
                    logger.info("Source track ended")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"Error in input loop, stopping after "
                            f"{consecutive_errors} consecutive errors: {e}"
                        )
                        break
                    logger.warning(
                        f"Transient error in input loop "
                        f"({consecutive_errors}/{max_consecutive_errors}): {e}"
                    )
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        finally:
            self._input_running = False
            stats = (
                self.frame_processor.get_frame_stats() if self.frame_processor else {}
            )
            logger.info(f"Input loop ended, stats: {stats}")

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Override to control frame rate."""
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "timestamp"):
            current_time = time.time()
            time_since_last_frame = current_time - self.last_frame_time

            target_interval = self.frame_ptime
            wait_time = target_interval - time_since_last_frame

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self.timestamp += int(self.frame_ptime * VIDEO_CLOCK_RATE)
            self.last_frame_time = time.time()
        else:
            self.start = time.time()
            self.last_frame_time = time.time()
            self.timestamp = 0

        return self.timestamp, VIDEO_TIME_BASE

    async def recv(self) -> VideoFrame:
        """Return the next processed frame from cloud."""
        # Lazy initialization
        await self._start()

        while True:
            if self.frame_processor:
                frame_packet = self.frame_processor.get_packet()
                if frame_packet is not None:
                    packet = ensure_video_packet(frame_packet)
                    frame_np = packet.tensor.numpy()
                    frame = VideoFrame.from_ndarray(frame_np, format="rgb24")
                    if not self._first_frame_emitted:
                        self._first_frame_emitted = True
                        self.frame_processor.notify_first_video_packet()
                        await asyncio.to_thread(
                            self.frame_processor.wait_for_av_sync_anchor
                        )
                    if self.preserve_output_timestamps and packet.timestamp.is_valid:
                        frame.pts = packet.timestamp.pts
                        frame.time_base = packet.timestamp.time_base
                    else:
                        pts, time_base = await self.next_timestamp()
                        frame.pts = pts
                        frame.time_base = time_base

                    self._last_frame = frame
                    return frame

            await asyncio.sleep(0.01)

    def update_parameters(self, params: dict) -> None:
        """Update pipeline parameters on cloud."""
        # Send to cloud first
        self.cloud_manager.send_parameters(params)

        # Handle local concerns (Spout/NDI settings) via FrameProcessor
        if self.frame_processor:
            self.frame_processor.update_parameters(params)

    def pause(self, paused: bool) -> None:
        """Pause/unpause the relay."""
        if self.frame_processor:
            self.frame_processor.paused = paused
        logger.info(f"{'Paused' if paused else 'Resumed'}")

    async def stop(self) -> None:
        """Stop the relay and clean up."""
        logger.info("Stopping...")

        self._input_running = False
        self._started = False  # Reset so next session starts fresh

        # Stop extra source input handlers and clear all multi-source/sink state
        for handler in self._extra_input_handlers:
            await handler.stop()
        self._extra_input_handlers.clear()
        self._extra_sink_tracks.clear()
        self._record_callbacks.clear()
        self._pending_extra_sources.clear()

        if self._input_task:
            self._input_task.cancel()
            try:
                await self._input_task
            except asyncio.CancelledError:
                pass
            self._input_task = None

        # Stop FrameProcessor (handles Spout cleanup and cloud callback removal)
        if self.frame_processor:
            self.frame_processor.stop()
            stats = self.frame_processor.get_frame_stats()
            logger.info(f"Stopped. Stats: {stats}")
            self.frame_processor = None
        else:
            logger.info("Stopped.")

        # Stop WebRTC connection to cloud - next session will start fresh
        await self.cloud_manager.stop_webrtc()

    # ------------------------------------------------------------------
    # Multi-source / multi-sink helpers
    # ------------------------------------------------------------------

    def add_extra_source_track(
        self, source_node_id: str, track: MediaStreamTrack
    ) -> None:
        """Register an extra browser source track to forward to cloud.

        *source_node_id* identifies which graph source node this browser
        track belongs to. The cloud input track index is resolved from
        ``webrtc_client.source_node_to_track_index`` — either immediately
        if the cloud relay is already started, or in :meth:`_start` once
        ``cloud_manager.start_webrtc(initial_parameters)`` has populated
        the mapping for the current graph.

        Storing a node ID rather than an index here is critical: when
        `pc.on("track")` fires for browser camera tracks, the cloud
        relay's webrtc_client may be missing or carry stale mapping from
        a previous session, so any precomputed index would point at the
        wrong cloud input track and frames from a Camera source would be
        delivered to a Syphon (or other hardware) source's pipeline.
        """
        if self._started:
            # TODO: This relies on duck-typed public members
            # (input_tracks/output_handlers) from get_webrtc_client().
            webrtc_client = self.cloud_manager.get_webrtc_client()
            if webrtc_client is not None:
                track_index = webrtc_client.source_node_to_track_index.get(
                    source_node_id
                )
                if track_index is not None and track_index < len(
                    webrtc_client.input_tracks
                ):
                    handler = CloudSourceInputHandler(
                        webrtc_client.input_tracks[track_index]
                    )
                    handler.start(track)
                    self._extra_input_handlers.append(handler)
                    logger.info(
                        f"Wired extra source track for node {source_node_id!r} "
                        f"to cloud input track {track_index} (immediate)"
                    )
                    return
            logger.warning(
                "Could not resolve cloud input track for source node "
                f"{source_node_id!r} (immediate path); buffering for _start"
            )
            self._pending_extra_sources.append((source_node_id, track))
        else:
            self._pending_extra_sources.append((source_node_id, track))

    def set_extra_sink_tracks(self, tracks: list[CloudSinkOutputTrack]) -> None:
        """Register extra sink output tracks that need cloud callbacks."""
        self._extra_sink_tracks = list(tracks)

    def set_record_callbacks(self, callbacks: list[tuple[str, Callable]]) -> None:
        """Register callbacks for record node outputs from cloud.

        Each entry is (record_node_id, callback) where callback receives a
        VideoFrame and pushes it into the frame_processor's record queue.
        """
        self._record_callbacks = list(callbacks)

    def get_stats(self) -> dict:
        """Get relay statistics."""
        if self.frame_processor:
            return self.frame_processor.get_frame_stats()
        return {}


class CloudSinkOutputTrack(MediaStreamTrack):
    """Lightweight bridge: cloud output track → browser for extra sinks.

    Frames are pushed in via :meth:`put_frame` (called from the
    ``CloudWebRTCClient`` receive loop) and pulled by aiortc via
    :meth:`recv`.
    """

    kind = "video"

    def __init__(self, fps: int = 30, frame_processor: FrameProcessor | None = None):
        super().__init__()
        self._queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=2)
        self.fps = fps
        self.frame_ptime = 1.0 / fps
        self._pts: int = 0
        self._last_send_time: float | None = None
        self._last_frame: VideoFrame | None = None
        self._first_frame_emitted = False
        self._frame_processor = frame_processor

    def put_frame(self, frame: VideoFrame) -> None:
        """Enqueue a frame received from cloud (drop-if-full)."""
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    async def recv(self) -> VideoFrame:
        while True:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                if self._last_frame is not None:
                    frame = self._last_frame
                else:
                    await asyncio.sleep(0.01)
                    continue

            if not self._first_frame_emitted:
                self._first_frame_emitted = True
                if self._frame_processor is not None:
                    self._frame_processor.notify_first_video_packet()
                    await asyncio.to_thread(
                        self._frame_processor.wait_for_av_sync_anchor
                    )

            # Pace output
            if self._last_send_time is not None:
                elapsed = time.time() - self._last_send_time
                wait = self.frame_ptime - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
                self._pts += int(self.frame_ptime * VIDEO_CLOCK_RATE)
            self._last_send_time = time.time()

            frame.pts = self._pts
            frame.time_base = VIDEO_TIME_BASE
            self._last_frame = frame
            return frame

    def get_last_frame(self):
        return self._last_frame


class CloudSourceInputHandler:
    """Bridges a browser source track to a cloud input track.

    Reads frames from *browser_track* and pushes them into
    *cloud_input_track* which is on the relay→cloud WebRTC connection.
    """

    def __init__(self, cloud_input_track):
        self.cloud_input_track = cloud_input_track
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self, browser_track: MediaStreamTrack) -> None:
        self._browser_track = browser_track
        self._running = True
        self._task = asyncio.create_task(self._input_loop())

    async def _input_loop(self) -> None:
        consecutive_errors = 0
        max_consecutive_errors = 10
        while self._running:
            try:
                frame = await self._browser_track.recv()
                consecutive_errors = 0
                self.cloud_input_track.put_frame(frame)
            except asyncio.CancelledError:
                break
            except MediaStreamError:
                logger.info("Cloud source input track ended")
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"CloudSourceInputHandler stopping after "
                        f"{consecutive_errors} errors: {e}"
                    )
                    break
                await asyncio.sleep(0.01)
        self._running = False

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
