from __future__ import annotations

import asyncio
import fractions
import logging
import threading
import time
from typing import TYPE_CHECKING

from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import VideoFrame

from .pipeline_manager import PipelineManager

if TYPE_CHECKING:
    from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class SinkOutputTrack(MediaStreamTrack):
    """Track that outputs frames from a specific sink node in a multi-sink graph.

    This is a lightweight wrapper: it waits for the primary VideoProcessingTrack's
    FrameProcessor to be ready, then reads frames from a specific sink node's
    output queue.
    """

    kind = "video"

    def __init__(
        self,
        primary_track: VideoProcessingTrack,
        sink_node_id: str,
    ):
        super().__init__()
        self.primary_track = primary_track
        self.sink_node_id = sink_node_id
        self.fps = 30
        self.frame_ptime = 1.0 / self.fps
        self._paused = False
        self._paused_lock = threading.Lock()
        self._last_frame = None
        self._last_send_time: float | None = None
        self._pts: int = 0
        self._frame_lock = threading.Lock()

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError

        if self._last_send_time is not None:
            elapsed = time.time() - self._last_send_time
            wait = self.frame_ptime - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._pts += int(self.frame_ptime * VIDEO_CLOCK_RATE)

        self._last_send_time = time.time()
        return self._pts, VIDEO_TIME_BASE

    async def recv(self) -> VideoFrame:
        # Wait for primary track's frame_processor to be ready
        while self.primary_track.frame_processor is None:
            await asyncio.sleep(0.05)

        fp = self.primary_track.frame_processor

        while True:
            with self._paused_lock:
                paused = self._paused

            frame = None
            if paused:
                frame = self._last_frame
            else:
                # Update FPS from the specific sink's feeder processor
                if fp:
                    self.fps = fp.get_fps_for_sink(self.sink_node_id)
                    self.frame_ptime = 1.0 / self.fps

                frame_tensor = fp.get_from_sink(self.sink_node_id)
                if frame_tensor is not None:
                    frame = VideoFrame.from_ndarray(
                        frame_tensor.numpy(), format="rgb24"
                    )

            if frame is not None:
                pts, time_base = await self.next_timestamp()
                frame.pts = pts
                frame.time_base = time_base
                with self._frame_lock:
                    self._last_frame = frame
                return frame

            await asyncio.sleep(0.01)

    def get_last_frame(self):
        with self._frame_lock:
            return self._last_frame

    def pause(self, paused: bool):
        with self._paused_lock:
            self._paused = paused


class SourceInputHandler:
    """Handles input from a WebRTC track and routes frames to a specific source node.

    Used in multi-source mode: each WebRTC video track from the browser is
    routed to a specific source node in the graph via put_to_source().
    """

    def __init__(
        self,
        frame_processor: FrameProcessor,
        source_node_id: str,
    ):
        self.frame_processor = frame_processor
        self.source_node_id = source_node_id
        self._running = False
        self._task: asyncio.Task | None = None
        self._track: MediaStreamTrack | None = None

    def start(self, track: MediaStreamTrack):
        self._track = track
        self._running = True
        self._task = asyncio.create_task(self._input_loop())

    async def _input_loop(self):
        consecutive_errors = 0
        max_consecutive_errors = 10
        while self._running:
            try:
                input_frame = await self._track.recv()
                consecutive_errors = 0
                self.frame_processor.put_to_source(input_frame, self.source_node_id)
            except asyncio.CancelledError:
                break
            except MediaStreamError:
                logger.info(f"Source track ended for node {self.source_node_id}")
                self._running = False
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"Input handler for {self.source_node_id} stopping "
                        f"after {consecutive_errors} errors: {e}"
                    )
                    self._running = False
                    break
                await asyncio.sleep(0.01)

    async def stop(self):
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class VideoProcessingTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        fps: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
        session_id: str | None = None,
        user_id: str | None = None,
        connection_id: str | None = None,
        connection_info: dict | None = None,
        tempo_sync=None,
        frame_processor: FrameProcessor | None = None,
        sink_node_id: str | None = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        self.session_id = session_id
        self.user_id = user_id
        self.connection_id = connection_id
        self.connection_info = connection_info
        self.tempo_sync = tempo_sync
        # When set, this track reads from a specific sink node's queue
        self.sink_node_id = sink_node_id
        # FPS variables (will be updated from FrameProcessor or input measurement)
        self.fps = fps
        self.frame_ptime = 1.0 / fps

        self.frame_processor = frame_processor
        self.input_task = None
        self.input_task_running = False
        # True when input is handled externally (e.g. all sources via SourceInputHandler)
        self._has_external_input = False
        self._paused = False
        self._paused_lock = threading.Lock()
        self._last_frame = None
        self._last_send_time: float | None = None
        self._pts: int = 0
        self._frame_lock = threading.Lock()

        # Server-side input mode - when enabled, frames come from the backend
        # instead of WebRTC (no browser video track needed)
        self._input_source_enabled = False
        if initial_parameters:
            input_source = initial_parameters.get("input_source")
            if input_source and input_source.get("enabled"):
                self._input_source_enabled = True
                logger.info(
                    f"Input source mode enabled: {input_source.get('source_type')}"
                )

    async def input_loop(self):
        """Background loop that continuously feeds frames to the processor"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        while self.input_task_running:
            try:
                input_frame = await self.track.recv()
                consecutive_errors = 0

                # Store raw VideoFrame for later processing (tracks input FPS internally)
                self.frame_processor.put(input_frame)

            except asyncio.CancelledError:
                break
            except MediaStreamError:
                logger.info("Source track ended")
                self.input_task_running = False
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"Error in input loop, stopping after "
                        f"{consecutive_errors} consecutive errors: {e}"
                    )
                    self.input_task_running = False
                    break
                logger.warning(
                    f"Transient error in input loop "
                    f"({consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                await asyncio.sleep(0.01)

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Pace output at the target frame rate and return a monotonic PTS."""
        if self.readyState != "live":
            raise MediaStreamError

        # Pace frames at the target interval
        if self._last_send_time is not None:
            elapsed = time.time() - self._last_send_time
            wait = self.frame_ptime - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._pts += int(self.frame_ptime * VIDEO_CLOCK_RATE)

        self._last_send_time = time.time()

        return self._pts, VIDEO_TIME_BASE

    def initialize_output_processing(self):
        """No-op guard; FrameProcessor is injected via constructor."""
        if not self.frame_processor:
            raise RuntimeError(
                "VideoProcessingTrack requires a FrameProcessor. "
                "Pass one via the constructor."
            )

    def initialize_input_processing(self, track: MediaStreamTrack):
        self.track = track
        self.input_task_running = True
        self.input_task = asyncio.create_task(self.input_loop())

    async def recv(self) -> VideoFrame:
        """Return the next available processed frame"""
        # Lazy initialization on first call
        self.initialize_output_processing()

        # Keep running while any input source is active
        while (
            self.input_task_running
            or self._input_source_enabled
            or self._has_external_input
        ):
            try:
                # Update FPS from the specific sink's feeder processor, or global
                if self.frame_processor:
                    if self.sink_node_id:
                        self.fps = self.frame_processor.get_fps_for_sink(
                            self.sink_node_id
                        )
                    else:
                        self.fps = self.frame_processor.get_fps()
                    self.frame_ptime = 1.0 / self.fps

                # If paused, wait for the appropriate frame interval before returning
                with self._paused_lock:
                    paused = self._paused

                frame = None
                if paused:
                    # When video is paused, return the last frame to freeze the playback video
                    frame = self._last_frame
                else:
                    # Read from specific sink node or fall back to legacy get()
                    if self.sink_node_id:
                        frame_tensor = self.frame_processor.get_from_sink(
                            self.sink_node_id
                        )
                        if frame_tensor is not None:
                            self.frame_processor.notify_primary_frame_output(
                                frame_tensor
                            )
                    else:
                        frame_tensor = self.frame_processor.get()

                    if frame_tensor is not None:
                        frame = VideoFrame.from_ndarray(
                            frame_tensor.numpy(), format="rgb24"
                        )

                if frame is not None:
                    pts, time_base = await self.next_timestamp()
                    frame.pts = pts
                    frame.time_base = time_base

                    with self._frame_lock:
                        self._last_frame = frame
                    return frame

                # No frame available, wait a bit before trying again
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error getting processed frame: {e}")
                raise

        raise Exception("Track stopped")

    def get_last_frame(self):
        """Return the most recently rendered frame, or None."""
        with self._frame_lock:
            return self._last_frame

    def pause(self, paused: bool):
        """Pause or resume the video track processing"""
        with self._paused_lock:
            self._paused = paused

        # Propagate to frame_processor so AudioProcessingTrack can check it
        if self.frame_processor:
            self.frame_processor.paused = paused

        logger.info(f"Video track {'paused' if paused else 'resumed'}")

    async def stop(self):
        self.input_task_running = False
        self._input_source_enabled = False
        self._has_external_input = False

        if self.input_task is not None:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        # Note: frame_processor.stop() is handled by Session.close(),
        # not here, because the FrameProcessor is shared with AudioProcessingTrack.

        super().stop()
