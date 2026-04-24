from __future__ import annotations

import asyncio
import fractions
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import VideoFrame

from scope.core.pacing import MediaPacingState, compute_pacing_decision

from .media_packets import VideoPacket, ensure_video_packet
from .pipeline_manager import PipelineManager
from .recording import ensure_even_video_frame

if TYPE_CHECKING:
    from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


async def _next_timestamp(
    track: MediaStreamTrack,
    state: dict,
) -> tuple[int, fractions.Fraction]:
    """Pace output at the target frame rate and return a monotonic PTS.

    ``state`` must contain ``fps``, ``_pts``, and ``_last_send_time`` keys.
    """
    if track.readyState != "live":
        raise MediaStreamError

    frame_ptime = 1.0 / state["fps"]
    last = state["_last_send_time"]
    if last is not None:
        wait = frame_ptime - (time.time() - last)
        if wait > 0:
            await asyncio.sleep(wait)
        state["_pts"] += int(frame_ptime * VIDEO_CLOCK_RATE)

    state["_last_send_time"] = time.time()
    return state["_pts"], VIDEO_TIME_BASE


async def _pace_preserved_timestamp(
    track: MediaStreamTrack,
    pacing_state: MediaPacingState,
    packet: VideoPacket,
) -> None:
    """Sleep until a valid packet timestamp lines up with wall clock."""
    if track.readyState != "live":
        raise MediaStreamError
    now_monotonic = time.monotonic()
    media_ts = (
        packet.timestamp.pts * float(packet.timestamp.time_base)
        if packet.timestamp.is_valid
        else None
    )
    decision = compute_pacing_decision(
        pacing_state,
        media_ts=media_ts,
        now_monotonic=now_monotonic,
    )
    if decision.sleep_s > 0:
        await asyncio.sleep(decision.sleep_s)
    if packet.timestamp.is_valid:
        pacing_state.prev_wall_monotonic = time.monotonic()


async def _run_input_loop(
    track: MediaStreamTrack,
    on_frame,
    label: str = "input",
) -> None:
    """Read frames from *track* and pass each to *on_frame*.

    Shared by VideoProcessingTrack.input_loop and SourceInputHandler.
    """
    consecutive_errors = 0
    max_consecutive_errors = 10
    while True:
        try:
            frame = await track.recv()
            consecutive_errors = 0
            on_frame(frame)
        except asyncio.CancelledError:
            break
        except MediaStreamError:
            logger.info(f"Source track ended ({label})")
            break
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                logger.error(
                    f"Input loop ({label}) stopping "
                    f"after {consecutive_errors} errors: {e}"
                )
                break
            await asyncio.sleep(0.01)


class QueueVideoTrack(MediaStreamTrack):
    """Track that reads video frames from a queue (torch.Tensor -> VideoFrame).

    Bridges a graph output queue to the aiortc MediaStreamTrack interface so it
    can be consumed by RecordingManager / MediaRecorder.
    """

    kind = "video"

    def __init__(self, frame_queue: queue.Queue, fps: float = 30.0):
        super().__init__()
        self._queue = frame_queue
        self._ts = {"fps": fps, "_pts": 0, "_last_send_time": None}
        self._pacing = MediaPacingState()

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError

        while True:
            try:
                packet = ensure_video_packet(self._queue.get_nowait())
            except queue.Empty:
                await asyncio.sleep(0.01)
                if self.readyState != "live":
                    raise MediaStreamError from None
                continue

            frame_squeezed = packet.tensor.squeeze(0)
            if frame_squeezed.is_cuda:
                frame_squeezed = frame_squeezed.cpu()

            video_frame = ensure_even_video_frame(
                VideoFrame.from_ndarray(frame_squeezed.numpy(), format="rgb24")
            )
            if packet.timestamp.is_valid:
                await _pace_preserved_timestamp(self, self._pacing, packet)
                video_frame.pts = packet.timestamp.pts
                video_frame.time_base = packet.timestamp.time_base
            else:
                pts, time_base = await _next_timestamp(self, self._ts)
                video_frame.pts = pts
                video_frame.time_base = time_base
            return video_frame


class NodeOutputTrack(MediaStreamTrack):
    """Track that outputs frames from a graph node (sink or record).

    Polls *frame_getter(fp)* for tensors each recv() call.  An optional
    *fps_getter(fp)* updates the playback rate each frame (defaults to 30 fps).
    """

    kind = "video"

    def __init__(
        self,
        frame_processor: FrameProcessor,
        frame_getter,
        fps_getter=None,
    ):
        super().__init__()
        self._frame_processor = frame_processor
        self._frame_getter = frame_getter
        self._fps_getter = fps_getter
        self._ts = {"fps": 30, "_pts": 0, "_last_send_time": None}
        self._pacing = MediaPacingState()

    async def recv(self) -> VideoFrame:
        fp = self._frame_processor

        while True:
            if self._fps_getter is not None:
                self._ts["fps"] = self._fps_getter(fp)

            packet = self._frame_getter(fp)
            if packet is not None:
                packet = ensure_video_packet(packet)
                frame = ensure_even_video_frame(
                    VideoFrame.from_ndarray(packet.tensor.numpy(), format="rgb24")
                )
                if packet.timestamp.is_valid:
                    await _pace_preserved_timestamp(self, self._pacing, packet)
                    frame.pts = packet.timestamp.pts
                    frame.time_base = packet.timestamp.time_base
                else:
                    pts, time_base = await _next_timestamp(self, self._ts)
                    frame.pts = pts
                    frame.time_base = time_base
                return frame

            await asyncio.sleep(0.01)
            if self.readyState != "live":
                raise MediaStreamError


def SinkOutputTrack(
    frame_processor: FrameProcessor,
    sink_node_id: str,
) -> NodeOutputTrack:
    """Create a NodeOutputTrack that reads from a sink node's output queue."""
    return NodeOutputTrack(
        frame_processor=frame_processor,
        frame_getter=lambda fp: fp.get_packet_from_sink(sink_node_id),
        fps_getter=lambda fp: fp.get_fps_for_sink(sink_node_id),
    )


def RecordOutputTrack(
    frame_processor: FrameProcessor,
    record_node_id: str,
) -> NodeOutputTrack:
    """Create a NodeOutputTrack that reads from a record node's output queue."""
    return NodeOutputTrack(
        frame_processor=frame_processor,
        frame_getter=lambda fp: fp.sink_manager.recording.get_packet(record_node_id),
    )


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
        self._task: asyncio.Task | None = None

    def start(self, track: MediaStreamTrack):
        self._task = asyncio.create_task(
            _run_input_loop(
                track,
                lambda frame: self.frame_processor.put_to_source(
                    frame, self.source_node_id
                ),
                label=f"source:{self.source_node_id}",
            )
        )

    async def stop(self):
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
        self.frame_processor = frame_processor
        self.input_task = None
        self.input_task_running = False
        # True when input is handled externally (e.g. all sources via SourceInputHandler)
        self.has_external_input = False
        self._paused = False
        self._paused_lock = threading.Lock()
        self._last_frame = None
        self._frame_lock = threading.Lock()
        self._ts = {"fps": fps, "_pts": 0, "_last_send_time": None}
        self._pacing = MediaPacingState()

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
        """Background loop that continuously feeds frames to the processor."""
        await _run_input_loop(
            self.track,
            self.frame_processor.put,
            label="primary",
        )
        self.input_task_running = False

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
        """Return the next available processed frame."""
        # Lazy initialization on first call
        self.initialize_output_processing()

        # Keep running while any input source is active
        while (
            self.input_task_running
            or self._input_source_enabled
            or self.has_external_input
        ):
            try:
                if self.frame_processor:
                    self._ts["fps"] = self.frame_processor.get_fps()

                # If paused, wait for the appropriate frame interval before returning
                with self._paused_lock:
                    paused = self._paused

                frame = None
                packet: VideoPacket | None = None
                if paused:
                    # When video is paused, return the last frame to freeze the playback video
                    frame = self._last_frame
                else:
                    packet = self.frame_processor.get_packet()

                    if packet is not None:
                        packet = ensure_video_packet(packet)
                        frame = VideoFrame.from_ndarray(
                            packet.tensor.numpy(), format="rgb24"
                        )

                if frame is not None:
                    if packet is not None and packet.timestamp.is_valid:
                        await _pace_preserved_timestamp(self, self._pacing, packet)
                        frame.pts = packet.timestamp.pts
                        frame.time_base = packet.timestamp.time_base
                    else:
                        pts, time_base = await _next_timestamp(self, self._ts)
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
        self.has_external_input = False

        if self.input_task is not None:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        # Note: frame_processor.stop() is handled by Session.close(),
        # not here, because the FrameProcessor is shared with AudioProcessingTrack.

        super().stop()
