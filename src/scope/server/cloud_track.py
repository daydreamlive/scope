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

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import VideoFrame

if TYPE_CHECKING:
    from .cloud_connection import CloudConnectionManager

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
        cloud_manager: CloudConnectionManager,
        fps: int = 30,
        initial_parameters: dict | None = None,
        notification_callback: Callable | None = None,
        user_id: str | None = None,
    ):
        super().__init__()
        self.cloud_manager = cloud_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        self.user_id = user_id

        # FPS control
        self.fps = fps
        self.frame_ptime = 1.0 / fps

        # Source track for input frames (from browser)
        self._source_track: MediaStreamTrack | None = None
        self._input_task: asyncio.Task | None = None
        self._input_running = False

        # FrameProcessor handles relay to cloud and Spout integration
        self.frame_processor = None
        self._last_frame: VideoFrame | None = None
        self._started = False

    def set_source_track(self, track: MediaStreamTrack) -> None:
        """Set the source track for input frames (from browser)."""
        self._source_track = track
        logger.info("[CLOUD] Source track set")

    async def _start(self) -> None:
        """Start the relay - called on first recv()."""
        if self._started:
            return

        self._started = True
        logger.info("[CLOUD] Starting cloud relay...")

        # Start WebRTC connection to cloud with this session's parameters
        logger.info("[CLOUD] Starting WebRTC connection to cloud...")
        await self.cloud_manager.start_webrtc(self.initial_parameters)

        # Create FrameProcessor in cloud mode
        from .frame_processor import FrameProcessor

        self.frame_processor = FrameProcessor(
            pipeline_manager=None,  # Not needed in cloud mode
            initial_parameters=self.initial_parameters,
            notification_callback=self.notification_callback,
            cloud_manager=self.cloud_manager,  # Enable cloud mode
            user_id=self.user_id,
        )
        self.frame_processor.start()

        # Start input processing if we have a source track
        if self._source_track is not None:
            self._input_running = True
            self._input_task = asyncio.create_task(self._input_loop())

        logger.info("[CLOUD] Relay started")

    async def _input_loop(self) -> None:
        """Background loop that receives frames from source and sends to cloud."""
        logger.info("[CLOUD] Input loop started")

        try:
            while self._input_running and self._source_track is not None:
                try:
                    # Get frame from browser
                    frame = await self._source_track.recv()

                    # Send through FrameProcessor (which relays to cloud)
                    if self.frame_processor:
                        self.frame_processor.put(frame)

                except MediaStreamError:
                    logger.info("[CLOUD] Source track ended")
                    break
                except Exception as e:
                    logger.error(f"[CLOUD] Error in input loop: {e}")
                    break

        except asyncio.CancelledError:
            pass
        finally:
            self._input_running = False
            stats = (
                self.frame_processor.get_frame_stats() if self.frame_processor else {}
            )
            logger.info(f"[CLOUD] Input loop ended, stats: {stats}")

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

        # Wait for a processed frame from FrameProcessor
        max_wait = 1.0  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if self.frame_processor:
                frame_tensor = self.frame_processor.get()
                if frame_tensor is not None:
                    # Convert tensor to VideoFrame
                    frame_np = frame_tensor.numpy()
                    frame = VideoFrame.from_ndarray(frame_np, format="rgb24")

                    pts, time_base = await self.next_timestamp()
                    frame.pts = pts
                    frame.time_base = time_base

                    self._last_frame = frame
                    return frame

            # No frame yet, wait a bit
            await asyncio.sleep(0.01)

        # No frame received, return last frame or black frame
        if self._last_frame is not None:
            pts, time_base = await self.next_timestamp()
            self._last_frame.pts = pts
            return self._last_frame

        # Return black frame as fallback
        black = np.zeros((512, 512, 3), dtype=np.uint8)
        frame = VideoFrame.from_ndarray(black, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def update_parameters(self, params: dict) -> None:
        """Update pipeline parameters on cloud."""
        # Handle Spout settings via FrameProcessor
        if self.frame_processor:
            self.frame_processor.update_parameters(params)

        # Also send to cloud
        self.cloud_manager.send_parameters_to_fal(params)

    def pause(self, paused: bool) -> None:
        """Pause/unpause the relay."""
        if self.frame_processor:
            self.frame_processor.paused = paused
        logger.info(f"[CLOUD] {'Paused' if paused else 'Resumed'}")

    async def stop(self) -> None:
        """Stop the relay and clean up."""
        logger.info("[CLOUD] Stopping...")

        self._input_running = False
        self._started = False  # Reset so next session starts fresh

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
            logger.info(f"[CLOUD] Stopped. Stats: {stats}")
            self.frame_processor = None
        else:
            logger.info("[CLOUD] Stopped.")

        # Stop WebRTC connection to cloud - next session will start fresh
        await self.cloud_manager.stop_webrtc()

        # Call parent stop() synchronously - MediaStreamTrack.stop() is not async
        super().stop()

    def get_stats(self) -> dict:
        """Get relay statistics."""
        if self.frame_processor:
            return self.frame_processor.get_frame_stats()
        return {}
