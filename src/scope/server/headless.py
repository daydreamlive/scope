"""Headless pipeline session — runs FrameProcessor without WebRTC."""

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class HeadlessSession:
    """Pipeline session without WebRTC. Runs FrameProcessor directly."""

    def __init__(
        self,
        frame_processor: "FrameProcessor",
    ):
        from .frame_processor import FrameProcessor

        self.frame_processor: FrameProcessor = frame_processor
        self._last_frame = None
        self._frame_lock = threading.Lock()
        self._frame_consumer_running = False
        self._frame_consumer_task: asyncio.Task | None = None

    def start_frame_consumer(self):
        """Start a background task that continuously pulls frames to keep the
        pipeline moving and caches the latest one for capture_frame."""
        if self._frame_consumer_running:
            return
        self._frame_consumer_running = True
        self._frame_consumer_task = asyncio.create_task(self._consume_frames())

    async def _consume_frames(self):
        """Pull frames from FrameProcessor so pipeline workers don't stall.

        Only drains sink queues that are NOT already consumed by per-node
        output sink threads (Syphon/NDI/Spout).  This avoids competing
        for frames on the same queue.
        """
        from av import VideoFrame

        fp = self.frame_processor

        while self._frame_consumer_running and fp.running:
            # Determine which sinks still need draining
            unhandled = fp.get_unhandled_sink_node_ids()

            if unhandled:
                # Drain each unhandled sink queue
                got_frame = False
                for sink_id in unhandled:
                    frame_tensor = fp.get_from_sink(sink_id)
                    if frame_tensor is not None:
                        got_frame = True
                        frame_np = frame_tensor.numpy()
                        with self._frame_lock:
                            self._last_frame = VideoFrame.from_ndarray(
                                frame_np, format="rgb24"
                            )
                if not got_frame:
                    await asyncio.sleep(0.01)
            elif fp.get_sink_node_ids():
                # All sinks have output threads — nothing to drain.
                # Just sleep; capture_frame can snapshot from output sinks.
                await asyncio.sleep(0.05)
            else:
                # No multi-sink graph — use legacy get()
                frame_tensor = fp.get()
                if frame_tensor is not None:
                    frame_np = frame_tensor.numpy()
                    with self._frame_lock:
                        self._last_frame = VideoFrame.from_ndarray(
                            frame_np, format="rgb24"
                        )
                else:
                    await asyncio.sleep(0.01)

    async def close(self):
        """Stop the frame processor and consumer."""
        self._frame_consumer_running = False
        if self._frame_consumer_task is not None:
            self._frame_consumer_task.cancel()
            try:
                await self._frame_consumer_task
            except asyncio.CancelledError:
                pass
        self.frame_processor.stop()
        logger.info("Headless session closed")

    def get_last_frame(self):
        """Return the most recently cached frame, or None."""
        with self._frame_lock:
            return self._last_frame

    def __str__(self):
        return f"HeadlessSession(running={self.frame_processor.running})"
