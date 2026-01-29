"""Custom MediaStreamTrack classes for fal.ai WebRTC communication.

FalOutputTrack: Sends frames from a queue to fal.ai via WebRTC
FalInputTrack: Receives processed frames from fal.ai and queues them
"""

from __future__ import annotations

import asyncio
import fractions
import time
from typing import TYPE_CHECKING

from aiortc.mediastreams import MediaStreamTrack

if TYPE_CHECKING:
    from av import VideoFrame


class FalOutputTrack(MediaStreamTrack):
    """Sends frames from queue to fal via WebRTC.

    This is the outbound track - frames are put into the queue
    and sent to fal.ai for processing.
    """

    kind = "video"

    def __init__(self, target_fps: int = 30):
        super().__init__()
        self.frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=30)
        self.target_fps = target_fps
        self._start_time = time.time()
        self._frame_count = 0

    async def recv(self) -> VideoFrame:
        """Called by aiortc to get next frame to send.

        This method is called by the WebRTC stack when it needs
        the next frame to encode and send.
        """
        frame = await self.frame_queue.get()

        # Set pts (presentation timestamp) and time_base
        self._frame_count += 1
        frame.pts = self._frame_count
        frame.time_base = fractions.Fraction(1, self.target_fps)

        return frame

    async def put_frame(self, frame: VideoFrame) -> bool:
        """Add frame to be sent to fal.

        Returns True if frame was queued, False if queue was full (frame dropped).
        """
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            # Drop oldest frame and add new one
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
                return True
            except asyncio.QueueEmpty:
                return False

    def put_frame_sync(self, frame: VideoFrame) -> bool:
        """Synchronous version for use from non-async contexts."""
        return self.put_frame_nowait(frame)

    def put_frame_nowait(self, frame: VideoFrame) -> bool:
        """Non-blocking frame put."""
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            return False


class FalInputTrack(MediaStreamTrack):
    """Receives processed frames from fal via WebRTC.

    This wraps an incoming track and makes frames available via a queue.
    Similar pattern to YOLOTrack in reference, but stores frames instead
    of processing them.
    """

    kind = "video"

    def __init__(self, source_track: MediaStreamTrack):
        super().__init__()
        self.source_track = source_track
        self.frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=30)
        self._consume_task: asyncio.Task | None = None

    def start_consuming(self) -> None:
        """Start consuming frames from source track."""
        self._consume_task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self) -> None:
        """Continuously receive frames from source and queue them."""
        while True:
            try:
                frame = await self.source_track.recv()
                try:
                    self.frame_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    # Drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except asyncio.QueueEmpty:
                        pass
            except Exception:
                break

    async def recv(self) -> VideoFrame:
        """Get next received frame."""
        return await self.frame_queue.get()

    def get_frame_nowait(self) -> VideoFrame | None:
        """Non-blocking frame get."""
        try:
            return self.frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop(self) -> None:
        """Stop consuming frames."""
        if self._consume_task:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
