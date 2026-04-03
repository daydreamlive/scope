"""RecordingCoordinator — owns per-record-node queues and recording managers.

Extracted from FrameProcessor to give recording its own single-responsibility
class.  FrameProcessor delegates all recording-related calls here.
"""

import logging
import queue
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class _RecordingEntry:
    manager: object  # RecordingManager
    track: object  # QueueVideoTrack


class RecordingCoordinator:
    """Manages per-record-node frame queues and RecordingManager instances."""

    def __init__(self):
        self._record_queues: dict[str, queue.Queue] = {}
        self._entries: dict[str, _RecordingEntry] = {}

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup_queues(
        self,
        record_queues: dict[str, queue.Queue] | list[str],
        maxsize: int = 30,
    ) -> None:
        """Set up record queues.

        Args:
            record_queues: Either a pre-built dict of node_id -> Queue
                (from graph executor) or a list of node IDs to create
                queues for (cloud mode).
            maxsize: Queue size when creating from a list.
        """
        if isinstance(record_queues, dict):
            self._record_queues = record_queues
        else:
            self._record_queues = {
                rec_id: queue.Queue(maxsize=maxsize) for rec_id in record_queues
            }
        if self._record_queues:
            logger.info(f"Created record queues for {list(self._record_queues.keys())}")

    def cleanup(self) -> None:
        """Stop all active recordings and clear state."""
        for node_id, entry in list(self._entries.items()):
            try:
                entry.track.stop()
            except Exception as e:
                logger.error(f"Error stopping record track for node {node_id}: {e}")
        self._entries.clear()
        self._record_queues.clear()

    # ------------------------------------------------------------------
    # Frame routing
    # ------------------------------------------------------------------

    def get_node_ids(self) -> list[str]:
        """Return the list of record node IDs."""
        return list(self._record_queues.keys())

    def get(self, record_node_id: str) -> torch.Tensor | None:
        """Read a frame from a record node's output queue."""
        rec_q = self._record_queues.get(record_node_id)
        if rec_q is None:
            return None
        try:
            frame = rec_q.get_nowait()
            frame = frame.squeeze(0)
            if frame.is_cuda:
                frame = frame.cpu()
            return frame
        except queue.Empty:
            return None

    def put(self, record_node_id: str, frame: torch.Tensor) -> bool:
        """Write a frame into a record node's queue (cloud mode).

        Returns True if the frame was enqueued, False if the queue is
        missing or full.
        """
        rec_q = self._record_queues.get(record_node_id)
        if rec_q is None:
            return False
        try:
            rec_q.put_nowait(frame)
            return True
        except queue.Full:
            return False

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    async def start_recording(self, node_id: str, fps: float) -> bool:
        """Start recording for a specific record node.

        Args:
            node_id: The record node to start recording for.
            fps: The current playback FPS for the recording.
        """
        rec_q = self._record_queues.get(node_id)
        if rec_q is None:
            logger.error(f"No record queue for node {node_id}")
            return False

        if node_id in self._entries:
            if self._entries[node_id].manager.is_recording_started:
                logger.info(f"Record node {node_id} already recording")
                return True

        from .recording import RecordingManager
        from .tracks import QueueVideoTrack

        track = QueueVideoTrack(rec_q, fps=fps)
        manager = RecordingManager(video_track=track)

        self._entries[node_id] = _RecordingEntry(manager=manager, track=track)

        await manager.start_recording()
        logger.info(f"Started recording for record node {node_id}")
        return True

    async def stop_recording(self, node_id: str) -> bool:
        """Stop recording for a specific record node."""
        entry = self._entries.get(node_id)
        if entry is None:
            return False
        await entry.manager.stop_recording()
        logger.info(f"Stopped recording for record node {node_id}")
        return True

    async def download_recording(self, node_id: str) -> str | None:
        """Finalize and return the recording file path for a record node."""
        entry = self._entries.get(node_id)
        if entry is None:
            return None
        path = await entry.manager.finalize_and_get_recording(restart_after=False)
        self._entries.pop(node_id, None)
        return path
