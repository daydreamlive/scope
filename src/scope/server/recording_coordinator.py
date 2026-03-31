"""RecordingCoordinator — owns per-record-node queues and recording managers.

Extracted from FrameProcessor to give recording its own single-responsibility
class.  FrameProcessor delegates all recording-related calls here.
"""

import logging
import queue
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


class RecordingCoordinator:
    """Manages per-record-node frame queues and RecordingManager instances."""

    def __init__(self, get_fps: Callable[[], float]):
        self._get_fps = get_fps
        self.record_queues: dict[str, queue.Queue] = {}
        self._managers: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup_queues(self, record_node_ids: list[str], maxsize: int = 30) -> None:
        """Create a frame queue for each record node."""
        for rec_id in record_node_ids:
            self.record_queues[rec_id] = queue.Queue(maxsize=maxsize)
        if self.record_queues:
            logger.info(f"Created record queues for {list(self.record_queues.keys())}")

    def cleanup(self) -> None:
        """Stop all active recordings and clear state."""
        for node_id, entry in list(self._managers.items()):
            try:
                entry["track"].stop()
            except Exception as e:
                logger.error(f"Error stopping record track for node {node_id}: {e}")
        self._managers.clear()
        self.record_queues.clear()

    # ------------------------------------------------------------------
    # Frame routing
    # ------------------------------------------------------------------

    def get_node_ids(self) -> list[str]:
        """Return the list of record node IDs."""
        return list(self.record_queues.keys())

    def put(self, record_node_id: str, frame) -> None:
        """Put a VideoFrame into a record node's queue.

        Called by cloud output callbacks to populate record queues with
        frames received from the cloud pipeline.
        """
        rec_q = self.record_queues.get(record_node_id)
        if rec_q is None:
            return
        try:
            frame_np = frame.to_ndarray(format="rgb24")
            frame_tensor = torch.as_tensor(frame_np, dtype=torch.uint8).unsqueeze(0)
            try:
                rec_q.put_nowait(frame_tensor)
            except queue.Full:
                try:
                    rec_q.get_nowait()
                    rec_q.put_nowait(frame_tensor)
                except queue.Empty:
                    pass
        except Exception as e:
            logger.error(f"Error putting frame to record node {record_node_id}: {e}")

    def get(self, record_node_id: str) -> torch.Tensor | None:
        """Read a frame from a record node's output queue."""
        rec_q = self.record_queues.get(record_node_id)
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

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    async def start_recording(self, node_id: str) -> bool:
        """Start recording for a specific record node."""
        rec_q = self.record_queues.get(node_id)
        if rec_q is None:
            logger.error(f"No record queue for node {node_id}")
            return False

        if node_id in self._managers:
            entry = self._managers[node_id]
            if entry["manager"].is_recording_started:
                logger.info(f"Record node {node_id} already recording")
                return True

        from .recording import RecordingManager
        from .tracks import QueueVideoTrack

        track = QueueVideoTrack(rec_q, fps=self._get_fps())
        manager = RecordingManager(video_track=track)

        self._managers[node_id] = {
            "manager": manager,
            "track": track,
        }

        await manager.start_recording()
        logger.info(f"Started recording for record node {node_id}")
        return True

    async def stop_recording(self, node_id: str) -> bool:
        """Stop recording for a specific record node."""
        entry = self._managers.get(node_id)
        if entry is None:
            return False
        await entry["manager"].stop_recording()
        logger.info(f"Stopped recording for record node {node_id}")
        return True

    async def download_recording(self, node_id: str) -> str | None:
        """Finalize and return the recording file path for a record node."""
        entry = self._managers.get(node_id)
        if entry is None:
            return None
        path = await entry["manager"].finalize_and_get_recording(restart_after=False)
        self._managers.pop(node_id, None)
        return path
