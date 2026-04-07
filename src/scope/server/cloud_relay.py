"""CloudRelay — manages the cloud frame/audio relay path.

Owns the output queues, frame counters, and send/receive logic for cloud
mode. FrameProcessor holds an optional CloudRelay instead of scattering
cloud-specific state across its own fields.
"""

import logging
import queue
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from aiortc.mediastreams import VideoFrame
    from av import AudioFrame

    from .cloud_connection import CloudConnectionManager

logger = logging.getLogger(__name__)


def compute_relay_video_mode(initial_parameters: dict | None) -> bool:
    """Return whether ``CloudRelay`` should forward frames to the cloud runner.

    ``input_mode == "video"`` is the primary signal for classic (non-graph)
    sessions. Graph workflows with at least one Source node always have video
    input to relay. Without this, :class:`CloudRelay` drops every frame when
    ``video_mode`` is false (e.g. Syphon works locally but not in cloud).
    """
    params = initial_parameters or {}
    if params.get("input_mode") == "video":
        return True
    graph_data = params.get("graph")
    if graph_data and isinstance(graph_data, dict):
        for node in graph_data.get("nodes", []):
            if node.get("type") == "source":
                return True
    return False


class CloudRelay:
    """Relay frames to/from a cloud pipeline instance.

    Provides:
    - send_frame / send_frame_to_source: push input frames to the cloud
    - on_frame_from_cloud / on_audio_from_cloud: callbacks for received output
    - get_frame / get_audio: consume received output for WebRTC delivery
    - Frame counters (frames_to_cloud, frames_from_cloud)
    """

    def __init__(
        self,
        cloud_manager: "CloudConnectionManager",
        video_mode: bool = False,
    ):
        self._cloud_manager = cloud_manager
        self._video_mode = video_mode

        # Output queues populated by cloud callbacks
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._audio_queue: queue.Queue = queue.Queue(maxsize=50)

        # Counters
        self.frames_to_cloud = 0
        self.frames_from_cloud = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def video_mode(self) -> bool:
        return self._video_mode

    @video_mode.setter
    def video_mode(self, value: bool) -> None:
        self._video_mode = value

    # ------------------------------------------------------------------
    # Sending frames to cloud
    # ------------------------------------------------------------------

    def send_frame(self, rgb_frame: np.ndarray) -> bool:
        """Send a frame to the cloud (generic / track 0)."""
        if not self._video_mode:
            if self.frames_to_cloud == 0:
                logger.warning(
                    "[CLOUD-RELAY] Dropping frame: video_mode=False "
                    "(Syphon/graph source detected but relay not enabled)"
                )
            return False
        sent = self._cloud_manager.send_frame(rgb_frame)
        if sent:
            self.frames_to_cloud += 1
            if self.frames_to_cloud == 1:
                logger.info("[CLOUD-RELAY] First frame sent to cloud (generic/track 0)")
        return sent

    def send_frame_to_source(self, rgb_frame: np.ndarray, source_node_id: str) -> bool:
        """Send a frame to a specific cloud source track."""
        if not self._video_mode:
            if self.frames_to_cloud == 0:
                logger.warning(
                    "[CLOUD-RELAY] Dropping frame for source %s: video_mode=False",
                    source_node_id,
                )
            return False
        # Try multi-track API if available, otherwise fall back to generic send
        get_idx = getattr(self._cloud_manager, "get_source_track_index", None)
        track_idx = get_idx(source_node_id) if get_idx is not None else None
        if track_idx is not None:
            sent = self._cloud_manager.send_frame_to_track(rgb_frame, track_idx)
        else:
            sent = self._cloud_manager.send_frame(rgb_frame)
        if sent:
            self.frames_to_cloud += 1
            if self.frames_to_cloud == 1:
                logger.info(
                    "[CLOUD-RELAY] First frame sent to cloud for source %s "
                    "(track_idx=%s)",
                    source_node_id,
                    track_idx,
                )
        return sent

    # ------------------------------------------------------------------
    # Receiving frames/audio from cloud (callbacks)
    # ------------------------------------------------------------------

    def on_frame_from_cloud(self, frame: "VideoFrame") -> None:
        """Callback when a processed video frame is received from cloud."""
        self.frames_from_cloud += 1
        if self.frames_from_cloud == 1:
            logger.info("[CLOUD-RELAY] First frame received from cloud")
        try:
            frame_np = frame.to_ndarray(format="rgb24")
            try:
                self._frame_queue.put_nowait(frame_np)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(frame_np)
                except queue.Empty:
                    pass
        except Exception as e:
            logger.error(f"Error processing frame from cloud: {e}")

    def on_audio_from_cloud(self, frame: "AudioFrame") -> None:
        """Callback when an audio frame is received from cloud.

        Converts the AudioFrame to a torch tensor and queues it.
        Packed formats (s16) store interleaved channels in a single plane,
        so to_ndarray() returns (1, samples*channels).  We de-interleave
        into (channels, samples) so AudioProcessingTrack sees the correct
        channel count and doesn't erroneously duplicate data.
        """
        try:
            n_channels = len(frame.layout.channels)
            audio_np = frame.to_ndarray()
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)

            # Packed formats (e.g. s16) have 1 plane with interleaved channels:
            # [L0, R0, L1, R1, ...].  De-interleave into (channels, samples).
            if audio_np.shape[0] == 1 and n_channels > 1:
                flat = audio_np.ravel()
                audio_np = flat.reshape(-1, n_channels).T

            audio_tensor = torch.from_numpy(audio_np.astype(np.float32))

            # Normalise int16 range to [-1, 1] float if needed
            if frame.format.name in ("s16", "s16p"):
                audio_tensor = audio_tensor / 32768.0

            try:
                self._audio_queue.put_nowait((audio_tensor, frame.sample_rate))
            except queue.Full:
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait((audio_tensor, frame.sample_rate))
                except queue.Empty:
                    pass
        except Exception as e:
            logger.error(f"Error processing audio from cloud: {e}")

    # ------------------------------------------------------------------
    # Consuming received output
    # ------------------------------------------------------------------

    def get_frame(self) -> torch.Tensor | None:
        """Get the next video frame received from cloud, or None."""
        try:
            frame_np = self._frame_queue.get_nowait()
            return torch.from_numpy(frame_np)
        except queue.Empty:
            return None

    def get_audio(self) -> tuple[torch.Tensor | None, int | None]:
        """Get the next audio chunk and sample rate, or (None, None)."""
        try:
            return self._audio_queue.get_nowait()
        except queue.Empty:
            return None, None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register callbacks on the cloud manager."""
        self._cloud_manager.add_frame_callback(self.on_frame_from_cloud)
        self._cloud_manager.add_audio_callback(self.on_audio_from_cloud)

    def stop(self) -> None:
        """Unregister callbacks from the cloud manager."""
        self._cloud_manager.remove_frame_callback(self.on_frame_from_cloud)
        self._cloud_manager.remove_audio_callback(self.on_audio_from_cloud)
