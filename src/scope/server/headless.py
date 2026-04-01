"""Headless pipeline session — runs FrameProcessor without WebRTC."""

import asyncio
import logging
import os
import shutil
import tempfile
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)

RECORDING_MAX_FPS = 30.0


class HeadlessRecorder:
    """Records frames to MP4 using PyAV for headless sessions (no WebRTC)."""

    def __init__(self):
        self._container = None
        self._stream = None
        self._recording = False
        self._file_path: str | None = None
        self._frame_count = 0
        self._lock = threading.Lock()
        self._initialized = False

    def start(self):
        """Mark recorder as active. The container is created lazily on the
        first frame so we can read width/height from the actual frame."""
        self._recording = True
        self._initialized = False
        self._frame_count = 0

    def _init_container(self, width: int, height: int):
        """Create the output container and stream from the first frame."""
        import av

        fd, self._file_path = tempfile.mkstemp(suffix=".mp4", prefix="scope_recording_")
        os.close(fd)
        self._container = av.open(self._file_path, "w")
        self._stream = self._container.add_stream(
            "libx264", rate=int(RECORDING_MAX_FPS)
        )
        # libx264 requires even dimensions
        self._stream.width = width + (width % 2)
        self._stream.height = height + (height % 2)
        self._stream.pix_fmt = "yuv420p"
        self._initialized = True
        logger.info(
            "Headless recorder initialized: %dx%d -> %s",
            width,
            height,
            self._file_path,
        )

    def write_frame(self, video_frame):
        """Write a VideoFrame to the recording."""
        if not self._recording:
            return
        import av

        with self._lock:
            if not self._recording:
                return
            arr = video_frame.to_ndarray(format="rgb24")
            h, w = arr.shape[:2]
            if not self._initialized:
                self._init_container(w, h)
            # Pad to even dims if needed
            pad_w = w % 2
            pad_h = h % 2
            if pad_w or pad_h:
                import numpy as np

                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
            self._frame_count += 1

    def stop(self) -> str | None:
        """Stop recording, finalize the MP4, and return the file path."""
        with self._lock:
            self._recording = False
            if self._container is not None:
                try:
                    for packet in self._stream.encode(None):
                        self._container.mux(packet)
                    self._container.close()
                except Exception as e:
                    logger.warning("Error finalizing recording container: %s", e)
                self._container = None
                self._stream = None
            self._initialized = False
            return self._file_path

    @property
    def is_recording(self):
        return self._recording

    @property
    def file_path(self):
        return self._file_path

    @property
    def frame_count(self):
        return self._frame_count


class HeadlessRecordingAdapter:
    """Adapts HeadlessSession recording to the RecordingManager interface
    so the existing ``/api/v1/recordings/{session_id}/*`` endpoints work
    transparently for headless sessions."""

    def __init__(self, session: "HeadlessSession"):
        self._session = session

    @property
    def is_recording_started(self) -> bool:
        return self._session.is_recording

    async def start_recording(self):
        self._session.start_recording()

    async def stop_recording(self):
        self._session.stop_recording()

    async def finalize_and_get_recording(self, restart_after: bool = True):
        return self._session.download_recording()


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
        self._recorder: HeadlessRecorder | None = None
        self._stopped_recording_path: str | None = None
        self.recording_manager = HeadlessRecordingAdapter(self)

    def start_frame_consumer(self):
        """Start a background task that continuously pulls frames to keep the
        pipeline moving and caches the latest one for capture_frame."""
        if self._frame_consumer_running:
            return
        self._frame_consumer_running = True
        self._frame_consumer_task = asyncio.create_task(self._consume_frames())

    async def _consume_frames(self):
        """Pull frames from FrameProcessor so pipeline workers don't stall."""
        from av import VideoFrame

        while self._frame_consumer_running and self.frame_processor.running:
            frame_tensor = self.frame_processor.get()
            if frame_tensor is not None:
                frame_np = frame_tensor.numpy()
                vf = VideoFrame.from_ndarray(frame_np, format="rgb24")
                with self._frame_lock:
                    self._last_frame = vf
                # Write to recorder if active
                if self._recorder and self._recorder.is_recording:
                    self._recorder.write_frame(vf)
                # Yield to event loop so HTTP handlers can run
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0.01)

    def start_recording(self) -> bool:
        """Start recording frames to MP4.

        Returns True if recording was started, False if already recording.
        """
        if self._recorder is not None and self._recorder.is_recording:
            return False
        self._recorder = HeadlessRecorder()
        self._recorder.start()
        logger.info("Headless recording started")
        return True

    def stop_recording(self) -> str | None:
        """Stop recording and return the file path, or None if not recording."""
        if self._recorder is None or not self._recorder.is_recording:
            return None
        file_path = self._recorder.stop()
        frame_count = self._recorder.frame_count
        self._recorder = None
        self._stopped_recording_path = file_path
        logger.info(
            "Headless recording stopped: %d frames, file=%s", frame_count, file_path
        )
        return file_path

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None and self._recorder.is_recording

    def download_recording(self) -> str | None:
        """Stop recording (if active) and return a copy for download.

        Works with both active recordings and previously stopped recordings.
        The file is copied to a download temp file and the original cleaned up.
        """
        # Stop active recording if any
        recording_file = self.stop_recording()
        # Fall back to previously stopped recording
        if not recording_file:
            recording_file = self._stopped_recording_path
        if not recording_file or not os.path.exists(recording_file):
            return None
        # Copy to a download file
        fd, download_path = tempfile.mkstemp(suffix=".mp4", prefix="scope_download_")
        os.close(fd)
        shutil.copy2(recording_file, download_path)
        # Clean up original
        try:
            os.remove(recording_file)
        except Exception as e:
            logger.warning("Failed to remove recording file %s: %s", recording_file, e)
        self._stopped_recording_path = None
        return download_path

    async def close(self):
        """Stop the frame processor and consumer."""
        self._frame_consumer_running = False
        # Stop any active recording
        if self._recorder and self._recorder.is_recording:
            self._recorder.stop()
            self._recorder = None
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
