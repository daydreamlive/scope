"""Headless pipeline session — runs FrameProcessor without WebRTC."""

import asyncio
import fractions
import logging
import os
import shutil
import tempfile
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)

RECORDING_MAX_FPS = 30.0
AUDIO_CLOCK_RATE = 48_000


class HeadlessMediaSink:
    """Base sink consumed by HeadlessSession fanout."""

    def on_video_frame(self, video_frame) -> None:
        raise NotImplementedError

    def on_audio_chunk(
        self, audio_tensor: "torch.Tensor | None", sample_rate: int | None
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _TsStreamBuffer:
    """Minimal PyAV output bridge backed by an async byte queue."""

    def __init__(self):
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._closed = False

    def write(self, data):
        if self._closed:
            return 0
        chunk = bytes(data)
        if chunk:
            self._queue.put_nowait(chunk)
        return len(chunk)

    def flush(self):
        return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(None)

    async def iter_chunks(self):
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk


class HeadlessTsStreamer(HeadlessMediaSink):
    """Streams headless output as MPEG-TS using PyAV."""

    def __init__(self, expect_audio: bool):
        self._expect_audio = expect_audio
        self._buffer = _TsStreamBuffer()
        self._container = None
        self._video_stream = None
        self._audio_stream = None
        self._audio_resampler = None
        self._initialized = False
        self._closed = False
        self._lock = threading.Lock()
        self._audio_samples_written = 0

    def _init_container(self, width: int, height: int):
        import av

        self._container = av.open(self._buffer, "w", format="mpegts")
        self._video_stream = self._container.add_stream(
            "libx264", rate=int(RECORDING_MAX_FPS)
        )
        self._video_stream.width = width + (width % 2)
        self._video_stream.height = height + (height % 2)
        self._video_stream.pix_fmt = "yuv420p"
        if self._expect_audio:
            self._audio_stream = self._container.add_stream(
                "aac", rate=AUDIO_CLOCK_RATE
            )
            self._audio_stream.layout = "stereo"
        self._initialized = True

    def on_video_frame(self, video_frame) -> None:
        if self._closed:
            return
        import av

        with self._lock:
            if self._closed:
                return
            arr = video_frame.to_ndarray(format="rgb24")
            h, w = arr.shape[:2]
            if not self._initialized:
                self._init_container(w, h)
            pad_w = w % 2
            pad_h = h % 2
            if pad_w or pad_h:
                import numpy as np

                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in self._video_stream.encode(frame):
                self._container.mux(packet)

    def on_audio_chunk(
        self, audio_tensor: "torch.Tensor | None", sample_rate: int | None
    ) -> None:
        if self._closed or not self._expect_audio:
            return
        if audio_tensor is None:
            return
        if sample_rate is None or sample_rate <= 0:
            return

        import av
        import numpy as np

        with self._lock:
            if self._closed or not self._initialized or self._audio_stream is None:
                return
            audio_np = audio_tensor.numpy()
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            if audio_np.shape[0] == 1:
                audio_np = np.vstack([audio_np, audio_np])
            elif audio_np.shape[0] > 2:
                audio_np = audio_np[:2]
            audio_np = np.asarray(audio_np, dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(audio_np, format="fltp", layout="stereo")
            frame.sample_rate = int(sample_rate)
            if self._audio_resampler is None:
                self._audio_resampler = av.audio.resampler.AudioResampler(
                    format="fltp",
                    layout="stereo",
                    rate=AUDIO_CLOCK_RATE,
                )
            resampled = self._audio_resampler.resample(frame)
            if resampled is None:
                return
            if not isinstance(resampled, list):
                resampled = [resampled]
            for resampled_frame in resampled:
                if resampled_frame is None:
                    continue
                resampled_frame.pts = self._audio_samples_written
                resampled_frame.time_base = fractions.Fraction(1, AUDIO_CLOCK_RATE)
                self._audio_samples_written += int(resampled_frame.samples)
                for packet in self._audio_stream.encode(resampled_frame):
                    self._container.mux(packet)

    async def iter_bytes(self):
        async for chunk in self._buffer.iter_chunks():
            yield chunk

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._container is not None:
                try:
                    if self._video_stream is not None:
                        for packet in self._video_stream.encode(None):
                            self._container.mux(packet)
                    if self._audio_stream is not None:
                        for packet in self._audio_stream.encode(None):
                            self._container.mux(packet)
                    self._container.close()
                except Exception as e:
                    logger.warning("Error finalizing MPEG-TS stream: %s", e)
                self._container = None
                self._video_stream = None
                self._audio_stream = None
                self._audio_resampler = None
            self._buffer.close()


class HeadlessRecorder(HeadlessMediaSink):
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

    def on_video_frame(self, video_frame) -> None:
        self.write_frame(video_frame)

    def on_audio_chunk(
        self, audio_tensor: "torch.Tensor | None", sample_rate: int | None
    ) -> None:
        # Existing headless MP4 path is video-only.
        return

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

    def close(self) -> None:
        self.stop()

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
        expect_audio: bool = False,
    ):
        from .frame_processor import FrameProcessor

        self.frame_processor: FrameProcessor = frame_processor
        self.expect_audio = expect_audio
        self._last_frame = None
        self._last_frames_by_sink: dict[str, object] = {}
        self._frame_lock = threading.Lock()
        self._sink_lock = threading.Lock()
        self._frame_consumer_running = False
        self._frame_consumer_task: asyncio.Task | None = None
        self._recorder: HeadlessRecorder | None = None
        self._stopped_recording_path: str | None = None
        self._media_sinks: list[HeadlessMediaSink] = []
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
            got_any = False

            # Consume from per-sink queues (multi-sink graph mode)
            for sid in self.frame_processor.get_sink_node_ids():
                sink_tensor = self.frame_processor.get_from_sink(sid)
                if sink_tensor is not None:
                    got_any = True
                    frame_np = sink_tensor.numpy()
                    vf = VideoFrame.from_ndarray(frame_np, format="rgb24")
                    with self._frame_lock:
                        self._last_frames_by_sink[sid] = vf
                        self._last_frame = vf

            # Also consume from the primary output queue
            frame_tensor = self.frame_processor.get()
            if frame_tensor is not None:
                got_any = True
                frame_np = frame_tensor.numpy()
                vf = VideoFrame.from_ndarray(frame_np, format="rgb24")
                with self._frame_lock:
                    self._last_frame = vf
                # Write to recorder if active
                if self._recorder and self._recorder.is_recording:
                    self._recorder.write_frame(vf)
                for sink in self._get_sinks_snapshot():
                    try:
                        sink.on_video_frame(vf)
                    except Exception as e:
                        logger.warning("Headless video sink failed: %s", e)
                        self.remove_media_sink(sink)
                        try:
                            sink.close()
                        except Exception as close_err:
                            logger.warning(
                                "Failed to close headless sink: %s", close_err
                            )

            while True:
                audio_tensor, sample_rate = self.frame_processor.get_audio()
                if audio_tensor is None and sample_rate is None:
                    break
                got_any = True
                for sink in self._get_sinks_snapshot():
                    try:
                        sink.on_audio_chunk(audio_tensor, sample_rate)
                    except Exception as e:
                        logger.warning("Headless audio sink failed: %s", e)
                        self.remove_media_sink(sink)
                        try:
                            sink.close()
                        except Exception as close_err:
                            logger.warning(
                                "Failed to close headless sink: %s", close_err
                            )

            if got_any:
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0.01)

    def _get_sinks_snapshot(self) -> list[HeadlessMediaSink]:
        with self._sink_lock:
            return list(self._media_sinks)

    def add_media_sink(self, sink: HeadlessMediaSink) -> None:
        with self._sink_lock:
            if sink not in self._media_sinks:
                self._media_sinks.append(sink)

    def remove_media_sink(self, sink: HeadlessMediaSink) -> None:
        with self._sink_lock:
            if sink in self._media_sinks:
                self._media_sinks.remove(sink)

    def create_ts_streamer(self) -> HeadlessTsStreamer:
        streamer = HeadlessTsStreamer(expect_audio=self.expect_audio)
        self.add_media_sink(streamer)
        return streamer

    def start_recording(self) -> bool:
        """Start recording frames to MP4.

        Returns True if recording was started, False if already recording.
        """
        if self._recorder is not None and self._recorder.is_recording:
            return False
        self._recorder = HeadlessRecorder()
        self._recorder.start()
        self.add_media_sink(self._recorder)
        logger.info("Headless recording started")
        return True

    def stop_recording(self) -> str | None:
        """Stop recording and return the file path, or None if not recording."""
        if self._recorder is None or not self._recorder.is_recording:
            return None
        file_path = self._recorder.stop()
        frame_count = self._recorder.frame_count
        self.remove_media_sink(self._recorder)
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
        if self._frame_consumer_task is not None:
            self._frame_consumer_task.cancel()
            try:
                await self._frame_consumer_task
            except asyncio.CancelledError:
                pass
        for sink in self._get_sinks_snapshot():
            try:
                sink.close()
            except Exception as e:
                logger.warning("Failed to close headless sink: %s", e)
            self.remove_media_sink(sink)
        self._recorder = None
        self.frame_processor.stop()
        logger.info("Headless session closed")

    def get_last_frame(self, sink_node_id: str | None = None):
        """Return the most recently cached frame, or None.

        If sink_node_id is provided, return the frame from that specific sink.
        """
        with self._frame_lock:
            if sink_node_id and sink_node_id in self._last_frames_by_sink:
                return self._last_frames_by_sink[sink_node_id]
            return self._last_frame

    def __str__(self):
        return f"HeadlessSession(running={self.frame_processor.running})"
