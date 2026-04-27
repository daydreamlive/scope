"""Recording-related utility functions for cleanup and download handling."""

import asyncio
import fractions
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, VideoFrame

logger = logging.getLogger(__name__)

# Constants
TEMP_FILE_PREFIXES = {
    "recording": "scope_recording_",
    "download": "scope_download_",
}

RECORDING_MAX_FPS = 30.0  # Must match MediaRecorder's hardcoded rate=30


def ensure_even_video_frame(frame: VideoFrame) -> VideoFrame:
    """Pad odd-dimension video frames so encoders like libx264 accept them."""
    pts = frame.pts
    time_base = frame.time_base
    arr = frame.to_ndarray(format="rgb24")
    h, w = arr.shape[:2]
    pad_w = w % 2
    pad_h = h % 2
    if not (pad_w or pad_h):
        return frame

    import numpy as np

    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    even_frame = VideoFrame.from_ndarray(padded, format="rgb24")
    even_frame.pts = pts
    if time_base is not None:
        even_frame.time_base = time_base
    return even_frame


class RecordingManager:
    """Manages recording functionality for video and/or audio tracks."""

    def __init__(
        self,
        video_track: MediaStreamTrack | None = None,
        audio_track: MediaStreamTrack | None = None,
    ):
        self.video_track = video_track
        self.audio_track = audio_track
        self.relay = None
        self.audio_relay = None

        # Recording state
        self.recording_file = None
        self.media_recorder = None
        self.recording_started = False
        self.recording_lock = threading.Lock()
        self.recording_track = None
        self.audio_recording_track = None

    def set_relay(self, relay: MediaRelay):
        """Set the MediaRelay instance for creating video recording track."""
        self.relay = relay

    def set_audio_relay(self, relay: MediaRelay):
        """Set the MediaRelay instance for creating audio recording track."""
        self.audio_relay = relay

    @staticmethod
    def _create_temp_file(suffix: str, prefix: str) -> str:
        """Create a temporary file and return its path."""
        temp_dir = tempfile.gettempdir()
        fd, file_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=temp_dir)
        os.close(fd)
        return file_path

    @staticmethod
    def _stop_track_safe(track: MediaStreamTrack | None) -> None:
        """Safely stop a recording track, ignoring errors."""
        if track:
            try:
                track.stop()
            except Exception as e:
                logger.warning(f"Error stopping recording track: {e}")

    def _create_recording_track(self) -> MediaStreamTrack | None:
        """Create a video recording track, preserving source timestamps."""
        if self.video_track is None:
            return None
        if self.relay:
            return self.relay.subscribe(self.video_track)
        logger.warning("No relay available for recording, using track directly")
        return self.video_track

    def _create_audio_recording_track(self) -> MediaStreamTrack | None:
        """Create an audio recording track, preserving source timestamps."""
        if self.audio_track is None:
            return None
        if self.audio_relay:
            return self.audio_relay.subscribe(self.audio_track)
        logger.warning("No audio relay available for recording, using track directly")
        return self.audio_track

    def _create_media_recorder(self, file_path: str) -> "ScopeMediaRecorder":
        """Create a native PyAV media recorder with MP4 options."""
        return ScopeMediaRecorder(file_path)

    async def start_recording(self):
        """Start recording frames to MP4 file using MediaRecorder."""
        with self.recording_lock:
            if self.recording_started:
                return

        recording_file = None
        media_recorder = None
        recording_track = None
        audio_recording_track = None

        try:
            recording_file = self._create_temp_file(
                ".mp4", TEMP_FILE_PREFIXES["recording"]
            )
            media_recorder = self._create_media_recorder(recording_file)

            recording_track = self._create_recording_track()
            if recording_track is not None:
                media_recorder.addTrack(recording_track)

            audio_recording_track = self._create_audio_recording_track()
            if audio_recording_track is not None:
                media_recorder.addTrack(audio_recording_track)

            await media_recorder.start()

            with self.recording_lock:
                if self.recording_started:
                    # Another thread started recording while we were doing I/O
                    await self._cleanup_recording(
                        media_recorder,
                        recording_track,
                        recording_file,
                        audio_recording_track,
                    )
                    return
                self.recording_file = recording_file
                self.media_recorder = media_recorder
                self.recording_track = recording_track
                self.audio_recording_track = audio_recording_track
                self.recording_started = True

            logger.info(f"Started recording to {recording_file}")
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            await self._cleanup_recording(
                media_recorder,
                recording_track,
                recording_file,
                audio_recording_track,
            )
            raise

    async def _cleanup_recording(
        self,
        media_recorder: "ScopeMediaRecorder | None",
        recording_track: MediaStreamTrack | None,
        recording_file: str | None,
        audio_recording_track: MediaStreamTrack | None = None,
    ) -> None:
        """Clean up recording resources."""
        if media_recorder:
            try:
                await media_recorder.stop()
            except Exception as e:
                logger.warning(f"Error stopping media recorder: {e}")
        self._stop_track_safe(recording_track)
        self._stop_track_safe(audio_recording_track)
        if recording_file and os.path.exists(recording_file):
            try:
                os.remove(recording_file)
            except Exception as e:
                logger.warning(f"Error removing recording file: {e}")

    def _extract_recording_state(self):
        """Extract and clear recording state, returning resources for cleanup."""
        with self.recording_lock:
            if not self.recording_started or not self.media_recorder:
                return None, None, None, None

            recording_file = self.recording_file
            media_recorder = self.media_recorder
            recording_track = self.recording_track
            audio_recording_track = self.audio_recording_track

            self.media_recorder = None
            self.recording_track = None
            self.audio_recording_track = None
            self.recording_started = False
            self.recording_file = None

            return (
                recording_file,
                media_recorder,
                recording_track,
                audio_recording_track,
            )

    async def stop_recording(self):
        """Stop recording and close the output file."""
        try:
            recording_file, media_recorder, recording_track, audio_recording_track = (
                self._extract_recording_state()
            )
            if not recording_file:
                return

            await media_recorder.stop()
            self._stop_track_safe(recording_track)
            self._stop_track_safe(audio_recording_track)
            logger.info(f"Stopped recording, saved to {recording_file}")
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            with self.recording_lock:
                self.media_recorder = None
                self.recording_started = False

    async def finalize_and_get_recording(self, restart_after: bool = True):
        """Finalize the current recording and return a copy for download.

        When restart_after is True (session-level recording), a new recording
        segment is started after the copy. Per-node queue-based recording
        passes restart_after=False so the caller can replace the track.
        """
        try:
            with self.recording_lock:
                has_active_recording = self.recording_started and self.media_recorder

            if not has_active_recording:
                return None

            recording_file, media_recorder, recording_track, audio_recording_track = (
                self._extract_recording_state()
            )

            if media_recorder:
                await media_recorder.stop()
                logger.info(f"Finalized recording: {recording_file}")

            self._stop_track_safe(recording_track)
            self._stop_track_safe(audio_recording_track)

            if recording_file and os.path.exists(recording_file):
                # Create a copy for download
                download_file = self._copy_single_segment(recording_file)

                # Continue recording after download
                if restart_after:
                    await self.start_recording()
                    logger.info("Continued recording after download")

                return download_file

            return None
        except Exception as e:
            logger.error(f"Error finalizing recording: {e}", exc_info=True)
            await self._try_restart_recording()
            return None

    async def _try_restart_recording(self):
        """Try to restart recording if it was stopped."""
        try:
            with self.recording_lock:
                needs_restart = not self.recording_started
            if needs_restart:
                await self.start_recording()
        except Exception as e:
            logger.error(f"Error restarting recording: {e}", exc_info=True)

    def _copy_single_segment(self, segment_path: str) -> str:
        """Copy a recording file to a download file."""
        download_file = self._create_temp_file(".mp4", TEMP_FILE_PREFIXES["download"])
        shutil.copy2(segment_path, download_file)
        logger.info(f"Created download copy: {download_file}")
        return download_file

    @staticmethod
    def _safe_remove_file(file_path: str) -> None:
        """Safely remove a file, logging warnings on failure."""
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove file {file_path}: {e}")

    async def delete_recording(self):
        """Delete all recording files."""
        files_to_delete = []
        media_recorder = None
        recording_track = None
        audio_recording_track = None

        try:
            # Extract recording state and stop the recorder before deleting
            recording_file, media_recorder, recording_track, audio_recording_track = (
                self._extract_recording_state()
            )
            if recording_file:
                files_to_delete.append(recording_file)
        except Exception as e:
            logger.error(f"Error getting recording file paths: {e}")

        # Stop the media recorder first to close the file handle
        if media_recorder:
            try:
                await media_recorder.stop()
            except Exception as e:
                logger.warning(f"Error stopping media recorder during delete: {e}")

        # Stop the recording tracks
        self._stop_track_safe(recording_track)
        self._stop_track_safe(audio_recording_track)

        # Now delete the file(s) - the file handle should be closed
        for file_path in files_to_delete:
            if file_path and os.path.exists(file_path):
                self._safe_remove_file(file_path)
                logger.info(f"Deleted recording file: {file_path}")

    @property
    def is_recording_started(self):
        """Check if recording has been started."""
        return self.recording_started


def cleanup_recording_files():
    """
    Clean up all recording files from previous sessions.
    This handles cases where the process crashed and files weren't cleaned up.
    """
    temp_dir = Path(tempfile.gettempdir())
    if not temp_dir.exists():
        return

    patterns = [
        f"{TEMP_FILE_PREFIXES['recording']}*.mp4",
        f"{TEMP_FILE_PREFIXES['download']}*.mp4",
    ]

    deleted_count = 0
    for pattern in patterns:
        try:
            for file_path in temp_dir.glob(pattern):
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Cleaned up recording file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete recording file {file_path}: {e}")
        except Exception as e:
            logger.warning(
                f"Error cleaning up recording files with pattern {pattern}: {e}"
            )

    if deleted_count > 0:
        logger.info(
            f"Cleaned up {deleted_count} recording file(s) from previous session(s)"
        )
    else:
        logger.debug("No recording files found to clean up")


def cleanup_temp_file(file_path: str):
    """Clean up temporary file after download."""
    if os.path.exists(file_path):
        RecordingManager._safe_remove_file(file_path)
        logger.info(f"Cleaned up temporary download file: {file_path}")


class _RecorderContext:
    def __init__(self, track: MediaStreamTrack):
        self.track = track
        self.started = False
        self.stream: Any | None = None
        self.task: asyncio.Task[None] | None = None
        self.codec_time_base_initialized = False


class ScopeMediaRecorder:
    """PyAV-based replacement for aiortc MediaRecorder.

    Keeps the same high-level API surface used by RecordingManager
    (`addTrack`, `start`, `stop`) while preserving incoming frame PTS/time_base.
    """

    def __init__(self, file_path: str):
        import av

        self._container = av.open(
            file_path,
            mode="w",
            format="mp4",
            options={
                # Force timestamps to start at zero (disable edit list).
                "use_editlist": "0",
                # Allow playback before file is fully loaded, e.g. over HTTP.
                "movflags": "+faststart",
            },
        )
        self._contexts: dict[MediaStreamTrack, _RecorderContext] = {}

    def addTrack(self, track: MediaStreamTrack) -> None:
        context = _RecorderContext(track)
        context.stream = self._create_stream(track.kind)
        self._contexts[track] = context

    async def start(self) -> None:
        for context in self._contexts.values():
            if context.task is None:
                context.task = asyncio.create_task(self._run_track(context))

    async def stop(self) -> None:
        tasks: list[asyncio.Task[None]] = []
        for ctx in self._contexts.values():
            if ctx.task is not None:
                ctx.task.cancel()
                tasks.append(ctx.task)
                ctx.task = None
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self._container is not None:
            for context in self._contexts.values():
                assert context.stream is not None
                for packet in context.stream.encode(None):
                    self._container.mux(packet)
            self._container.close()
            self._container = None

        self._contexts = {}

    async def _run_track(self, context: _RecorderContext) -> None:
        while True:
            try:
                frame = await context.track.recv()
            except asyncio.CancelledError:
                return
            except MediaStreamError:
                return

            if isinstance(frame, VideoFrame):
                frame = ensure_even_video_frame(frame)
            elif not isinstance(frame, AudioFrame):
                raise TypeError("Only audio or video frames can be recorded")

            if self._container is None:
                return
            if context.stream is None:
                return

            if isinstance(frame, VideoFrame) and not context.started:
                context.stream.width = frame.width
                context.stream.height = frame.height
                context.started = True
            elif isinstance(frame, AudioFrame) and not context.started:
                try:
                    context.stream.rate = frame.sample_rate or context.stream.rate
                    context.stream.layout = frame.layout.name
                except Exception:
                    pass
                context.started = True

            self._initialize_codec_time_base(context, frame)
            for packet in context.stream.encode(frame):
                self._container.mux(packet)

    def _create_stream(self, kind: str):
        assert self._container is not None
        if kind == "video":
            stream = self._container.add_stream("libx264", rate=int(RECORDING_MAX_FPS))
            stream.pix_fmt = "yuv420p"
            return stream

        return self._container.add_stream("aac")

    @staticmethod
    def _initialize_codec_time_base(
        context: _RecorderContext, frame: AudioFrame | VideoFrame
    ) -> None:
        if context.codec_time_base_initialized:
            return
        if frame.time_base is None:
            return
        try:
            context.stream.codec_context.time_base = fractions.Fraction(frame.time_base)
            context.codec_time_base_initialized = True
        except Exception:
            # If the codec rejects this time base, keep encoder defaults.
            return
