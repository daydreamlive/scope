"""Recording-related utility functions for cleanup and download handling."""

import fractions
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaRecorder, MediaRelay
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE

logger = logging.getLogger(__name__)

# Constants
TEMP_FILE_PREFIXES = {
    "recording": "scope_recording_",
    "download": "scope_download_",
}

# Environment variables
RECORDING_ENABLED = os.getenv("RECORDING_ENABLED", "false").lower() == "true"
RECORDING_STARTUP_CLEANUP_ENABLED = (
    os.getenv("RECORDING_STARTUP_CLEANUP_ENABLED", "true").lower() == "true"
)

RECORDING_MAX_FPS = 30.0  # Must match MediaRecorder's hardcoded rate=30


class TimestampNormalizingTrack(MediaStreamTrack):
    """Wraps a track and assigns wall-clock timestamps starting from 0.

    Uses monotonic wall-clock time to compute PTS so that the recorded
    MP4 plays back at real-time speed regardless of the source track's
    own PTS cadence.  This is critical for cloud-relay recordings where
    frames may arrive slower than the source track's nominal rate (e.g.
    CloudTrack stamps every frame at 1/30 s intervals even when network
    round-trips deliver them at 10-15 FPS).

    Important: We must create a copy of the frame rather than modifying it
    in place, because the relay shares frame objects across all subscribers.
    Modifying in place would affect the WebRTC sender and cause encoding errors.
    """

    def __init__(self, source_track: MediaStreamTrack):
        super().__init__()
        self.kind = source_track.kind
        self._source = source_track
        self._start_time: float | None = None
        self._last_frame_time: float | None = None
        self._min_frame_interval = 1.0 / RECORDING_MAX_FPS

    async def recv(self):
        import av

        while True:
            frame = await self._source.recv()

            # Frame rate limiting - skip frames arriving faster than MAX_RECORDING_FPS
            current_time = time.monotonic()
            if self._last_frame_time is not None:
                elapsed = current_time - self._last_frame_time
                if elapsed < self._min_frame_interval:
                    continue  # Skip this frame
            self._last_frame_time = current_time

            if self._start_time is None:
                self._start_time = current_time

            # Create a new frame with wall-clock-based timestamp.
            # Pad to even dimensions — libx264 requires width and height divisible by 2.
            arr = frame.to_ndarray(format="rgb24")
            h, w = arr.shape[:2]
            pad_w = w % 2
            pad_h = h % 2
            if pad_w or pad_h:
                import numpy as np

                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            new_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            new_frame.pts = int((current_time - self._start_time) * VIDEO_CLOCK_RATE)
            new_frame.time_base = VIDEO_TIME_BASE
            return new_frame

    def stop(self):
        self._source.stop()
        super().stop()


class AudioTimestampNormalizingTrack(MediaStreamTrack):
    """Wraps an audio track and assigns wall-clock timestamps starting from 0.

    Analogous to TimestampNormalizingTrack but for AudioFrame objects.
    Uses wall-clock time for PTS to stay in sync with the video track's
    wall-clock timestamps.  Unlike video, audio frames are not rate-limited
    here because the source AudioProcessingTrack already paces at 20ms
    intervals.
    """

    kind = "audio"

    def __init__(self, source_track: MediaStreamTrack):
        super().__init__()
        self._source = source_track
        self._start_time: float | None = None

    async def recv(self):
        from av import AudioFrame as AvAudioFrame

        frame = await self._source.recv()

        current_time = time.monotonic()
        if self._start_time is None:
            self._start_time = current_time

        # Create a copy with wall-clock PTS (relay shares frame objects,
        # so we must not mutate in place).
        new_frame = AvAudioFrame(
            format=frame.format.name,
            layout=frame.layout.name,
            samples=frame.samples,
        )
        new_frame.sample_rate = frame.sample_rate
        new_frame.pts = int((current_time - self._start_time) * frame.sample_rate)
        new_frame.time_base = fractions.Fraction(1, frame.sample_rate)
        for i, plane in enumerate(frame.planes):
            new_frame.planes[i].update(bytes(plane))
        return new_frame

    def stop(self):
        self._source.stop()
        super().stop()


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
        """Create a video recording track.

        Returns None if no video track is configured.  The track is wrapped
        in TimestampNormalizingTrack to ensure frame timestamps start from 0
        for each new recording.
        """
        if self.video_track is None:
            return None
        if self.relay:
            relay_track = self.relay.subscribe(self.video_track)
            return TimestampNormalizingTrack(relay_track)
        else:
            logger.warning("No relay available for recording, using track directly")
            return TimestampNormalizingTrack(self.video_track)

    def _create_audio_recording_track(self) -> MediaStreamTrack | None:
        """Create an audio recording track.

        Returns None if no audio track is configured.
        """
        if self.audio_track is None:
            return None
        if self.audio_relay:
            relay_track = self.audio_relay.subscribe(self.audio_track)
            return AudioTimestampNormalizingTrack(relay_track)
        else:
            logger.warning(
                "No audio relay available for recording, using track directly"
            )
            return AudioTimestampNormalizingTrack(self.audio_track)

    def _create_media_recorder(self, file_path: str) -> MediaRecorder:
        """Create a MediaRecorder instance with standard settings."""
        return MediaRecorder(
            file_path,
            format="mp4",
        )

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
        media_recorder: MediaRecorder | None,
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
    if not RECORDING_STARTUP_CLEANUP_ENABLED:
        logger.info(
            "Recording startup cleanup disabled via RECORDING_STARTUP_CLEANUP_ENABLED"
        )
        return

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
