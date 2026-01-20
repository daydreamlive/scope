"""Recording-related utility functions for cleanup and download handling."""

import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaRecorder, MediaRelay

logger = logging.getLogger(__name__)

# Constants
TEMP_FILE_PREFIXES = {
    "recording": "scope_recording_",
    "download": "scope_download_",
}

# Environment variables
RECORDING_ENABLED = os.getenv("RECORDING_ENABLED", "true").lower() == "true"
RECORDING_MAX_LENGTH_STR = os.getenv("RECORDING_MAX_LENGTH", "1h")
RECORDING_STARTUP_CLEANUP_ENABLED = (
    os.getenv("RECORDING_STARTUP_CLEANUP_ENABLED", "true").lower() == "true"
)

RECORDING_MAX_FPS = 30.0  # Must match MediaRecorder's hardcoded rate=30


def _parse_time_duration(duration_str: str) -> float:
    """
    Parse a time duration string (e.g., '1h', '30m', '120s') to seconds.

    Args:
        duration_str: Duration string like '1h', '30m', '120s', or just a number (treated as seconds)

    Returns:
        Duration in seconds
    """
    duration_str = duration_str.strip().lower()

    if duration_str.endswith("h"):
        return float(duration_str[:-1]) * 3600
    elif duration_str.endswith("m"):
        return float(duration_str[:-1]) * 60
    elif duration_str.endswith("s"):
        return float(duration_str[:-1])
    else:
        # Try to parse as seconds
        try:
            return float(duration_str)
        except ValueError:
            logger.warning(
                f"Invalid duration format: {duration_str}, defaulting to 3600s (1h)"
            )
            return 3600.0


RECORDING_MAX_LENGTH_SECONDS = _parse_time_duration(RECORDING_MAX_LENGTH_STR)


class TimestampNormalizingTrack(MediaStreamTrack):
    """Wraps a track and normalizes frame timestamps to start from 0.

    This is needed because when starting a new recording, the source track's
    frames have PTS values that continue from the previous recording. Without
    normalization, the MP4 encoder interprets these high PTS values as
    indicating the frame should appear later in the video, causing black
    frames at the start.

    Important: We must create a copy of the frame rather than modifying it
    in place, because the relay shares frame objects across all subscribers.
    Modifying in place would affect the WebRTC sender and cause encoding errors.
    """

    def __init__(self, source_track: MediaStreamTrack):
        super().__init__()
        self.kind = source_track.kind
        self._source = source_track
        self._base_pts = None  # Will be set on first frame
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

            # Capture the first frame's PTS as our base
            if self._base_pts is None:
                self._base_pts = frame.pts

            # Create a new frame with normalized timestamp
            arr = frame.to_ndarray(format="rgb24")
            new_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            new_frame.pts = frame.pts - self._base_pts
            new_frame.time_base = frame.time_base
            return new_frame

    def stop(self):
        self._source.stop()
        super().stop()


class RecordingManager:
    """Manages recording functionality for a video track."""

    def __init__(self, video_track: MediaStreamTrack):
        """
        Initialize the recording manager.

        Args:
            video_track: The video track to record from
        """
        self.video_track = video_track
        self.relay = None

        # Recording state
        self.recording_file = None
        self.media_recorder = None
        self.recording_started = False
        self.recording_lock = threading.Lock()
        self.recording_track = None

        # Max length tracking
        self.first_recording_start_time = None
        self.max_length_reached = False

    def set_relay(self, relay: MediaRelay):
        """Set the MediaRelay instance for creating recording track."""
        self.relay = relay

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

    def _create_recording_track(self) -> MediaStreamTrack:
        """Create a recording track from the video track.

        The track is wrapped in TimestampNormalizingTrack to ensure frame
        timestamps start from 0 for each new recording.
        """
        if self.relay:
            relay_track = self.relay.subscribe(self.video_track)
            return TimestampNormalizingTrack(relay_track)
        else:
            logger.warning("No relay available for recording, using track directly")
            return TimestampNormalizingTrack(self.video_track)

    def _create_media_recorder(self, file_path: str) -> MediaRecorder:
        """Create a MediaRecorder instance with standard settings."""
        return MediaRecorder(
            file_path,
            format="mp4",
        )

    async def start_recording(self):
        """Start recording frames to MP4 file using MediaRecorder."""
        if not RECORDING_ENABLED:
            logger.debug(
                "Recording is disabled via RECORDING_ENABLED environment variable"
            )
            return

        with self.recording_lock:
            if self.recording_started:
                return

            # Check if max length has been reached
            if self.max_length_reached:
                return

        recording_file = None
        media_recorder = None
        recording_track = None

        try:
            recording_file = self._create_temp_file(
                ".mp4", TEMP_FILE_PREFIXES["recording"]
            )
            media_recorder = self._create_media_recorder(recording_file)
            recording_track = self._create_recording_track()
            media_recorder.addTrack(recording_track)
            await media_recorder.start()

            with self.recording_lock:
                if self.recording_started:
                    # Another thread started recording while we were doing I/O
                    await self._cleanup_recording(
                        media_recorder, recording_track, recording_file
                    )
                    return
                self.recording_file = recording_file
                self.media_recorder = media_recorder
                self.recording_track = recording_track
                self.recording_started = True

                # Track first recording start time
                if self.first_recording_start_time is None:
                    self.first_recording_start_time = time.time()

            logger.info(f"Started recording to {recording_file}")
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            await self._cleanup_recording(
                media_recorder, recording_track, recording_file
            )
            raise

    def check_max_length(self) -> bool:
        """
        Check if recording has exceeded max length and stop if necessary.
        This should be called periodically during recording.

        Returns:
            True if max length was reached and recording should be stopped, False otherwise
        """
        if not RECORDING_ENABLED or self.max_length_reached:
            return False

        with self.recording_lock:
            if not self.recording_started:
                return False

            # Calculate total duration from start time
            if self.first_recording_start_time is not None:
                total_duration = time.time() - self.first_recording_start_time

                if total_duration >= RECORDING_MAX_LENGTH_SECONDS:
                    if not self.max_length_reached:
                        # Only log once when max length is first reached
                        logger.info(
                            f"Recording max length reached (total: {total_duration:.2f}s, max: {RECORDING_MAX_LENGTH_SECONDS}s). Stopping recording."
                        )
                    self.max_length_reached = True
                    return True

        return False

    async def stop_recording_if_max_length_reached(self):
        """Stop current recording if max length has been reached."""
        if self.max_length_reached and self.recording_started:
            await self.stop_recording()

    async def _cleanup_recording(
        self,
        media_recorder: MediaRecorder | None,
        recording_track: MediaStreamTrack | None,
        recording_file: str | None,
    ) -> None:
        """Clean up recording resources."""
        if media_recorder:
            try:
                await media_recorder.stop()
            except Exception as e:
                logger.warning(f"Error stopping media recorder: {e}")
        self._stop_track_safe(recording_track)
        if recording_file and os.path.exists(recording_file):
            try:
                os.remove(recording_file)
            except Exception as e:
                logger.warning(f"Error removing recording file: {e}")

    def _extract_recording_state(self):
        """Extract and clear recording state, returning resources for cleanup."""
        with self.recording_lock:
            if not self.recording_started or not self.media_recorder:
                return None, None, None

            recording_file = self.recording_file
            media_recorder = self.media_recorder
            recording_track = self.recording_track

            self.media_recorder = None
            self.recording_track = None
            self.recording_started = False
            self.recording_file = None

            return recording_file, media_recorder, recording_track

    async def stop_recording(self):
        """Stop recording and close the output file."""
        try:
            recording_file, media_recorder, recording_track = (
                self._extract_recording_state()
            )
            if not recording_file:
                return

            await media_recorder.stop()
            self._stop_track_safe(recording_track)
            logger.info(f"Stopped recording, saved to {recording_file}")
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            with self.recording_lock:
                self.media_recorder = None
                self.recording_started = False

    async def finalize_and_get_recording(self):
        """Finalize the current recording and return a copy for download."""
        try:
            with self.recording_lock:
                has_active_recording = self.recording_started and self.media_recorder

            if not has_active_recording:
                return None

            recording_file, media_recorder, recording_track = (
                self._extract_recording_state()
            )

            if media_recorder:
                await media_recorder.stop()
                logger.info(f"Finalized recording: {recording_file}")

            self._stop_track_safe(recording_track)

            if recording_file and os.path.exists(recording_file):
                # Create a copy for download
                download_file = self._copy_single_segment(recording_file)

                # Continue recording if max length not reached
                if not self.max_length_reached:
                    await self.start_recording()
                    logger.info("Continued recording after download")
                else:
                    logger.info("Skipped starting new recording (max length reached)")

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
        try:
            with self.recording_lock:
                if self.recording_file:
                    files_to_delete.append(self.recording_file)
                self.recording_file = None
        except Exception as e:
            logger.error(f"Error getting recording file paths: {e}")

        for file_path in files_to_delete:
            if file_path and os.path.exists(file_path):
                self._safe_remove_file(file_path)
                logger.info(f"Deleted recording file: {file_path}")

    def get_recording_path(self):
        """Get the path to the recording file."""
        return Path(self.recording_file) if self.recording_file else None

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
