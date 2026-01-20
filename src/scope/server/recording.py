"""Recording-related utility functions for cleanup and download handling."""

import asyncio
import inspect
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import ffmpeg
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaRecorder, MediaRelay
from av import VideoFrame

logger = logging.getLogger(__name__)

# Constants
FRAME_QUEUE_MAXSIZE = 2
FFMPEG_TIMEOUT = 30.0
TEMP_FILE_PREFIXES = {
    "recording": "scope_recording_",
    "download": "scope_download_",
    "concat": "scope_concat_",
}

# Environment variables
RECORDING_ENABLED = os.getenv("RECORDING_ENABLED", "true").lower() == "true"
RECORDING_MAX_LENGTH_STR = os.getenv("RECORDING_MAX_LENGTH", "1h")
RECORDING_STARTUP_CLEANUP_ENABLED = (
    os.getenv("RECORDING_STARTUP_CLEANUP_ENABLED", "true").lower() == "true"
)


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


class RecordingTrack(MediaStreamTrack):
    """A track wrapper that only forwards frames when not paused."""

    kind = "video"

    def __init__(
        self,
        source_track: MediaStreamTrack,
        is_paused: callable,
        recording_manager=None,
    ):
        super().__init__()
        self.source_track = source_track
        self.is_paused = is_paused
        self.recording_manager = recording_manager
        self._frame_queue = asyncio.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self._consumer_task = None
        self._running = True

    def _try_queue_frame(self, frame: VideoFrame):
        """Try to queue a frame, dropping oldest if queue is full."""
        try:
            self._frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except asyncio.QueueEmpty:
                pass

    async def _frame_consumer(self):
        """Background task that consumes frames from source track and queues them when not paused."""
        try:
            while self._running:
                frame = await self.source_track.recv()
                if not self.is_paused():
                    self._try_queue_frame(frame)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in frame consumer: {e}")

    async def recv(self) -> VideoFrame:
        """Return frames only when not paused. When paused, wait (MediaRecorder will pause)."""
        # Check max recording length periodically (on every frame)
        # Only check if recording is active and max length hasn't been reached yet
        if (
            self.recording_manager
            and not self.is_paused()
            and self.recording_manager.is_recording_started
            and not self.recording_manager.max_length_reached
        ):
            if self.recording_manager.check_max_length():
                await self.recording_manager.stop_recording_if_max_length_reached()

        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._frame_consumer())
        return await self._frame_queue.get()

    async def stop(self):
        """Stop the track and cleanup."""
        self._running = False
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        result = super().stop()
        if result is not None and inspect.iscoroutine(result):
            await result


class RecordingManager:
    """Manages recording functionality for a video track."""

    def __init__(self, video_track: MediaStreamTrack, is_paused: callable):
        """
        Initialize the recording manager.

        Args:
            video_track: The video track to record from
            is_paused: Callable that returns True when paused, False otherwise
        """
        self.video_track = video_track
        self.is_paused = is_paused
        self.relay = None

        # Recording state
        self.recording_file = None
        self.media_recorder = None
        self.recording_started = False
        self.recording_lock = threading.Lock()
        self.recording_segments = []
        self.segment_durations = {}  # Map segment file path -> duration in seconds
        self.recording_track = None

        # Max length tracking
        self.first_recording_start_time = None
        self.current_segment_start_time = None  # Start time of current active segment
        self.total_recorded_duration = (
            0.0  # Sum of all finalized segment durations in seconds
        )
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
    async def _stop_track_safe(track: MediaStreamTrack | None) -> None:
        """Safely stop a recording track, ignoring errors."""
        if track:
            try:
                await track.stop()
            except Exception as e:
                logger.warning(f"Error stopping recording track: {e}")

    def _create_recording_track(self) -> RecordingTrack:
        """Create a pause-aware recording track from the video track."""
        source_track = (
            self.relay.subscribe(self.video_track) if self.relay else self.video_track
        )
        if not self.relay:
            logger.warning(
                "No relay available for recording, using track directly with pause awareness"
            )
        return RecordingTrack(source_track, self.is_paused, recording_manager=self)

    def _create_media_recorder(self, file_path: str) -> MediaRecorder:
        """Create a MediaRecorder instance with standard settings."""
        return MediaRecorder(
            file_path,
            format="mp4",
            options={
                "vcodec": "libx264",
                "preset": "ultrafast",
                "crf": "23",
                "tune": "zerolatency",
            },
        )

    async def start_recording(self):
        """Start recording frames to MP4 file using MediaRecorder."""
        if not RECORDING_ENABLED:
            logger.debug(
                "Recording is disabled via RECORDING_ENABLED environment variable"
            )
            return

        if self.is_paused():
            logger.debug("Skipping recording start - video is paused")
            return

        with self.recording_lock:
            if self.recording_started:
                return

            # Check if max length has been reached
            if self.max_length_reached:
                return

            # Check if total recorded duration would exceed max length
            if self.total_recorded_duration >= RECORDING_MAX_LENGTH_SECONDS:
                logger.info(
                    f"Recording max length reached (total: {self.total_recorded_duration:.2f}s, max: {RECORDING_MAX_LENGTH_SECONDS}s)"
                )
                self.max_length_reached = True
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

                # Track current segment start time
                self.current_segment_start_time = time.time()

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

            # Calculate total duration: finalized segments + current segment elapsed time
            current_segment_duration = 0.0
            if self.current_segment_start_time is not None:
                current_segment_duration = time.time() - self.current_segment_start_time

            total_duration = self.total_recorded_duration + current_segment_duration

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
        """Stop current recording segment if max length has been reached."""
        if self.max_length_reached and self.recording_started:
            await self.finalize_recording_segment()

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
        await self._stop_track_safe(recording_track)
        if recording_file and os.path.exists(recording_file):
            try:
                os.remove(recording_file)
            except Exception as e:
                logger.warning(f"Error removing recording file: {e}")

    async def finalize_recording_segment(self):
        """Finalize the current recording segment when stream is paused."""
        try:
            recording_file, media_recorder, recording_track = (
                self._extract_recording_state()
            )
            if not recording_file:
                return

            if media_recorder:
                await media_recorder.stop()
                logger.info(f"Finalized recording segment on pause: {recording_file}")

            await self._stop_track_safe(recording_track)

            if os.path.exists(recording_file):
                # Calculate segment duration from start time
                segment_duration = 0.0
                if self.current_segment_start_time is not None:
                    segment_duration = time.time() - self.current_segment_start_time

                with self.recording_lock:
                    self.recording_segments.append(recording_file)
                    self.segment_durations[recording_file] = segment_duration
                    self.total_recorded_duration += segment_duration
                    self.current_segment_start_time = (
                        None  # Reset current segment tracking
                    )

                    # Check if max length reached
                    if self.total_recorded_duration >= RECORDING_MAX_LENGTH_SECONDS:
                        self.max_length_reached = True
                        logger.info(
                            f"Recording max length reached (total: {self.total_recorded_duration:.2f}s, max: {RECORDING_MAX_LENGTH_SECONDS}s)"
                        )

                    logger.info(
                        f"Added recording segment to list: {recording_file} (duration: {segment_duration:.2f}s, total: {self.total_recorded_duration:.2f}s)"
                    )
        except Exception as e:
            logger.error(
                f"Error finalizing recording segment on pause: {e}", exc_info=True
            )

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
            self.current_segment_start_time = None

            return recording_file, media_recorder, recording_track

    async def start_new_recording_segment(self):
        """Start a new recording segment when stream is resumed."""
        try:
            with self.recording_lock:
                was_recording = len(self.recording_segments) > 0
                if self.max_length_reached:
                    logger.info(
                        "Max recording length reached, not starting new segment"
                    )
                    return

            if was_recording:
                await self.start_recording()
                logger.info("Started new recording segment on resume")
        except Exception as e:
            logger.error(
                f"Error starting new recording segment on resume: {e}", exc_info=True
            )

    def handle_pause_state_change(self, paused: bool, was_paused: bool):
        """Handle recording segmentation when pause state changes.

        Finalizes current recording segment on pause, starts new segment on resume.
        This method is designed to be called from a synchronous context and will
        schedule async tasks if an event loop is available.

        Args:
            paused: Current pause state
            was_paused: Previous pause state
        """
        try:
            loop = asyncio.get_event_loop()
            if paused and not was_paused:
                loop.create_task(self.finalize_recording_segment())
            elif not paused and was_paused:
                loop.create_task(self.start_new_recording_segment())
        except RuntimeError:
            logger.warning("No event loop available for recording segmentation")

    async def stop_recording(self):
        """Stop recording and close the output file."""
        try:
            recording_file, media_recorder, recording_track = (
                self._extract_recording_state()
            )
            if not recording_file:
                return

            await media_recorder.stop()
            await self._stop_track_safe(recording_track)
            logger.info(f"Stopped recording, saved to {recording_file}")
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            with self.recording_lock:
                self.media_recorder = None
                self.recording_started = False

    async def finalize_and_get_recording(self):
        """Finalize the current recording and return a copy for download.
        Uses segment-based recording: each export concatenates all segments from stream start."""
        try:
            with self.recording_lock:
                has_active_recording = self.recording_started and self.media_recorder
                has_segments = len(self.recording_segments) > 0

            if not has_active_recording:
                return (
                    await self._concatenate_segments_for_download()
                    if has_segments
                    else None
                )

            recording_file, media_recorder, recording_track = (
                self._extract_recording_state()
            )

            if media_recorder:
                await media_recorder.stop()
                logger.info(f"Finalized recording segment: {recording_file}")

            await self._stop_track_safe(recording_track)

            if recording_file and os.path.exists(recording_file):
                # Calculate segment duration from start time
                segment_duration = 0.0
                if self.current_segment_start_time is not None:
                    segment_duration = time.time() - self.current_segment_start_time

                with self.recording_lock:
                    self.recording_segments.append(recording_file)
                    self.segment_durations[recording_file] = segment_duration
                    self.total_recorded_duration += segment_duration
                    self.current_segment_start_time = (
                        None  # Reset current segment tracking
                    )

                    # Check if max length reached
                    if self.total_recorded_duration >= RECORDING_MAX_LENGTH_SECONDS:
                        self.max_length_reached = True
                        logger.info(
                            f"Recording max length reached (total: {self.total_recorded_duration:.2f}s, max: {RECORDING_MAX_LENGTH_SECONDS}s)"
                        )

                    logger.info(
                        f"Finalized recording segment: {recording_file} (duration: {segment_duration:.2f}s, total: {self.total_recorded_duration:.2f}s)"
                    )

            download_file = await self._concatenate_segments_for_download()

            if not self.is_paused() and not self.max_length_reached:
                await self.start_recording()
                logger.info("Continued recording after download")
            elif self.max_length_reached:
                logger.info(
                    "Skipped starting new recording segment (max length reached)"
                )
            else:
                logger.info("Skipped starting new recording segment (stream is paused)")

            return download_file
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

    async def _concatenate_segments_for_download(self):
        """Concatenate recording segments into a single file for download.
        Only includes segments up to the max length limit."""
        with self.recording_lock:
            segments = list(self.recording_segments)

        existing_segments = [s for s in segments if os.path.exists(s)]
        if not existing_segments:
            logger.warning("No existing recording segments found")
            return None

        # Filter segments to only include those within max length
        filtered_segments = []
        cumulative_duration = 0.0

        with self.recording_lock:
            segment_durations = self.segment_durations.copy()

        for segment in existing_segments:
            # Get stored duration, or 0.0 if not found (shouldn't happen)
            segment_duration = segment_durations.get(segment, 0.0)
            if cumulative_duration + segment_duration <= RECORDING_MAX_LENGTH_SECONDS:
                filtered_segments.append(segment)
                cumulative_duration += segment_duration
            else:
                # This segment would exceed max length, stop here
                remaining_time = RECORDING_MAX_LENGTH_SECONDS - cumulative_duration
                if remaining_time > 0:
                    # Optionally trim the last segment, but for simplicity, we'll just include it
                    # The segment might be slightly over, but that's acceptable
                    filtered_segments.append(segment)
                logger.info(
                    f"Stopping at segment {segment} to respect max length limit "
                    f"(cumulative: {cumulative_duration:.2f}s, max: {RECORDING_MAX_LENGTH_SECONDS}s)"
                )
                break

        if not filtered_segments:
            logger.warning("No segments within max length limit")
            return None

        if len(filtered_segments) == 1:
            return self._copy_single_segment(filtered_segments[0])

        return await self._concatenate_multiple_segments(filtered_segments)

    def _copy_single_segment(self, segment_path: str) -> str:
        """Copy a single segment to a download file."""
        download_file = self._create_temp_file(".mp4", TEMP_FILE_PREFIXES["download"])
        shutil.copy2(segment_path, download_file)
        logger.info(f"Created download copy from single segment: {download_file}")
        return download_file

    async def _concatenate_multiple_segments(self, segments: list[str]) -> str | None:
        """Concatenate multiple segments using ffmpeg."""
        concat_list_file = None
        download_file = None

        try:
            concat_list_file = self._create_concat_list_file(segments)
            download_file = self._create_temp_file(
                ".mp4", TEMP_FILE_PREFIXES["download"]
            )

            await self._run_ffmpeg_concat(concat_list_file, download_file)
            logger.info(f"Concatenated {len(segments)} segments into {download_file}")
            return download_file
        except (TimeoutError, ffmpeg.Error) as e:
            logger.error(f"ffmpeg concatenation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error concatenating segments: {e}", exc_info=True)
            return None
        finally:
            if concat_list_file:
                self._safe_remove_file(concat_list_file)

    def _create_concat_list_file(self, segments: list[str]) -> str:
        """Create a temporary file list for ffmpeg concat demuxer."""
        concat_list_file = self._create_temp_file(".txt", TEMP_FILE_PREFIXES["concat"])
        with open(concat_list_file, "w") as f:
            for segment in segments:
                abs_path = os.path.abspath(segment)
                f.write(f"file '{abs_path}'\n")
        return concat_list_file

    async def _run_ffmpeg_concat(self, concat_list_file: str, output_file: str):
        """Run ffmpeg to concatenate segments."""
        input_stream = ffmpeg.input(concat_list_file, format="concat", safe=0)
        output_stream = ffmpeg.output(input_stream, output_file, c="copy")

        def run_ffmpeg():
            ffmpeg.run(
                output_stream,
                overwrite_output=True,
                quiet=True,
                capture_stdout=True,
                capture_stderr=True,
            )

        await asyncio.wait_for(asyncio.to_thread(run_ffmpeg), timeout=FFMPEG_TIMEOUT)

    @staticmethod
    def _safe_remove_file(file_path: str) -> None:
        """Safely remove a file, logging warnings on failure."""
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove file {file_path}: {e}")

    async def delete_recording(self):
        """Delete all recording files and segments."""
        files_to_delete = []
        try:
            with self.recording_lock:
                if self.recording_file:
                    files_to_delete.append(self.recording_file)
                files_to_delete.extend(self.recording_segments)
                self.recording_file = None
                self.recording_segments = []
                self.segment_durations = {}
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
        f"{TEMP_FILE_PREFIXES['concat']}*.txt",
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
