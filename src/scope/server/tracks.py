import asyncio
import fractions
import logging
import threading
import time

import numpy as np
import torch
from aiortc import MediaStreamTrack
from aiortc.mediastreams import (
    AUDIO_PTIME,
    VIDEO_CLOCK_RATE,
    VIDEO_TIME_BASE,
    MediaStreamError,
)
from av import AudioFrame, VideoFrame

from .frame_processor import FrameProcessor
from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

# Audio constants for WebRTC
AUDIO_SAMPLE_RATE = 48000  # WebRTC standard
AUDIO_CHANNELS = 2  # Stereo
AUDIO_SAMPLES_PER_FRAME = int(AUDIO_PTIME * AUDIO_SAMPLE_RATE)  # Samples per 20ms frame


class VideoProcessingTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        fps: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        # FPS variables (will be updated from FrameProcessor or input measurement)
        self.fps = fps
        self.frame_ptime = 1.0 / fps

        self.frame_processor = None
        self.input_task = None
        self.input_task_running = False
        self._paused = False
        self._paused_lock = threading.Lock()
        self._last_frame = None

        # Spout input mode - when enabled, frames come from Spout instead of WebRTC
        self._spout_receiver_enabled = False
        if initial_parameters:
            spout_receiver = initial_parameters.get("spout_receiver")
            if spout_receiver and spout_receiver.get("enabled"):
                self._spout_receiver_enabled = True
                logger.info("Spout input mode enabled")

    async def input_loop(self):
        """Background loop that continuously feeds frames to the processor"""
        while self.input_task_running:
            try:
                input_frame = await self.track.recv()

                # Store raw VideoFrame for later processing (tracks input FPS internally)
                self.frame_processor.put(input_frame)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Stop the input loop on connection errors to avoid spam
                logger.error(f"Error in input loop, stopping: {e}")
                self.input_task_running = False
                break

    # Copied from https://github.com/livepeer/fastworld/blob/e649ef788cd33d78af6d8e1da915cd933761535e/backend/track.py#L267
    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Override to control frame rate"""
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "timestamp"):
            # Calculate wait time based on current frame rate
            current_time = time.time()
            time_since_last_frame = current_time - self.last_frame_time

            # Wait for the appropriate interval based on current FPS
            target_interval = self.frame_ptime  # Current frame period
            wait_time = target_interval - time_since_last_frame

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Update timestamp and last frame time
            self.timestamp += int(self.frame_ptime * VIDEO_CLOCK_RATE)
            self.last_frame_time = time.time()
        else:
            self.start = time.time()
            self.last_frame_time = time.time()
            self.timestamp = 0

        return self.timestamp, VIDEO_TIME_BASE

    def initialize_output_processing(self):
        if not self.frame_processor:
            self.frame_processor = FrameProcessor(
                pipeline_manager=self.pipeline_manager,
                initial_parameters=self.initial_parameters,
                notification_callback=self.notification_callback,
            )
            self.frame_processor.start()

    def initialize_input_processing(self, track: MediaStreamTrack):
        self.track = track
        self.input_task_running = True
        self.input_task = asyncio.create_task(self.input_loop())

    async def recv(self) -> VideoFrame:
        """Return the next available processed frame"""
        # Lazy initialization on first call
        self.initialize_output_processing()

        # Keep running while either WebRTC input is active OR Spout input is enabled
        while self.input_task_running or self._spout_receiver_enabled:
            try:
                # Update FPS: use minimum of input FPS and pipeline FPS
                if self.frame_processor:
                    self.fps = self.frame_processor.get_output_fps()
                    self.frame_ptime = 1.0 / self.fps

                # If paused, wait for the appropriate frame interval before returning
                with self._paused_lock:
                    paused = self._paused

                frame = None
                if paused:
                    # When video is paused, return the last frame to freeze the playback video
                    frame = self._last_frame
                else:
                    # When video is not paused, get the next frame from the frame processor
                    frame_tensor = self.frame_processor.get()
                    if frame_tensor is not None:
                        frame = VideoFrame.from_ndarray(
                            frame_tensor.numpy(), format="rgb24"
                        )

                if frame is not None:
                    pts, time_base = await self.next_timestamp()
                    frame.pts = pts
                    frame.time_base = time_base

                    self._last_frame = frame
                    return frame

                # No frame available, wait a bit before trying again
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error getting processed frame: {e}")
                raise

        raise Exception("Track stopped")

    def pause(self, paused: bool):
        """Pause or resume the video track processing"""
        with self._paused_lock:
            self._paused = paused
        logger.info(f"Video track {'paused' if paused else 'resumed'}")

    async def stop(self):
        self.input_task_running = False
        self._spout_receiver_enabled = False

        if self.input_task is not None:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        if self.frame_processor is not None:
            self.frame_processor.stop()

        await super().stop()


class AudioProcessingTrack(MediaStreamTrack):
    """Audio track that provides audio from LTX2 pipeline generation."""

    kind = "audio"

    def __init__(
        self,
        pipeline_manager: PipelineManager,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.audio_buffer = []  # Buffer for audio samples
        self.audio_buffer_lock = threading.Lock()
        self.audio_sample_rate = AUDIO_SAMPLE_RATE
        self.audio_position = 0  # Current position in samples
        self._running = True

    def set_audio_from_pipeline(self):
        """Extract audio from the current pipeline if available."""
        try:
            pipeline = self.pipeline_manager.get_pipeline()
            if hasattr(pipeline, "audio_samples") and pipeline.audio_samples is not None:
                audio_samples = pipeline.audio_samples
                source_sample_rate = pipeline.audio_sample_rate

                # Convert to numpy if needed
                if isinstance(audio_samples, torch.Tensor):
                    audio_samples = audio_samples.cpu().numpy()

                # Resample if needed (LTX2 uses 24kHz, WebRTC needs 48kHz)
                if source_sample_rate != self.audio_sample_rate:
                    audio_samples = self._resample_audio(
                        audio_samples, source_sample_rate, self.audio_sample_rate
                    )

                # Ensure stereo
                if audio_samples.ndim == 1:
                    # Mono to stereo
                    audio_samples = np.stack([audio_samples, audio_samples], axis=0)
                elif audio_samples.shape[0] == 1:
                    # Mono to stereo (from [1, samples] to [2, samples])
                    audio_samples = np.repeat(audio_samples, 2, axis=0)

                with self.audio_buffer_lock:
                    self.audio_buffer = audio_samples.T  # Shape: [samples, channels]
                    self.audio_position = 0

                logger.info(
                    f"Loaded audio buffer: {self.audio_buffer.shape} samples at {self.audio_sample_rate}Hz"
                )
        except Exception as e:
            logger.error(f"Error loading audio from pipeline: {e}")

    def _resample_audio(
        self, audio: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """Simple linear resampling of audio."""
        if source_rate == target_rate:
            return audio

        # Calculate resampling ratio
        ratio = target_rate / source_rate

        # Handle channel dimension
        if audio.ndim == 1:
            channels = 1
            audio = audio.reshape(1, -1)
        else:
            channels = audio.shape[0]

        resampled_channels = []
        for ch in range(channels):
            channel_data = audio[ch]
            # Create new time indices
            old_indices = np.arange(len(channel_data))
            new_length = int(len(channel_data) * ratio)
            new_indices = np.linspace(0, len(channel_data) - 1, new_length)
            # Interpolate
            resampled = np.interp(new_indices, old_indices, channel_data)
            resampled_channels.append(resampled)

        result = np.stack(resampled_channels, axis=0)
        return result.squeeze() if channels == 1 else result

    async def recv(self) -> AudioFrame:
        """Return the next audio frame."""
        if not self._running:
            raise MediaStreamError

        # Load audio if buffer is empty
        with self.audio_buffer_lock:
            if len(self.audio_buffer) == 0 or self.audio_position >= len(
                self.audio_buffer
            ):
                # Try to load new audio from pipeline
                self.set_audio_from_pipeline()

                # If still no audio, return silence
                if len(self.audio_buffer) == 0:
                    await asyncio.sleep(AUDIO_PTIME)
                    return self._create_silence_frame()

        # Extract samples for this frame
        with self.audio_buffer_lock:
            end_pos = min(
                self.audio_position + AUDIO_SAMPLES_PER_FRAME, len(self.audio_buffer)
            )
            samples = self.audio_buffer[self.audio_position : end_pos]

            # Pad with zeros if not enough samples
            if len(samples) < AUDIO_SAMPLES_PER_FRAME:
                padding = np.zeros(
                    (AUDIO_SAMPLES_PER_FRAME - len(samples), AUDIO_CHANNELS),
                    dtype=np.float32,
                )
                samples = np.vstack([samples, padding])

            self.audio_position = end_pos

            # Loop audio if we reached the end
            if self.audio_position >= len(self.audio_buffer):
                self.audio_position = 0

        # Create audio frame
        # Convert float32 [-1, 1] to int16
        samples_int16 = (samples * 32767).astype(np.int16)

        frame = AudioFrame.from_ndarray(
            samples_int16, format="s16", layout="stereo"
        )
        frame.sample_rate = self.audio_sample_rate

        # Set PTS for synchronization
        if not hasattr(self, "audio_pts"):
            self.audio_pts = 0
        frame.pts = self.audio_pts
        self.audio_pts += AUDIO_SAMPLES_PER_FRAME

        return frame

    def _create_silence_frame(self) -> AudioFrame:
        """Create a frame of silence."""
        samples = np.zeros((AUDIO_SAMPLES_PER_FRAME, AUDIO_CHANNELS), dtype=np.int16)
        frame = AudioFrame.from_ndarray(samples, format="s16", layout="stereo")
        frame.sample_rate = self.audio_sample_rate

        if not hasattr(self, "audio_pts"):
            self.audio_pts = 0
        frame.pts = self.audio_pts
        self.audio_pts += AUDIO_SAMPLES_PER_FRAME

        return frame

    async def stop(self):
        self._running = False
        await super().stop()
