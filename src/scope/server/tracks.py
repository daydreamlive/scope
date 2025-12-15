import asyncio
import fractions
import logging
import threading
import time

from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import AudioFrame, VideoFrame
import numpy as np
import torch

from .frame_processor import FrameProcessor
from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


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
    kind = "audio"

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        initial_parameters: dict | None = None,
        notification_callback: callable | None = None,
        chunk_size: int = 960,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.parameters = initial_parameters or {}
        self.notification_callback = notification_callback

        self.chunk_size = chunk_size
        self.sample_rate = 48_000
        self.time_base = fractions.Fraction(1, self.sample_rate)
        self.timestamp = 0

        self._prepared = False
        self._reset_requested = True
        self._lock = threading.Lock()
        self._paused = False
        self._stop_notified = False

    def _prepare_pipeline(self):
        pipeline = self.pipeline_manager.get_pipeline()
        pipeline.prepare(**self.parameters, chunk_size=self.chunk_size)
        self.sample_rate = getattr(pipeline, "sample_rate", self.sample_rate)
        self.time_base = fractions.Fraction(1, self.sample_rate)
        self.timestamp = 0
        self._prepared = True

    def update_parameters(self, params: dict):
        with self._lock:
            self.parameters.update(params)
            if "chunk_size" in params and isinstance(params["chunk_size"], int):
                self.chunk_size = params["chunk_size"]
            self._reset_requested = True

    def pause(self, paused: bool):
        self._paused = paused

    def _next_chunk(self) -> np.ndarray | None:
        with self._lock:
            if self._reset_requested or not self._prepared:
                self._prepare_pipeline()
                self._reset_requested = False

            pipeline = self.pipeline_manager.get_pipeline()
            try:
                chunk_tensor = pipeline(
                    chunk_size=self.chunk_size,
                    **self.parameters,
                )
            except Exception as exc:
                logger.error("AudioProcessingTrack: pipeline error %s", exc)
                return None

            if chunk_tensor is None:
                return None

            chunk_tensor = torch.as_tensor(chunk_tensor)
            if chunk_tensor.numel() == 0:
                return None

            chunk_np = chunk_tensor.detach().cpu().numpy()
            chunk_np = np.squeeze(chunk_np)
            chunk_np = np.clip(chunk_np, -1.0, 1.0)
            if chunk_np.dtype != np.int16:
                chunk_np = (chunk_np * 32767.0).astype(np.int16)

            return chunk_np

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._paused:
            await asyncio.sleep(self.chunk_size / self.sample_rate)
            return self._silence_frame()

        loop = asyncio.get_event_loop()
        chunk = await loop.run_in_executor(None, self._next_chunk)

        if chunk is None or chunk.size == 0:
            if not self._stop_notified and self.notification_callback:
                try:
                    self.notification_callback({"type": "stream_stopped"})
                finally:
                    self._stop_notified = True
            raise MediaStreamError("Audio stream ended")

        frame = AudioFrame.from_ndarray(
            chunk.reshape(-1, 1), format="s16", layout="mono"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self.timestamp
        frame.time_base = self.time_base
        self.timestamp += chunk.shape[0]

        await asyncio.sleep(chunk.shape[0] / self.sample_rate)
        return frame

    def _silence_frame(self) -> AudioFrame:
        silence = np.zeros(self.chunk_size, dtype=np.int16)
        frame = AudioFrame.from_ndarray(
            silence.reshape(-1, 1), format="s16", layout="mono"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self.timestamp
        frame.time_base = self.time_base
        self.timestamp += silence.shape[0]
        return frame
