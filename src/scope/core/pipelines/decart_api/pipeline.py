import asyncio
import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from ..interface import Pipeline, Requirements
from ..schema import DecartApiConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

try:
    from decart import DecartClient
    from decart import models as decart_models
    from decart.realtime.client import RealtimeClient
    from decart.realtime.types import RealtimeConnectOptions, ModelState
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
    from av import VideoFrame

    DECART_AVAILABLE = True
    REALTIME_AVAILABLE = True
except ImportError as e:
    DECART_AVAILABLE = False
    REALTIME_AVAILABLE = False
    logger.warning(
        f"Decart SDK not available: {e}. Install with: pip install decart"
    )


class FrameSourceTrack(MediaStreamTrack):
    """Custom MediaStreamTrack that feeds frames from a queue to Decart API."""

    kind = "video"

    def __init__(self, frame_queue: queue.Queue, fps: int = 22, width: int = 1280, height: int = 704):
        super().__init__()
        self.frame_queue = frame_queue
        self.fps = fps
        self.width = width
        self.height = height
        self.frame_ptime = 1.0 / fps
        self.timestamp = 0
        self.start_time = None
        self.last_frame_time = None

    async def recv(self) -> VideoFrame:
        """Return the next frame from the queue."""
        # Wait for a frame to be available
        try:
            frame_np = self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            # If no frame available, create a black frame
            # Use configured dimensions
            frame_np = np.zeros(
                (self.height, self.width, 3), dtype=np.uint8
            )  # Use configured dimensions

        # Convert numpy array to VideoFrame
        frame = VideoFrame.from_ndarray(frame_np, format="rgb24")

        # Set timestamp
        if self.start_time is None:
            self.start_time = time.time()
            self.last_frame_time = self.start_time
            self.timestamp = 0
        else:
            current_time = time.time()
            time_since_last = current_time - self.last_frame_time
            wait_time = self.frame_ptime - time_since_last
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.timestamp += int(self.frame_ptime * VIDEO_CLOCK_RATE)
            self.last_frame_time = time.time()

        frame.pts = self.timestamp
        frame.time_base = VIDEO_TIME_BASE
        return frame


class DecartApiPipeline(Pipeline):
    """Pipeline that processes video frames through Decart API."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return DecartApiConfig

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if not DECART_AVAILABLE:
            raise ImportError(
                "Decart SDK is required. Install with: pip install decart"
            )

        self.height = config.height
        self.width = config.width
        if device is not None:
            self.device = device
        else:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_name)
        self.dtype = dtype

        # Get API key from environment
        api_key = os.getenv("DECART_API_KEY")
        if not api_key:
            raise ValueError(
                "DECART_API_KEY environment variable is required"
            )

        # Initialize Decart client
        self.client = DecartClient(api_key=api_key)

        # Get the model from environment variable, default to mirage_v2
        model_name = os.getenv("DECART_MODEL", "mirage_v2")
        logger.info(f"Using Decart model: {model_name}")
        self.model = decart_models.realtime(model_name)

        # Store current prompt
        self.current_prompt = None
        # Track last prompt we saw in prompts parameter to detect stale values
        self.last_prompts_value = None

        # WebRTC connection state
        self.realtime_client = None
        self.connection_established = False
        self.async_loop = None
        self.async_thread = None
        self.shutdown_event = threading.Event()

        # Queues for frame exchange between sync and async worlds
        self.input_frame_queue = queue.Queue(maxsize=10)
        self.output_frame_queue = queue.Queue(maxsize=10)

        # MediaStreamTrack for feeding frames to Decart
        self.local_track = None

        # Start async thread for WebRTC connection
        if REALTIME_AVAILABLE:
            self._start_async_thread()
            # Wait for connection to establish (with timeout)
            max_wait = 10.0
            wait_interval = 0.1
            waited = 0.0
            while (
                not self.connection_established
                and waited < max_wait
                and not self.shutdown_event.is_set()
            ):
                time.sleep(wait_interval)
                waited += wait_interval
            if self.connection_established:
                logger.info("Decart connection established")
            else:
                logger.warning(
                    f"Decart connection not established after {waited}s, "
                    "will use passthrough mode"
                )

        logger.info("DecartApiPipeline initialized")

    def _start_async_thread(self):
        """Start background thread for async WebRTC operations."""
        def run_async_loop():
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)
            try:
                self.async_loop.run_until_complete(self._async_worker())
            except Exception as e:
                logger.error(f"Async worker error: {e}", exc_info=True)

        self.async_thread = threading.Thread(
            target=run_async_loop, daemon=True
        )
        self.async_thread.start()

    async def _async_worker(self):
        """Async worker that manages WebRTC connection and frame processing."""
        try:
            # Create local track for feeding frames to Decart
            # Use config resolution to preserve aspect ratio (e.g., square input -> square output)
            logger.info(
                f"Creating FrameSourceTrack with config resolution: "
                f"{self.width}x{self.height} @ {self.model.fps}fps"
            )
            self.local_track = FrameSourceTrack(
                self.input_frame_queue,
                fps=self.model.fps,
                width=self.width,
                height=self.height,
            )

            # Set up callback for receiving processed frames
            def on_remote_stream(remote_track: MediaStreamTrack):
                """Callback when remote stream is available."""
                logger.info(
                    f"Remote stream received from Decart: {remote_track}"
                )
                self.connection_established = True
                # Start receiving frames from remote track
                receive_task = asyncio.create_task(
                    self._receive_remote_frames(remote_track)
                )
                # Store task to prevent garbage collection
                if not hasattr(self, '_receive_tasks'):
                    self._receive_tasks = []
                self._receive_tasks.append(receive_task)

            # Create connection options
            initial_state = None
            if self.current_prompt:
                from decart.types import Prompt
                initial_state = ModelState(
                    prompt=Prompt(text=self.current_prompt, enrich=True)
                )

            options = RealtimeConnectOptions(
                model=self.model,
                on_remote_stream=on_remote_stream,
                initial_state=initial_state,
            )

            # Connect to Decart realtime API
            logger.info("Connecting to Decart realtime API...")
            self.realtime_client = await RealtimeClient.connect(
                base_url=self.client.base_url,
                api_key=self.client.api_key,
                local_track=self.local_track,
                options=options,
            )
            logger.info("Connected to Decart realtime API")

            # Keep connection alive
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Error in async worker: {e}", exc_info=True)
            self.connection_established = False

    async def _receive_remote_frames(self, remote_track: MediaStreamTrack):
        """Receive processed frames from Decart and queue them."""
        logger.info("Starting to receive remote frames from Decart")
        frame_count = 0
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Receive frame from Decart
                    frame = await asyncio.wait_for(
                        remote_track.recv(), timeout=1.0
                    )
                    # Convert VideoFrame to numpy array
                    frame_np = frame.to_ndarray(format="rgb24")
                    frame_count += 1
                    if frame_count % 30 == 0:
                        logger.debug(
                            f"Received {frame_count} frames from Decart"
                        )
                    # Put in output queue
                    try:
                        self.output_frame_queue.put_nowait(frame_np)
                    except queue.Full:
                        # Drop oldest frame if queue is full
                        try:
                            self.output_frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.output_frame_queue.put_nowait(frame_np)
                except asyncio.TimeoutError:
                    logger.debug("Timeout waiting for remote frame")
                    continue
                except Exception as e:
                    logger.error(f"Error receiving remote frame: {e}")
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in receive_remote_frames: {e}", exc_info=True)

    def prepare(self, **kwargs) -> Requirements:
        """Return input requirements for video mode."""
        # Process one frame at a time for realtime
        return Requirements(input_size=1)

    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process video frames through Decart API.

        Args:
            **kwargs: Pipeline parameters including:
                - video: Input video frames (list of tensors or tensor)
                - prompts: Text prompt for style transformation
                - transition: Optional transition dict with target_prompts
                  (takes precedence over prompts if provided)

        Returns:
            Processed frames as tensor in THWC format [0, 1] range
        """
        input_video = kwargs.get("video")
        prompts = kwargs.get("prompts")
        transition = kwargs.get("transition")

        if input_video is None:
            raise ValueError(
                "Input video cannot be None for DecartApiPipeline"
            )

        # Convert input to list of tensors if needed
        if isinstance(input_video, list):
            tensor_frames = input_video
        else:
            # Assume it's a BCTHW tensor, convert to list
            # Rearrange to THWC and split into frames
            if len(input_video.shape) == 5:  # BCTHW
                input_video = rearrange(input_video, "B C T H W -> B T C H W")
                input_video = input_video.squeeze(0)  # Remove batch dim -> T C H W
            input_video = rearrange(input_video, "T C H W -> T H W C")
            # Convert to list of (1, H, W, C) tensors
            tensor_frames = [
                input_video[i].unsqueeze(0)
                for i in range(input_video.shape[0])
            ]

        # Extract prompt text - simple: check transition first, then prompts
        # But only update if it's actually different from current
        prompt_text = None

        # Get prompt from transition.target_prompts if available
        if transition is not None:
            target_prompts = transition.get("target_prompts")
            if target_prompts and len(target_prompts) > 0:
                first_prompt = target_prompts[0]
                extracted = first_prompt.get("text", "")
                # Only use if different from current (prevents duplicate updates)
                if extracted != self.current_prompt:
                    prompt_text = extracted
                    logger.info(
                        f"Using prompt from transition: '{prompt_text}' "
                        f"(current: '{self.current_prompt}')"
                    )


        # Update prompt if we have a new one
        if prompt_text:
            logger.info(
                f"Prompt change detected: '{self.current_prompt}' -> "
                f"'{prompt_text}'"
            )
            self._update_prompt(prompt_text)

        # Process frames through Decart API
        # Decart's realtime API uses WebRTC streams with async handling.
        processed_frames = []
        for frame_tensor in tensor_frames:
            # Convert tensor to numpy array (H, W, C) in [0, 255] range
            # frame_tensor is (1, H, W, C) from the list
            frame_np = frame_tensor.squeeze(0).cpu().numpy()  # (H, W, C)

            # Ensure it's 3D (H, W, C)
            if frame_np.ndim == 2:
                # If grayscale, convert to RGB
                frame_np = np.stack([frame_np] * 3, axis=-1)
            elif frame_np.ndim != 3:
                raise ValueError(f"Unexpected frame shape: {frame_np.shape}")

            # Normalize to [0, 255] if needed
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255.0).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

            # Resize if needed to match configured output resolution
            # Use config resolution to preserve aspect ratio (e.g., square input -> square output)
            needs_resize = (
                frame_np.shape[0] != self.height
                or frame_np.shape[1] != self.width
            )
            if needs_resize:
                img = Image.fromarray(frame_np)
                img = img.resize(
                    (self.width, self.height),
                    Image.Resampling.LANCZOS
                )
                frame_np = np.array(img)
                # Ensure it's still 3D after resize
                if frame_np.ndim == 2:
                    frame_np = np.stack([frame_np] * 3, axis=-1)

            # Send frame to Decart API via WebRTC
            processed_frame_np = self._process_frame_with_decart(
                frame_np, prompt_text
            )

            # Convert processed frame back to tensor
            # processed_frame_np is (H, W, C)
            processed_frame = (
                torch.from_numpy(processed_frame_np).float() / 255.0
            )
            # processed_frame is now (H, W, C) - ensure it's 3D
            if processed_frame.ndim != 3:
                raise ValueError(
                    f"Processed frame must be 3D (H, W, C), "
                    f"got {processed_frame.shape}"
                )
            processed_frames.append(processed_frame)

        # Stack frames and return in THWC format
        # Always return (T, H, W, C) format where T is the number of frames
        if len(processed_frames) == 1:
            # For single frame, ensure we have time dimension
            output = processed_frames[0].unsqueeze(0)  # (1, H, W, C)
        else:
            output = torch.stack(processed_frames, dim=0)  # (T, H, W, C)

        return output

    def _process_frame_with_decart(
        self, frame_np: np.ndarray, prompt_text: str | None
    ) -> np.ndarray:
        """Process a frame through Decart API or pass through."""
        if REALTIME_AVAILABLE and self.connection_established:
            # Note: Prompt updates are now handled in __call__ before
            # processing frames, so we don't need to check here

            # Send frame to input queue for WebRTC stream
            try:
                self.input_frame_queue.put_nowait(frame_np)
            except queue.Full:
                # Drop oldest frame if queue is full
                try:
                    self.input_frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.input_frame_queue.put_nowait(frame_np)

            # Get processed frame from output queue
            try:
                processed_frame_np = self.output_frame_queue.get(timeout=2.0)
                # Resize output frame to match configured resolution
                # Decart might return frames at model resolution, so ensure output matches config
                if (processed_frame_np.shape[0] != self.height or
                    processed_frame_np.shape[1] != self.width):
                    img = Image.fromarray(processed_frame_np)
                    img = img.resize(
                        (self.width, self.height),
                        Image.Resampling.LANCZOS
                    )
                    processed_frame_np = np.array(img)
            except queue.Empty:
                # Timeout - use input frame as fallback
                logger.warning(
                    "Timeout waiting for processed frame from Decart, "
                    "using input frame"
                )
                processed_frame_np = frame_np
        else:
            # Fallback: pass through if realtime not available
            if not REALTIME_AVAILABLE:
                logger.debug(
                    "Realtime API not available, passing through"
                )
            elif not self.connection_established:
                logger.debug(
                    "Connection not established yet, passing through"
                )
            processed_frame_np = frame_np

        return processed_frame_np

    def _update_prompt(self, prompt_text: str):
        """Update prompt in Decart API."""
        if not self.realtime_client:
            logger.warning(
                "Cannot update prompt: realtime_client is not initialized"
            )
            return
        if not self.async_loop:
            logger.warning(
                "Cannot update prompt: async_loop is not available"
            )
            return

        try:
            logger.info(f"Calling set_prompt with: '{prompt_text}'")
            # set_prompt expects a string, not a Prompt object
            # Based on the error, it calls .strip() on the prompt
            future = asyncio.run_coroutine_threadsafe(
                self.realtime_client.set_prompt(prompt_text),
                self.async_loop,
            )
            # Increase timeout to 5 seconds as API calls may take longer
            future.result(timeout=5.0)
            self.current_prompt = prompt_text
            logger.info(f"Successfully updated prompt to: '{prompt_text}'")
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout updating prompt to '{prompt_text}' "
                "(API call took > 5 seconds)"
            )
        except Exception as e:
            logger.error(
                f"Failed to update prompt to '{prompt_text}': {e}",
                exc_info=True
            )

    def __del__(self):
        """Cleanup on pipeline destruction."""
        # Signal shutdown
        if hasattr(self, 'shutdown_event'):
            self.shutdown_event.set()

        # Cleanup async connection
        if hasattr(self, 'async_loop') and self.async_loop:
            if hasattr(self, 'realtime_client') and self.realtime_client:
                try:
                    # Schedule disconnect in async loop
                    asyncio.run_coroutine_threadsafe(
                        self._disconnect_async(), self.async_loop
                    )
                except Exception as e:
                    logger.warning(f"Error disconnecting: {e}")

        # Cleanup Decart client
        if hasattr(self, 'client') and self.client:
            try:
                # Client cleanup will happen when it goes out of scope
                pass
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

    async def _disconnect_async(self):
        """Async cleanup method."""
        if hasattr(self, 'realtime_client') and self.realtime_client:
            try:
                # Disconnect realtime client
                # Note: RealtimeClient might have a disconnect method
                if hasattr(self.realtime_client, 'disconnect'):
                    await self.realtime_client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting realtime client: {e}")
