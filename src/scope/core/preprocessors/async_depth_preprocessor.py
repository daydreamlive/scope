"""Asynchronous depth preprocessor using ZeroMQ for parallel processing.

This module provides a depth preprocessing solution that runs in a separate process,
allowing the main pipeline to process frames in parallel with depth estimation.

Architecture:
    - DepthPreprocessorWorker: Runs in a separate process, loads the depth model,
      and processes frames received via ZeroMQ
    - DepthPreprocessorClient: Runs in the main process, sends frames to the worker
      and receives depth maps asynchronously

Usage:
    # Start the worker process
    client = DepthPreprocessorClient(encoder="vitl")
    client.start()

    # Send frames for processing (non-blocking)
    client.submit_frames(frames, chunk_id=0)

    # Get processed depth maps (returns latest available)
    depth_result = client.get_depth_result()

    # Stop the worker
    client.stop()
"""

import logging
import multiprocessing as mp
import pickle
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ZeroMQ socket configuration
ZMQ_TIMEOUT_MS = 1000  # 1 second timeout for recv
ZMQ_HWM = 100  # High water mark for socket buffers (number of messages)


def _find_free_port() -> int:
    """Find an available port for ZeroMQ socket."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@dataclass
class DepthResult:
    """Container for depth estimation result."""

    chunk_id: int
    depth: Any  # torch.Tensor [1, 3, F, H, W] in [-1, 1]
    timestamp: float


def _run_depth_worker(
    encoder: str,
    input_port: int,
    output_port: int,
    ready_event: mp.Event,
    stop_event: mp.Event,
):
    """Worker process function that runs the depth model.

    This function runs in a separate process and handles:
    - Loading the depth model
    - Receiving frames via ZeroMQ
    - Processing frames through the depth model
    - Sending depth results back via ZeroMQ

    Args:
        encoder: Encoder size ("vits", "vitb", or "vitl")
        input_port: ZeroMQ port for receiving frames
        output_port: ZeroMQ port for sending results
        ready_event: Event to signal when model is loaded
        stop_event: Event to signal shutdown
    """
    import torch
    import zmq

    # Configure logging for worker process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info(f"[DepthWorker] Starting depth worker process (encoder={encoder}, PID={mp.current_process().pid})")

    # Ensure CUDA is initialized in this process
    if torch.cuda.is_available():
        logger.info(f"[DepthWorker] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("[DepthWorker] CUDA not available, falling back to CPU")

    # Import and load the depth model
    from scope.core.preprocessors import VideoDepthAnything

    try:
        logger.info("[DepthWorker] Initializing depth model...")
        depth_model = VideoDepthAnything(
            encoder=encoder,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
        logger.info("[DepthWorker] Loading depth model weights...")
        depth_model.load_model()
        logger.info("[DepthWorker] Depth model loaded successfully!")
    except Exception as e:
        logger.error(f"[DepthWorker] Failed to load depth model: {e}", exc_info=True)
        raise

    # Setup ZeroMQ sockets
    context = zmq.Context()

    # Pull socket for receiving frames
    input_socket = context.socket(zmq.PULL)
    input_socket.setsockopt(zmq.RCVHWM, ZMQ_HWM)  # Set high water mark
    input_socket.bind(f"tcp://*:{input_port}")
    input_socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)

    # Push socket for sending results
    output_socket = context.socket(zmq.PUSH)
    output_socket.setsockopt(zmq.SNDHWM, ZMQ_HWM)  # Set high water mark
    output_socket.bind(f"tcp://*:{output_port}")

    logger.info(
        f"[DepthWorker] ZeroMQ sockets bound (input={input_port}, output={output_port})"
    )

    # Signal that we're ready
    ready_event.set()

    try:
        while not stop_event.is_set():
            try:
                # Receive frame data (non-blocking with timeout)
                message = input_socket.recv()
                data = pickle.loads(message)

                chunk_id = data["chunk_id"]
                frames = data["frames"]  # numpy array [F, H, W, C]
                target_height = data["target_height"]
                target_width = data["target_width"]

                logger.debug(
                    f"[DepthWorker] Received chunk {chunk_id}, "
                    f"frames shape: {frames.shape}"
                )

                # Convert numpy to torch tensor
                frames_tensor = torch.from_numpy(frames).float()

                # Run depth estimation
                start_time = time.time()
                depth = depth_model.infer(frames_tensor)  # [F, H, W]
                inference_time = time.time() - start_time
                num_frames = frames.shape[0]

                logger.info(
                    f"[DepthWorker] Chunk {chunk_id}: {num_frames} frames in "
                    f"{inference_time:.3f}s ({num_frames / inference_time:.1f} FPS)"
                )

                # Post-process depth for VACE
                import torch.nn.functional as F

                F_dim, H, W = depth.shape

                # Resize to target dimensions if needed
                if H != target_height or W != target_width:
                    depth = depth.unsqueeze(1)  # [F, 1, H, W]
                    depth = F.interpolate(
                        depth,
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    depth = depth.squeeze(1)  # [F, H, W]

                # Convert single-channel to 3-channel RGB (replicate)
                depth = depth.unsqueeze(1).repeat(1, 3, 1, 1)  # [F, 3, H, W]

                # Normalize to [-1, 1] for VAE encoding
                depth = depth * 2.0 - 1.0

                # Add batch dimension and rearrange to [1, 3, F, H, W]
                depth = depth.unsqueeze(0).permute(0, 2, 1, 3, 4)

                # Keep as float32 for numpy serialization (numpy doesn't support bfloat16)
                # The client will convert to bfloat16 when moving to GPU
                depth_cpu = depth.float().cpu()

                # Send result
                result = {
                    "chunk_id": chunk_id,
                    "depth": depth_cpu.numpy(),  # Convert to numpy for serialization
                    "timestamp": time.time(),
                }
                output_socket.send(pickle.dumps(result))

                logger.debug(
                    f"[DepthWorker] Sent result for chunk {chunk_id}, "
                    f"depth shape: {depth_cpu.shape}"
                )

            except zmq.error.Again:
                # Timeout, continue to check stop event
                continue
            except Exception as e:
                logger.error(f"[DepthWorker] Error processing frame: {e}")
                continue

    except Exception as e:
        logger.error(f"[DepthWorker] Fatal error: {e}")
    finally:
        logger.info("[DepthWorker] Shutting down...")
        input_socket.close()
        output_socket.close()
        context.term()
        depth_model.offload()
        logger.info("[DepthWorker] Shutdown complete")


class DepthPreprocessorClient:
    """Client for asynchronous depth preprocessing.

    This class manages communication with the depth worker process and provides
    a non-blocking interface for submitting frames and retrieving depth results.

    The client maintains a result buffer that stores the most recent depth results,
    allowing the main pipeline to retrieve depth maps without blocking on computation.

    Args:
        encoder: Encoder size ("vits", "vitb", or "vitl")
        input_port: ZeroMQ port for sending frames to worker
        output_port: ZeroMQ port for receiving results from worker
        result_buffer_size: Maximum number of results to buffer

    Example:
        client = DepthPreprocessorClient(encoder="vitl")
        client.start()

        # Submit frames for processing (returns immediately)
        client.submit_frames(frames, chunk_id=0)

        # Later, retrieve the result (non-blocking)
        result = client.get_depth_result()
        if result is not None:
            depth_tensor = result.depth  # Use in pipeline

        client.stop()
    """

    def __init__(
        self,
        encoder: str = "vitl",
        input_port: int | None = None,
        output_port: int | None = None,
        result_buffer_size: int = 16,  # Increased from 4 for better buffering
    ):
        self.encoder = encoder
        # Ports will be dynamically allocated in start() if not provided
        self.input_port = input_port
        self.output_port = output_port
        self.result_buffer_size = result_buffer_size

        # Process management
        self._worker_process: mp.Process | None = None
        self._ready_event: mp.Event | None = None
        self._stop_event: mp.Event | None = None

        # ZeroMQ sockets (initialized in start())
        self._context = None
        self._input_socket = None
        self._output_socket = None

        # Result buffer
        self._result_buffer: deque[DepthResult] = deque(maxlen=result_buffer_size)
        self._result_lock = threading.Lock()

        # Receiver thread
        self._receiver_thread: threading.Thread | None = None
        self._receiver_running = False

        # Tracking
        self._pending_chunks: set[int] = set()
        self._pending_lock = threading.Lock()
        self._next_chunk_id = 0

        self._started = False

    def start(self, timeout: float = 60.0) -> bool:
        """Start the depth worker process and connect sockets.

        Args:
            timeout: Maximum time to wait for worker to be ready (seconds)

        Returns:
            True if started successfully, False otherwise
        """
        if self._started:
            logger.warning("DepthPreprocessorClient already started")
            return True

        logger.info("Starting DepthPreprocessorClient...")

        # Dynamically allocate ports if not provided
        if self.input_port is None:
            self.input_port = _find_free_port()
        if self.output_port is None:
            self.output_port = _find_free_port()

        logger.info(
            f"Using ports: input={self.input_port}, output={self.output_port}"
        )

        # Use 'spawn' context for CUDA compatibility
        # (CUDA cannot be re-initialized in forked subprocesses)
        ctx = mp.get_context("spawn")

        # Create multiprocessing events using spawn context
        self._ready_event = ctx.Event()
        self._stop_event = ctx.Event()

        # Start the worker process using spawn context
        self._worker_process = ctx.Process(
            target=_run_depth_worker,
            args=(
                self.encoder,
                self.input_port,
                self.output_port,
                self._ready_event,
                self._stop_event,
            ),
            daemon=True,
        )
        self._worker_process.start()

        # Wait for worker to be ready
        logger.info(f"Waiting for depth worker to initialize (timeout={timeout}s)...")
        start_wait = time.time()
        if not self._ready_event.wait(timeout=timeout):
            wait_time = time.time() - start_wait
            logger.error(f"Depth worker failed to start within timeout ({wait_time:.1f}s)")
            self.stop()
            return False

        wait_time = time.time() - start_wait
        logger.info(f"Depth worker ready after {wait_time:.1f}s, connecting sockets...")

        # Setup ZeroMQ client sockets
        import zmq

        self._context = zmq.Context()

        # Push socket for sending frames
        self._input_socket = self._context.socket(zmq.PUSH)
        self._input_socket.setsockopt(zmq.SNDHWM, ZMQ_HWM)  # Set high water mark
        self._input_socket.connect(f"tcp://localhost:{self.input_port}")

        # Pull socket for receiving results
        self._output_socket = self._context.socket(zmq.PULL)
        self._output_socket.setsockopt(zmq.RCVHWM, ZMQ_HWM)  # Set high water mark
        self._output_socket.connect(f"tcp://localhost:{self.output_port}")
        self._output_socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)

        # Start receiver thread
        self._receiver_running = True
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, daemon=True
        )
        self._receiver_thread.start()

        self._started = True
        logger.info("DepthPreprocessorClient started successfully")
        return True

    def stop(self):
        """Stop the depth worker and clean up resources."""
        if not self._started and self._worker_process is None:
            return

        logger.info("Stopping DepthPreprocessorClient...")

        # Stop receiver thread
        self._receiver_running = False
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=2.0)
            self._receiver_thread = None

        # Signal worker to stop
        if self._stop_event is not None:
            self._stop_event.set()

        # Clean up ZeroMQ
        if self._input_socket is not None:
            self._input_socket.close()
            self._input_socket = None
        if self._output_socket is not None:
            self._output_socket.close()
            self._output_socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

        # Wait for worker process
        if self._worker_process is not None:
            self._worker_process.join(timeout=5.0)
            if self._worker_process.is_alive():
                logger.warning("Worker process did not exit gracefully, terminating...")
                self._worker_process.terminate()
                self._worker_process.join(timeout=2.0)
            self._worker_process = None

        # Clear buffers
        with self._result_lock:
            self._result_buffer.clear()
        with self._pending_lock:
            self._pending_chunks.clear()

        self._started = False
        logger.info("DepthPreprocessorClient stopped")

    def submit_frames(
        self,
        frames: "np.ndarray | list",
        target_height: int,
        target_width: int,
    ) -> int:
        """Submit frames for depth processing.

        This method is non-blocking and returns immediately after queuing the frames.
        The chunk_id can be used to track which result corresponds to which input.

        Args:
            frames: Video frames as numpy array [F, H, W, C] in [0, 255] range,
                   or list of tensors that will be converted
            target_height: Target output height for depth maps
            target_width: Target output width for depth maps

        Returns:
            chunk_id: Unique identifier for this submission

        Raises:
            RuntimeError: If client is not started
        """
        if not self._started:
            raise RuntimeError("DepthPreprocessorClient not started")

        import torch

        # Convert list of tensors to numpy array if needed
        if isinstance(frames, list):
            # Assume list of tensors, each (1, H, W, C)
            stacked = torch.cat(frames, dim=0)  # [F, H, W, C]
            frames_np = stacked.cpu().numpy()
        elif isinstance(frames, torch.Tensor):
            frames_np = frames.cpu().numpy()
        else:
            frames_np = frames

        # Ensure correct dtype
        if frames_np.dtype != np.uint8:
            if frames_np.max() <= 1.0:
                frames_np = (frames_np * 255).astype(np.uint8)
            else:
                frames_np = frames_np.astype(np.uint8)

        # Assign chunk ID
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1

        # Track pending chunk
        with self._pending_lock:
            self._pending_chunks.add(chunk_id)

        # Send to worker
        data = {
            "chunk_id": chunk_id,
            "frames": frames_np,
            "target_height": target_height,
            "target_width": target_width,
        }
        self._input_socket.send(pickle.dumps(data))

        logger.debug(
            f"Submitted chunk {chunk_id}, frames shape: {frames_np.shape}, "
            f"target: {target_height}x{target_width}"
        )
        return chunk_id

    def get_depth_result(self, wait: bool = False, timeout: float = 5.0) -> DepthResult | None:
        """Get the latest depth result.

        Args:
            wait: If True, wait for a result if none available
            timeout: Maximum time to wait (seconds) if wait=True

        Returns:
            DepthResult with depth tensor, or None if no result available
        """
        import torch

        if wait:
            start_time = time.time()
            while time.time() - start_time < timeout:
                with self._result_lock:
                    if self._result_buffer:
                        result_dict = self._result_buffer.popleft()
                        # Convert numpy back to torch tensor
                        depth_tensor = torch.from_numpy(result_dict["depth"])
                        return DepthResult(
                            chunk_id=result_dict["chunk_id"],
                            depth=depth_tensor,
                            timestamp=result_dict["timestamp"],
                        )
                time.sleep(0.01)
            return None

        with self._result_lock:
            if self._result_buffer:
                result_dict = self._result_buffer.popleft()
                # Convert numpy back to torch tensor
                depth_tensor = torch.from_numpy(result_dict["depth"])
                return DepthResult(
                    chunk_id=result_dict["chunk_id"],
                    depth=depth_tensor,
                    timestamp=result_dict["timestamp"],
                )
        return None

    def get_latest_depth_result(self) -> DepthResult | None:
        """Get the most recent depth result, discarding older ones.

        Returns:
            Most recent DepthResult, or None if no results available
        """
        import torch

        with self._result_lock:
            if not self._result_buffer:
                return None

            # Get the latest result
            result_dict = self._result_buffer[-1]

            # Clear all older results
            self._result_buffer.clear()

            # Convert numpy back to torch tensor
            depth_tensor = torch.from_numpy(result_dict["depth"])
            return DepthResult(
                chunk_id=result_dict["chunk_id"],
                depth=depth_tensor,
                timestamp=result_dict["timestamp"],
            )

    def has_pending_results(self) -> bool:
        """Check if there are pending chunks being processed."""
        with self._pending_lock:
            return len(self._pending_chunks) > 0

    def get_buffer_size(self) -> int:
        """Get the number of results currently in buffer."""
        with self._result_lock:
            return len(self._result_buffer)

    def is_running(self) -> bool:
        """Check if the client is running."""
        return self._started and self._worker_process is not None and self._worker_process.is_alive()

    def _receiver_loop(self):
        """Background thread that receives results from the worker."""
        logger.info("Depth result receiver thread started")

        while self._receiver_running:
            try:
                # Receive result (with timeout to allow checking running flag)
                import zmq

                message = self._output_socket.recv()
                result = pickle.loads(message)

                chunk_id = result["chunk_id"]

                # Remove from pending
                with self._pending_lock:
                    self._pending_chunks.discard(chunk_id)

                # Add to result buffer
                with self._result_lock:
                    self._result_buffer.append(result)

                logger.debug(
                    f"Received depth result for chunk {chunk_id}, "
                    f"buffer size: {len(self._result_buffer)}"
                )

            except Exception as e:
                # zmq.error.Again is expected on timeout
                if "Again" not in str(type(e)):
                    logger.error(f"Error in receiver loop: {e}")
                continue

        logger.info("Depth result receiver thread stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
