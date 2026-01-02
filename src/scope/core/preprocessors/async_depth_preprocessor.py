"""Asynchronous preprocessor using ZeroMQ for parallel processing.

This module provides a preprocessing solution that runs in a completely
separate process (not subprocess), allowing the main pipeline to process frames
in parallel with preprocessing while maintaining complete CUDA context isolation.

Architecture:
    - preprocessor_worker_process.py: Runs as an independent process via subprocess.Popen,
      loads the preprocessor pipeline, and processes frames received via ZeroMQ
    - AsyncPreprocessorClient: Runs in the main process, sends frames to the worker
      and receives preprocessed results asynchronously

Usage:
    # Start the worker process
    client = AsyncPreprocessorClient(preprocessor_type="depthanything", encoder="vits")
    client.start()

    # Send frames for processing (non-blocking)
    client.submit_frames(frames, target_height=512, target_width=512)

    # Get processed results (returns latest available)
    result = client.get_latest_result()

    # Stop the worker
    client.stop()
"""

import logging
import pickle
import signal
import subprocess
import sys
import tempfile
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
class PreprocessorResult:
    """Container for preprocessor result."""

    chunk_id: int
    data: Any  # torch.Tensor [1, C, F, H, W] in [-1, 1]
    timestamp: float




class AsyncPreprocessorClient:
    """Client for asynchronous preprocessing.

    This class manages communication with the preprocessor worker process and provides
    a non-blocking interface for submitting frames and retrieving preprocessed results.

    The worker runs as a completely separate process (via subprocess.Popen) for
    complete CUDA context isolation from the main pipeline.

    The client maintains a result buffer that stores the most recent preprocessor results,
    allowing the main pipeline to retrieve preprocessed frames without blocking on computation.

    Args:
        preprocessor_type: Type of preprocessor ("depthanything", "passthrough", etc.)
        encoder: For depthanything, encoder size ("vits", "vitb", or "vitl"). Ignored for other types.
        input_port: ZeroMQ port for sending frames to worker
        output_port: ZeroMQ port for receiving results from worker
        result_buffer_size: Maximum number of results to buffer

    Example:
        client = AsyncPreprocessorClient(preprocessor_type="depthanything", encoder="vits")
        client.start()

        # Submit frames for processing (returns immediately)
        client.submit_frames(frames, target_height=512, target_width=512)

        # Later, retrieve the result (non-blocking)
        result = client.get_latest_result()
        if result is not None:
            preprocessed_tensor = result.data  # Use in pipeline

        client.stop()
    """

    def __init__(
        self,
        preprocessor_type: str,
        encoder: str | None = None,
        input_port: int | None = None,
        output_port: int | None = None,
        result_buffer_size: int = 16,  # Increased from 4 for better buffering
    ):
        self.preprocessor_type = preprocessor_type
        self.encoder = encoder or "vits"  # Default encoder for depthanything
        # Ports will be dynamically allocated in start() if not provided
        self.input_port = input_port
        self.output_port = output_port
        self.result_buffer_size = result_buffer_size

        # Process management (using subprocess.Popen for complete process isolation)
        self._worker_process: subprocess.Popen | None = None
        self._ready_file: Path | None = None  # File-based ready signal
        self._temp_dir: tempfile.TemporaryDirectory | None = None

        # ZeroMQ sockets (initialized in start())
        self._context = None
        self._input_socket = None
        self._output_socket = None

        # Result buffer - stores PreprocessorResult with torch tensors (converted in receiver thread)
        self._result_buffer: deque[PreprocessorResult] = deque(maxlen=result_buffer_size)
        self._result_lock = threading.Lock()

        # Receiver thread
        self._receiver_thread: threading.Thread | None = None
        self._receiver_running = False

        # Tracking
        self._pending_chunks: set[int] = set()
        self._pending_lock = threading.Lock()
        self._next_chunk_id = 0

        self._started = False

        # Throttling: track consumption rate to slow down worker when buffer is full
        self._last_result_consumed_time = 0.0
        self._target_fps: float | None = None  # Target FPS from main pipeline
        self._throttle_lock = threading.Lock()

    def start(self, timeout: float = 60.0) -> bool:
        """Start the preprocessor worker process and connect sockets.

        The worker is started as a completely separate process using subprocess.Popen,
        ensuring complete CUDA context isolation from the main pipeline.

        Args:
            timeout: Maximum time to wait for worker to be ready (seconds)

        Returns:
            True if started successfully, False otherwise
        """
        if self._started:
            logger.warning("AsyncPreprocessorClient already started")
            return True

        logger.info(f"Starting AsyncPreprocessorClient for {self.preprocessor_type}...")

        # Dynamically allocate ports if not provided
        if self.input_port is None:
            self.input_port = _find_free_port()
        if self.output_port is None:
            self.output_port = _find_free_port()

        logger.info(
            f"Using ports: input={self.input_port}, output={self.output_port}"
        )

        # Create temporary directory for ready file
        self._temp_dir = tempfile.TemporaryDirectory(prefix="preprocessor_worker_")
        self._ready_file = Path(self._temp_dir.name) / "ready"

        # Start the worker as a completely separate process using subprocess.Popen
        # This ensures complete CUDA context isolation
        worker_module = "scope.core.preprocessors.preprocessor_worker_process"
        cmd = [
            sys.executable, "-m", worker_module,
            "--preprocessor-type", self.preprocessor_type,
            "--input-port", str(self.input_port),
            "--output-port", str(self.output_port),
            "--ready-file", str(self._ready_file),
        ]
        # Add encoder argument only for depthanything
        if self.preprocessor_type == "depthanything":
            cmd.extend(["--encoder", self.encoder])

        logger.info(f"Starting separate worker process: {' '.join(cmd)}")

        # Start the worker process
        self._worker_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Start a thread to forward worker stdout to our logger
        self._stdout_thread = threading.Thread(
            target=self._forward_stdout, daemon=True
        )
        self._stdout_thread.start()

        logger.info(f"Worker process started (PID={self._worker_process.pid})")

        # Wait for worker to be ready (file-based signaling)
        logger.info(f"Waiting for preprocessor worker to initialize (timeout={timeout}s)...")
        start_wait = time.time()

        while time.time() - start_wait < timeout:
            # Check if process died
            if self._worker_process.poll() is not None:
                wait_time = time.time() - start_wait
                logger.error(
                    f"Worker process died during startup (exit code: {self._worker_process.returncode}) "
                    f"after {wait_time:.1f}s"
                )
                self.stop()
                return False

            # Check if ready file exists
            if self._ready_file.exists():
                break

            time.sleep(0.1)
        else:
            wait_time = time.time() - start_wait
            logger.error(f"Preprocessor worker failed to start within timeout ({wait_time:.1f}s)")
            self.stop()
            return False

        wait_time = time.time() - start_wait
        logger.info(f"Preprocessor worker ready after {wait_time:.1f}s, connecting sockets...")

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
        logger.info("AsyncPreprocessorClient started successfully")
        return True

    def _forward_stdout(self):
        """Forward worker stdout to our logger."""
        try:
            if self._worker_process and self._worker_process.stdout:
                for line in self._worker_process.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info(f"[DepthWorker] {line}")
        except Exception:
            pass  # Process closed

    def stop(self):
        """Stop the preprocessor worker and clean up resources."""
        if not self._started and self._worker_process is None:
            return

        logger.info("Stopping AsyncPreprocessorClient...")

        # Stop receiver thread
        self._receiver_running = False
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=2.0)
            self._receiver_thread = None

        # Clean up ZeroMQ first (so worker recv() fails and can notice SIGTERM)
        if self._input_socket is not None:
            self._input_socket.close()
            self._input_socket = None
        if self._output_socket is not None:
            self._output_socket.close()
            self._output_socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

        # Signal worker to stop using SIGTERM
        if self._worker_process is not None:
            if self._worker_process.poll() is None:  # Still running
                logger.info(f"Sending SIGTERM to worker process (PID={self._worker_process.pid})...")
                try:
                    self._worker_process.send_signal(signal.SIGTERM)
                except OSError:
                    pass  # Process already dead

                # Wait for graceful shutdown
                try:
                    self._worker_process.wait(timeout=5.0)
                    logger.info(f"Worker process exited with code {self._worker_process.returncode}")
                except subprocess.TimeoutExpired:
                    logger.warning("Worker process did not exit gracefully, killing...")
                    self._worker_process.kill()
                    try:
                        self._worker_process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass

            self._worker_process = None

        # Clean up stdout forwarding thread
        if hasattr(self, '_stdout_thread') and self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1.0)
            self._stdout_thread = None

        # Clean up temp directory
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None
            self._ready_file = None

        # Clear buffers
        with self._result_lock:
            self._result_buffer.clear()
        with self._pending_lock:
            self._pending_chunks.clear()

        self._started = False
        logger.info("AsyncPreprocessorClient stopped")

    def submit_frames(
        self,
        frames: "np.ndarray | list",
        target_height: int,
        target_width: int,
    ) -> int | None:
        """Submit frames for preprocessing.

        This method is non-blocking and returns immediately after queuing the frames.
        The chunk_id can be used to track which result corresponds to which input.

        Implements throttling: if buffer is nearly full, skips submission to create
        backpressure and reduce GPU contention with main pipeline.

        Args:
            frames: Video frames as numpy array [F, H, W, C] in [0, 255] range,
                   or list of tensors that will be converted
            target_height: Target output height for preprocessed frames
            target_width: Target output width for preprocessed frames

        Returns:
            chunk_id: Unique identifier for this submission, or None if skipped due to throttling

        Raises:
            RuntimeError: If client is not started
        """
        if not self._started:
            raise RuntimeError("AsyncPreprocessorClient not started")

        import torch

        # Throttling: skip submission if buffer is getting full
        # This reduces GPU contention with main pipeline
        with self._result_lock:
            buffer_size = len(self._result_buffer)

        # Skip if buffer is more than half full (preprocessor is ahead of consumer)
        if buffer_size > self.result_buffer_size // 2:
            logger.debug(
                f"Throttling preprocessor submission, buffer {buffer_size}/{self.result_buffer_size}"
            )
            return None

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

    def get_result(self, wait: bool = False, timeout: float = 5.0) -> PreprocessorResult | None:
        """Get the latest preprocessor result.

        Args:
            wait: If True, wait for a result if none available
            timeout: Maximum time to wait (seconds) if wait=True

        Returns:
            PreprocessorResult with preprocessed tensor, or None if no result available
        """
        if wait:
            start_time = time.time()
            while time.time() - start_time < timeout:
                with self._result_lock:
                    if self._result_buffer:
                        # Buffer now stores PreprocessorResult directly (tensor already converted)
                        result = self._result_buffer.popleft()
                        self._last_result_consumed_time = time.time()
                        return result
                time.sleep(0.01)
            return None

        with self._result_lock:
            if self._result_buffer:
                # Buffer now stores PreprocessorResult directly (tensor already converted)
                result = self._result_buffer.popleft()
                self._last_result_consumed_time = time.time()
                return result
        return None

    def get_latest_result(self) -> PreprocessorResult | None:
        """Get the most recent preprocessor result, discarding older ones.

        Returns:
            Most recent PreprocessorResult, or None if no results available
        """
        with self._result_lock:
            if not self._result_buffer:
                return None

            # Get the latest result (tensor already converted in receiver thread)
            result = self._result_buffer[-1]

            # Clear all older results
            self._result_buffer.clear()

            self._last_result_consumed_time = time.time()
            return result

    # Backward compatibility aliases
    def get_depth_result(self, wait: bool = False, timeout: float = 5.0) -> PreprocessorResult | None:
        """Backward compatibility alias for get_result()."""
        return self.get_result(wait, timeout)

    def get_latest_depth_result(self) -> PreprocessorResult | None:
        """Backward compatibility alias for get_latest_result()."""
        return self.get_latest_result()

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
        return (
            self._started
            and self._worker_process is not None
            and self._worker_process.poll() is None  # poll() returns None if still running
        )

    def set_target_fps(self, fps: float | None):
        """Set target FPS to throttle depth processing to match main pipeline.

        When the depth worker is running much faster than the main pipeline,
        this creates backpressure to avoid wasting GPU resources.

        Args:
            fps: Target FPS (e.g., from main pipeline), or None to disable throttling
        """
        with self._throttle_lock:
            self._target_fps = fps
            if fps is not None:
                logger.debug(f"Depth throttle target set to {fps:.1f} FPS")

    def _receiver_loop(self):
        """Background thread that receives results from the worker.

        Performs numpy→torch conversion here (async) to avoid blocking main thread.
        Also implements throttling by sleeping when buffer is full.
        """
        import torch

        logger.info("Depth result receiver thread started")

        while self._receiver_running:
            try:
                # Throttle: if buffer is getting full, slow down to match consumption rate
                with self._result_lock:
                    buffer_size = len(self._result_buffer)

                if buffer_size >= self.result_buffer_size - 1:
                    # Buffer nearly full, sleep to allow consumer to catch up
                    # This creates backpressure to the worker
                    time.sleep(0.05)
                    continue

                # Receive result (with timeout to allow checking running flag)
                import zmq

                message = self._output_socket.recv()
                result = pickle.loads(message)

                chunk_id = result["chunk_id"]

                # Remove from pending
                with self._pending_lock:
                    self._pending_chunks.discard(chunk_id)

                # Convert numpy to torch tensor HERE (async, not in main thread)
                # This is the expensive operation we want to do in background
                preprocessed_tensor = torch.from_numpy(result["data"])

                # Pin memory for faster GPU transfer when .to(device) is called later
                # This makes the subsequent cuda() call non-blocking and faster
                if torch.cuda.is_available():
                    preprocessed_tensor = preprocessed_tensor.pin_memory()

                # Create PreprocessorResult with tensor already converted
                preprocessor_result = PreprocessorResult(
                    chunk_id=chunk_id,
                    data=preprocessed_tensor,
                    timestamp=result["timestamp"],
                )

                # Add to result buffer
                with self._result_lock:
                    self._result_buffer.append(preprocessor_result)

                logger.debug(
                    f"Received preprocessor result for chunk {chunk_id}, "
                    f"buffer size: {len(self._result_buffer)}"
                )

            except Exception as e:
                # zmq.error.Again is expected on timeout
                if "Again" not in str(type(e)):
                    logger.error(f"Error in receiver loop: {e}")
                continue

        logger.info("Preprocessor result receiver thread stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
