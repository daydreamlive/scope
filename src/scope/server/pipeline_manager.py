"""Pipeline Manager for lazy loading and managing ML pipelines."""

import asyncio
import logging
import multiprocessing as mp
import os
import threading
from enum import Enum
from typing import Any

from .pipeline_proxy import PipelineProxy
from .pipeline_worker import WorkerCommand, WorkerResponse, pipeline_worker_process

logger = logging.getLogger(__name__)

# Constants
PIPELINE_LOAD_TIMEOUT = 300  # 5 minutes
WORKER_SHUTDOWN_TIMEOUT = 5  # seconds
WORKER_TERMINATE_TIMEOUT = 3  # seconds
WORKER_KILL_TIMEOUT = 1  # seconds


class PipelineNotAvailableException(Exception):
    """Exception raised when pipeline is not available for processing."""

    pass


class PipelineStatus(Enum):
    """Pipeline loading status enumeration."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class PipelineManager:
    """Manager for ML pipeline lifecycle using separate process for GPU isolation."""

    def __init__(self):
        self._status = PipelineStatus.NOT_LOADED
        self._pipeline = None
        self._pipeline_id = None
        self._load_params = None
        self._error_message = None
        self._lock = threading.RLock()  # Single reentrant lock for all access

        # Worker process management
        self._worker_process = None
        self._command_queue = None
        self._response_queue = None

    @property
    def status(self) -> PipelineStatus:
        """Get current pipeline status."""
        return self._status

    @property
    def pipeline_id(self) -> str | None:
        """Get current pipeline ID."""
        return self._pipeline_id

    @property
    def error_message(self) -> str | None:
        """Get last error message."""
        return self._error_message

    def get_pipeline(self):
        """Get the loaded pipeline instance (thread-safe).

        Returns a proxy object that forwards calls to the worker process.
        """
        with self._lock:
            if self._status != PipelineStatus.LOADED or self._worker_process is None:
                raise PipelineNotAvailableException(
                    f"Pipeline not available. Status: {self._status.value}"
                )
            # Return a proxy that will forward calls to the worker
            return PipelineProxy(self._command_queue, self._response_queue)

    def get_status_info(self) -> dict[str, Any]:
        """Get detailed status information (thread-safe).

        Note: If status is ERROR, the error message is returned once and then cleared
        to prevent persistence across page reloads.
        """
        with self._lock:
            # Capture current state before clearing
            current_status = self._status
            error_message = self._error_message
            pipeline_id = self._pipeline_id
            load_params = self._load_params

            # Capture loaded LoRA adapters if pipeline exposes them
            # Note: With worker process, we can't directly access pipeline attributes
            # This would require a worker command to query the pipeline state
            loaded_lora_adapters = None

            # If there's an error, clear it after capturing it
            # This ensures errors don't persist across page reloads
            if self._status == PipelineStatus.ERROR and error_message:
                self._error_message = None
                # Reset status to NOT_LOADED after error is retrieved
                self._status = PipelineStatus.NOT_LOADED
                self._pipeline_id = None
                self._load_params = None

            # Return the captured state (with error status if it was an error)
            return {
                "status": current_status.value,
                "pipeline_id": pipeline_id,
                "load_params": load_params,
                "loaded_lora_adapters": loaded_lora_adapters,
                "error": error_message,
            }

    async def get_pipeline_async(self):
        """Get the loaded pipeline instance (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_pipeline)

    async def get_status_info_async(self) -> dict[str, Any]:
        """Get detailed status information (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_status_info)

    async def load_pipeline(
        self, pipeline_id: str | None = None, load_params: dict | None = None
    ) -> bool:
        """
        Load a pipeline asynchronously.

        Args:
            pipeline_id: ID of pipeline to load. If None, uses PIPELINE env var.
            load_params: Pipeline-specific load parameters.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._load_pipeline_sync_wrapper, pipeline_id, load_params
        )

    def _load_pipeline_sync_wrapper(
        self, pipeline_id: str | None = None, load_params: dict | None = None
    ) -> bool:
        """Synchronous wrapper for pipeline loading with proper locking."""

        if pipeline_id is None:
            pipeline_id = os.getenv("PIPELINE", "longlive")

        with self._lock:
            # Normalize None to empty dict for comparison
            current_params = self._load_params or {}
            new_params = load_params or {}

            # If already loaded with same type and same params, return success
            if (
                self._status == PipelineStatus.LOADED
                and self._pipeline_id == pipeline_id
                and current_params == new_params
            ):
                logger.info(
                    f"Pipeline {pipeline_id} already loaded with matching parameters"
                )
                return True

            # If a different pipeline is loaded OR same pipeline with different params, unload it first
            if self._status == PipelineStatus.LOADED and (
                self._pipeline_id != pipeline_id or current_params != new_params
            ):
                self._unload_pipeline_unsafe()

            # If already loading, someone else is handling it
            if self._status == PipelineStatus.LOADING:
                logger.info("Pipeline already loading by another thread")
                return False

            # Mark as loading
            self._status = PipelineStatus.LOADING
            self._error_message = None

        # Release lock during slow loading operation
        logger.info(f"Loading pipeline in worker process: {pipeline_id}")

        try:
            # Start worker process and load pipeline
            success = self._start_worker_and_load_pipeline(pipeline_id, load_params)

            if not success:
                raise RuntimeError("Failed to load pipeline in worker process")

            # Hold lock while updating state with loaded pipeline
            with self._lock:
                self._pipeline_id = pipeline_id
                self._load_params = load_params
                self._status = PipelineStatus.LOADED

            logger.info(
                f"Pipeline {pipeline_id} loaded successfully in worker process"
            )
            return True

        except Exception as e:
            error_msg = f"Failed to load pipeline {pipeline_id}: {str(e)}"
            logger.error(error_msg)

            # Hold lock while updating state with error
            with self._lock:
                self._status = PipelineStatus.ERROR
                self._error_message = error_msg
                self._pipeline_id = None
                self._load_params = None

            # Cleanup worker on failure
            self._stop_worker()

            return False

    def _apply_load_params(
        self,
        config: dict,
        load_params: dict | None,
        default_height: int,
        default_width: int,
        default_seed: int = 42,
    ) -> None:
        """Extract and apply common load parameters (resolution, seed, LoRAs) to config.

        Args:
            config: Pipeline config dict to update
            load_params: Load parameters dict (may contain height, width, seed, loras, lora_merge_mode)
            default_height: Default height if not in load_params
            default_width: Default width if not in load_params
            default_seed: Default seed if not in load_params
        """
        height = default_height
        width = default_width
        seed = default_seed
        loras = None
        lora_merge_mode = "permanent_merge"

        if load_params:
            height = load_params.get("height", default_height)
            width = load_params.get("width", default_width)
            seed = load_params.get("seed", default_seed)
            loras = load_params.get("loras", None)
            lora_merge_mode = load_params.get("lora_merge_mode", lora_merge_mode)

        config["height"] = height
        config["width"] = width
        config["seed"] = seed
        if loras:
            config["loras"] = loras
        # Pass merge_mode directly to mixin, not via config
        config["_lora_merge_mode"] = lora_merge_mode

    def _unload_pipeline_unsafe(self):
        """Unload the current pipeline. Must be called with lock held.

        This will kill the worker process to ensure proper VRAM cleanup.
        """
        if self._pipeline_id:
            logger.info(f"Unloading pipeline: {self._pipeline_id}")

        # Stop the worker process (this ensures VRAM is cleaned up)
        self._stop_worker()

        # Reset state
        self._status = PipelineStatus.NOT_LOADED
        self._pipeline = None
        self._pipeline_id = None
        self._load_params = None
        self._error_message = None

    def _start_worker_and_load_pipeline(
        self, pipeline_id: str, load_params: dict | None = None
    ) -> bool:
        """Start worker process and load pipeline in it.

        Returns:
            bool: True if successful, False otherwise
        """
        # Stop any existing worker first
        self._stop_worker()

        # Create communication queues with spawn context for better CUDA compatibility
        # Using 'spawn' ensures a clean process without CUDA context issues
        ctx = mp.get_context("spawn")
        self._command_queue = ctx.Queue()
        self._response_queue = ctx.Queue()

        # Start worker process with spawn context
        self._worker_process = ctx.Process(
            target=pipeline_worker_process,
            args=(self._command_queue, self._response_queue),
            daemon=False,  # We want to control its lifecycle explicitly
        )
        self._worker_process.start()

        logger.info(f"Started worker process (PID: {self._worker_process.pid})")

        # Send load command
        self._command_queue.put(
            {
                "command": WorkerCommand.LOAD_PIPELINE.value,
                "pipeline_id": pipeline_id,
                "load_params": load_params,
            }
        )

        # Wait for response with timeout
        try:
            response = self._response_queue.get(timeout=PIPELINE_LOAD_TIMEOUT)

            if response["status"] == WorkerResponse.SUCCESS.value:
                logger.info(
                    f"Pipeline loaded successfully in worker: {response.get('message')}"
                )
                return True
            else:
                error_msg = response.get("error", "Unknown error")
                logger.error(f"Failed to load pipeline in worker: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Error waiting for pipeline load response: {e}")
            return False

    def _stop_worker(self):
        """Stop the worker process if it's running.

        This ensures proper VRAM cleanup by killing the process.
        """
        if self._worker_process is not None and self._worker_process.is_alive():
            logger.info(f"Stopping worker process (PID: {self._worker_process.pid})")

            # Try graceful shutdown first
            try:
                self._command_queue.put(None)  # Shutdown signal
                self._worker_process.join(timeout=WORKER_SHUTDOWN_TIMEOUT)
            except Exception as e:
                logger.warning(f"Error during graceful shutdown: {e}")

            # Force terminate if still alive
            if self._worker_process.is_alive():
                logger.warning(
                    "Worker process did not shut down gracefully, terminating..."
                )
                self._worker_process.terminate()
                self._worker_process.join(timeout=WORKER_TERMINATE_TIMEOUT)

            # Final kill if still alive
            if self._worker_process.is_alive():
                logger.warning("Worker process did not terminate, killing...")
                self._worker_process.kill()
                self._worker_process.join(timeout=WORKER_KILL_TIMEOUT)

            logger.info("Worker process stopped")

        # Clean up queues
        if self._command_queue is not None:
            try:
                self._command_queue.close()
                self._command_queue.join_thread()
            except Exception:
                pass
            self._command_queue = None

        if self._response_queue is not None:
            try:
                self._response_queue.close()
                self._response_queue.join_thread()
            except Exception:
                pass
            self._response_queue = None

        self._worker_process = None

    def unload_pipeline(self):
        """Unload the current pipeline (thread-safe)."""
        with self._lock:
            self._unload_pipeline_unsafe()

    def is_loaded(self) -> bool:
        """Check if pipeline is loaded and ready (thread-safe)."""
        with self._lock:
            return self._status == PipelineStatus.LOADED
