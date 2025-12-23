"""Pipeline Manager for lazy loading and managing ML pipelines."""

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Any

import zmq

logger = logging.getLogger(__name__)

# Constants
PIPELINE_LOAD_TIMEOUT = 300  # 5 minutes
WORKER_SHUTDOWN_TIMEOUT = 5  # seconds
WORKER_TERMINATE_TIMEOUT = 3  # seconds
WORKER_KILL_TIMEOUT = 1  # seconds


class WorkerCommand(Enum):
    """Commands that can be sent to the worker process."""

    LOAD_PIPELINE = "load_pipeline"
    UNLOAD_PIPELINE = "unload_pipeline"
    CREATE_FRAME_PROCESSOR = "create_frame_processor"
    DESTROY_FRAME_PROCESSOR = "destroy_frame_processor"
    PUT_FRAME = "put_frame"
    GET_FRAME = "get_frame"
    UPDATE_PARAMETERS = "update_parameters"
    GET_FPS = "get_fps"
    SHUTDOWN = "shutdown"


class WorkerResponse(Enum):
    """Response types from worker process."""

    SUCCESS = "success"
    ERROR = "error"
    PIPELINE_LOADED = "pipeline_loaded"
    PIPELINE_NOT_LOADED = "pipeline_not_loaded"
    RESULT = "result"
    FRAME_PROCESSOR_CREATED = "frame_processor_created"
    FRAME = "frame"


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
        self._worker_process: subprocess.Popen | None = None

        # ZMQ context and sockets
        self._zmq_context: zmq.Context | None = None
        self._command_socket: zmq.Socket | None = None
        self._response_socket: zmq.Socket | None = None
        self._command_port: int | None = None
        self._response_port: int | None = None

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

        Note: Pipeline is now loaded in worker process. Use create_frame_processor()
        to get a FrameProcessor that uses the pipeline directly in the worker process.
        """
        with self._lock:
            if self._status != PipelineStatus.LOADED or self._worker_process is None:
                raise PipelineNotAvailableException(
                    f"Pipeline not available. Status: {self._status.value}"
                )
            # Pipeline is in worker process, return a placeholder to indicate it's loaded
            # Actual pipeline access is through FrameProcessor in worker process
            return None

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
            # This will raise RuntimeError if loading fails
            self._start_worker_and_load_pipeline(pipeline_id, load_params)

            # Hold lock while updating state with loaded pipeline
            with self._lock:
                self._pipeline_id = pipeline_id
                self._load_params = load_params
                self._status = PipelineStatus.LOADED

            logger.info(f"Pipeline {pipeline_id} loaded successfully in worker process")
            return True

        except Exception as e:
            # Capture full error message including traceback for better debugging
            error_details = str(e)
            # If the error already contains detailed info, use it; otherwise add traceback
            if (
                "traceback" not in error_details.lower()
                and "Worker process" not in error_details
            ):
                error_details = f"{error_details}\n{traceback.format_exc()}"
            error_msg = f"Failed to load pipeline {pipeline_id}: {error_details}"
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

    def _find_free_port(self) -> int:
        """Find a free port to use for ZMQ communication."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _get_pipeline_env_dir(self, pipeline_id: str) -> tuple[Path, bool]:
        """Get the environment directory for a specific pipeline.

        Each pipeline has its own pyproject.toml with pipeline-specific dependencies.
        For plugin pipelines, this returns the plugin's installed directory.

        Args:
            pipeline_id: The pipeline identifier

        Returns:
            Tuple of (path to the pipeline's environment directory, is_plugin)
        """
        from .models_config import get_plugins_dir

        project_root = Path(__file__).parent.parent.parent.parent

        # Map built-in pipeline IDs to their directory names
        builtin_pipeline_dir_map = {
            "passthrough": "passthrough",
            "streamdiffusionv2": "streamdiffusionv2",
            "longlive": "longlive",
            "krea-realtime-video": "krea_realtime_video",
            "reward-forcing": "reward_forcing",
        }

        # Check if it's a built-in pipeline
        if pipeline_id in builtin_pipeline_dir_map:
            pipeline_dir = (
                project_root
                / "src"
                / "scope"
                / "core"
                / "pipelines"
                / builtin_pipeline_dir_map[pipeline_id]
            )
            if pipeline_dir.exists() and (pipeline_dir / "pyproject.toml").exists():
                return pipeline_dir, False

        # Check if it's a plugin pipeline
        plugins_dir = get_plugins_dir()
        plugin_dir = plugins_dir / pipeline_id
        if plugin_dir.exists() and (plugin_dir / "pyproject.toml").exists():
            logger.info(f"Found plugin pipeline at: {plugin_dir}")
            return plugin_dir, True

        # Fallback to base worker_env for unknown pipelines
        worker_env_dir = project_root / "worker_env"
        if not worker_env_dir.exists():
            raise RuntimeError(f"Worker environment directory not found: {worker_env_dir}")
        return worker_env_dir, False

    def _prepare_plugin_environment(self, plugin_dir: Path) -> None:
        """Prepare a plugin's environment by updating pyproject.toml with source overrides.

        Plugins may depend on 'daydream-scope' or 'scope-worker-base', but these
        aren't published to PyPI. We update the plugin's pyproject.toml to add
        source overrides pointing to the local packages.

        Args:
            plugin_dir: Path to the plugin directory
        """
        # Try tomllib (Python 3.11+) first, fall back to tomli for Python 3.10
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        project_root = Path(__file__).parent.parent.parent.parent
        worker_env_dir = project_root / "worker_env"

        pyproject_path = plugin_dir / "pyproject.toml"
        if not pyproject_path.exists():
            logger.warning(f"No pyproject.toml found at {pyproject_path}")
            return

        # Read existing pyproject.toml
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)

        # Check if we've already added source overrides
        existing_sources = pyproject_data.get("tool", {}).get("uv", {}).get("sources", {})
        if "daydream-scope" in existing_sources:
            logger.debug(f"Source overrides already exist in {pyproject_path}")
            return

        # Check if scope-worker-base is in dependencies
        dependencies = pyproject_data.get("project", {}).get("dependencies", [])
        needs_worker_base = not any(
            "scope-worker-base" in dep for dep in dependencies
        )

        # Append source overrides and scope-worker-base dependency to pyproject.toml
        additions = "\n# Auto-generated by daydream-scope for plugin dependency resolution\n"

        if needs_worker_base:
            # Rewrite dependencies section to include scope-worker-base
            new_deps = dependencies + ["scope-worker-base"]
            deps_str = ", ".join(f'"{d}"' for d in new_deps)
            # We'll add scope-worker-base via the sources mechanism

        additions += f'''[tool.uv.sources]
daydream-scope = {{ path = "{project_root}", editable = true }}
scope-worker-base = {{ path = "{worker_env_dir}", editable = true }}
'''

        # If dependencies need updating, we need to rewrite the file
        if needs_worker_base:
            # Read file as text and add scope-worker-base to dependencies
            with open(pyproject_path, "r") as f:
                content = f.read()

            # Add scope-worker-base to dependencies list
            import re

            # Find the dependencies line and add scope-worker-base
            dep_pattern = r'(dependencies\s*=\s*\[)([^\]]*)(])'

            def add_worker_base(match):
                prefix = match.group(1)
                deps = match.group(2)
                suffix = match.group(3)
                # Add scope-worker-base if not already there
                if "scope-worker-base" not in deps:
                    deps = deps.rstrip()
                    if deps and not deps.endswith(","):
                        deps += ","
                    deps += '\n    "scope-worker-base",\n'
                return prefix + deps + suffix

            content = re.sub(dep_pattern, add_worker_base, content)
            content += additions

            with open(pyproject_path, "w") as f:
                f.write(content)
        else:
            with open(pyproject_path, "a") as f:
                f.write(additions)

        # Remove any existing .venv and uv.lock so they get regenerated
        venv_path = plugin_dir / ".venv"
        lock_path = plugin_dir / "uv.lock"
        if venv_path.exists():
            import shutil

            shutil.rmtree(venv_path)
            logger.debug(f"Removed existing .venv at {venv_path}")
        if lock_path.exists():
            lock_path.unlink()
            logger.debug(f"Removed existing uv.lock at {lock_path}")

        logger.info(f"Added source overrides to {pyproject_path}")

    def _start_worker_and_load_pipeline(
        self, pipeline_id: str, load_params: dict | None = None
    ) -> bool:
        """Start worker process and load pipeline in it.

        Returns:
            bool: True if successful, False otherwise

        Raises:
            RuntimeError: If worker process dies unexpectedly or fails to load pipeline
        """
        # Stop any existing worker first
        self._stop_worker()

        # Get the pipeline-specific environment directory
        project_root = Path(__file__).parent.parent.parent.parent
        pipeline_env_dir, is_plugin = self._get_pipeline_env_dir(pipeline_id)

        # For plugins, prepare the environment with source overrides
        if is_plugin:
            self._prepare_plugin_environment(pipeline_env_dir)

        logger.info(f"Using pipeline environment: {pipeline_env_dir}")

        # Set up ZMQ context and sockets
        self._zmq_context = zmq.Context()

        # Find free ports for communication
        self._command_port = self._find_free_port()
        self._response_port = self._find_free_port()

        # PUSH socket for sending commands (bind)
        self._command_socket = self._zmq_context.socket(zmq.PUSH)
        self._command_socket.bind(f"tcp://127.0.0.1:{self._command_port}")

        # PULL socket for receiving responses (bind)
        self._response_socket = self._zmq_context.socket(zmq.PULL)
        self._response_socket.bind(f"tcp://127.0.0.1:{self._response_port}")

        # Get models directory from environment or config
        from .models_config import get_models_dir

        models_dir = str(get_models_dir())

        # Build environment for the worker process
        env = os.environ.copy()
        env["SCOPE_PROJECT_ROOT"] = str(project_root)
        env["MODELS_DIR"] = models_dir

        # Start worker process using uv run
        cmd = [
            "uv",
            "run",
            "scope-worker",
            "--command-port",
            str(self._command_port),
            "--response-port",
            str(self._response_port),
        ]

        logger.info(f"Starting worker process: {' '.join(cmd)}")
        logger.info(f"Working directory: {pipeline_env_dir}")

        self._worker_process = subprocess.Popen(
            cmd,
            cwd=str(pipeline_env_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Start a thread to log worker output
        def log_worker_output():
            if self._worker_process and self._worker_process.stdout:
                for line in iter(self._worker_process.stdout.readline, b""):
                    if line:
                        logger.info(f"[Worker] {line.decode('utf-8').rstrip()}")

        output_thread = threading.Thread(target=log_worker_output, daemon=True)
        output_thread.start()

        logger.info(f"Started worker process (PID: {self._worker_process.pid})")

        # Give the worker a moment to start and connect
        time.sleep(0.5)

        # Send load command
        command = {
            "command": WorkerCommand.LOAD_PIPELINE.value,
            "pipeline_id": pipeline_id,
            "load_params": load_params,
        }
        self._command_socket.send(json.dumps(command).encode("utf-8"))

        # Wait for response with timeout, checking if worker is still alive
        start_time = time.time()
        check_interval = 500  # milliseconds for ZMQ poll

        while True:
            # Check if worker process is still alive
            if self._worker_process.poll() is not None:
                # Worker died unexpectedly - get exit code
                exit_code = self._worker_process.returncode
                if exit_code != 0:
                    # Process crashed (e.g., OOM, segmentation fault)
                    if exit_code == -9:  # SIGKILL (often OOM killer)
                        error_msg = (
                            f"Worker process was killed (likely out of memory). "
                            f"Exit code: {exit_code}. "
                            f"Pipeline loading failed for {pipeline_id}."
                        )
                    elif exit_code < 0:
                        error_msg = (
                            f"Worker process crashed with signal {abs(exit_code)}. "
                            f"Pipeline loading failed for {pipeline_id}."
                        )
                    else:
                        error_msg = (
                            f"Worker process exited unexpectedly with code {exit_code}. "
                            f"Pipeline loading failed for {pipeline_id}."
                        )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                else:
                    # Process exited normally but we didn't get a response
                    error_msg = (
                        f"Worker process exited before sending response. "
                        f"Pipeline loading failed for {pipeline_id}."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed >= PIPELINE_LOAD_TIMEOUT:
                error_msg = (
                    f"Timeout waiting for pipeline load response after {PIPELINE_LOAD_TIMEOUT}s. "
                    f"Pipeline loading failed for {pipeline_id}."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Try to get response with short timeout to allow periodic worker checks
            if self._response_socket.poll(timeout=check_interval):
                message = self._response_socket.recv()
                response = json.loads(message.decode("utf-8"))

                if response["status"] == WorkerResponse.SUCCESS.value:
                    logger.info(
                        f"Pipeline loaded successfully in worker: {response.get('message')}"
                    )
                    return True
                else:
                    # Worker sent an error response
                    error_msg = response.get("error", "Unknown error")
                    logger.error(f"Failed to load pipeline in worker: {error_msg}")
                    raise RuntimeError(f"Pipeline loading failed: {error_msg}")

    def _stop_worker(self):
        """Stop the worker process if it's running.

        This ensures proper VRAM cleanup by killing the process.
        """
        if self._worker_process is not None and self._worker_process.poll() is None:
            logger.info(f"Stopping worker process (PID: {self._worker_process.pid})")

            # Try graceful shutdown first by sending shutdown command
            try:
                if self._command_socket:
                    # Set linger to 0 for immediate close (avoid blocking on pending messages)
                    self._command_socket.setsockopt(zmq.LINGER, 0)
                    command = {"command": WorkerCommand.SHUTDOWN.value}
                    # Use NOBLOCK to avoid blocking if receiver is not consuming
                    self._command_socket.send(
                        json.dumps(command).encode("utf-8"), zmq.NOBLOCK
                    )
                    # Wait for process to exit
                    try:
                        self._worker_process.wait(timeout=WORKER_SHUTDOWN_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        pass
            except zmq.ZMQError:
                # Socket error (e.g., would block), skip graceful shutdown
                pass
            except Exception as e:
                logger.warning(f"Error during graceful shutdown: {e}")

            # Force terminate if still alive
            if self._worker_process.poll() is None:
                logger.warning(
                    "Worker process did not shut down gracefully, terminating..."
                )
                self._worker_process.terminate()
                try:
                    self._worker_process.wait(timeout=WORKER_TERMINATE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    pass

            # Final kill if still alive
            if self._worker_process.poll() is None:
                logger.warning("Worker process did not terminate, killing...")
                self._worker_process.kill()
                try:
                    self._worker_process.wait(timeout=WORKER_KILL_TIMEOUT)
                except subprocess.TimeoutExpired:
                    pass

            logger.info("Worker process stopped")

        # Clean up ZMQ sockets - set LINGER to 0 first to avoid blocking on close
        if self._command_socket is not None:
            try:
                self._command_socket.setsockopt(zmq.LINGER, 0)
                self._command_socket.close()
            except Exception:
                pass
            self._command_socket = None

        if self._response_socket is not None:
            try:
                self._response_socket.setsockopt(zmq.LINGER, 0)
                self._response_socket.close()
            except Exception:
                pass
            self._response_socket = None

        if self._zmq_context is not None:
            try:
                self._zmq_context.term()
            except Exception:
                pass
            self._zmq_context = None

        self._worker_process = None
        self._command_port = None
        self._response_port = None

    def unload_pipeline(self):
        """Unload the current pipeline (thread-safe)."""
        with self._lock:
            self._unload_pipeline_unsafe()

    def is_loaded(self) -> bool:
        """Check if pipeline is loaded and ready (thread-safe)."""
        with self._lock:
            return self._status == PipelineStatus.LOADED

    def create_frame_processor(self, frame_processor_id: str, initial_parameters: dict = None):
        """Create a FrameProcessor in the worker process (thread-safe).

        Args:
            frame_processor_id: Unique identifier for this FrameProcessor
            initial_parameters: Initial parameters for the FrameProcessor

        Returns:
            FrameProcessorProxy instance
        """
        from .frame_processor_proxy import FrameProcessorProxy

        with self._lock:
            if self._status != PipelineStatus.LOADED or self._worker_process is None:
                raise PipelineNotAvailableException(
                    f"Pipeline not available. Status: {self._status.value}"
                )

            # Send command to create FrameProcessor in worker
            command = {
                "command": WorkerCommand.CREATE_FRAME_PROCESSOR.value,
                "frame_processor_id": frame_processor_id,
                "initial_parameters": initial_parameters or {},
            }
            self._command_socket.send(json.dumps(command).encode("utf-8"))

            # Wait for response
            if self._response_socket.poll(timeout=60000):  # 60 seconds
                message = self._response_socket.recv()
                response = json.loads(message.decode("utf-8"))

                if response["status"] == WorkerResponse.FRAME_PROCESSOR_CREATED.value:
                    return FrameProcessorProxy(
                        frame_processor_id=frame_processor_id,
                        command_socket=self._command_socket,
                        response_socket=self._response_socket,
                    )
                else:
                    error_msg = response.get("error", "Unknown error")
                    raise RuntimeError(f"Failed to create FrameProcessor: {error_msg}")
            else:
                raise RuntimeError("Timeout waiting for FrameProcessor creation")
