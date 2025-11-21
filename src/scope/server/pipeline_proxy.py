"""Pipeline Proxy - Forwards pipeline calls to worker process."""

import logging
import multiprocessing as mp

from .pipeline_worker import (
    WorkerCommand,
    WorkerResponse,
    _deserialize_tensors,
    _serialize_tensors,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 60  # seconds


class PipelineProxy:
    """Proxy object that forwards pipeline calls to the worker process."""

    def __init__(self, command_queue: mp.Queue, response_queue: mp.Queue):
        self._command_queue = command_queue
        self._response_queue = response_queue
        self._attr_cache = {}  # Cache for hasattr checks

    def _call_worker(self, method: str, *args, **kwargs):
        """Call a method on the worker process pipeline.

        Args:
            method: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result from the worker process
        """
        # Serialize arguments for transmission
        serialized_args = _serialize_tensors(args)
        serialized_kwargs = _serialize_tensors(kwargs)

        # Send command
        self._command_queue.put(
            {
                "command": WorkerCommand.CALL_PIPELINE.value,
                "method": method,
                "args": serialized_args,
                "kwargs": serialized_kwargs,
            }
        )

        # Wait for response
        try:
            response = self._response_queue.get(timeout=DEFAULT_TIMEOUT)

            if response["status"] == WorkerResponse.RESULT.value:
                # Deserialize and return result
                return _deserialize_tensors(response["result"])
            elif response["status"] == WorkerResponse.ERROR.value:
                error_msg = response.get("error", "Unknown error")
                # Re-raise AttributeError so hasattr() checks work correctly
                if "AttributeError" in error_msg:
                    raise AttributeError(error_msg.replace("AttributeError: ", ""))
                raise RuntimeError(f"Worker process error: {error_msg}")
            else:
                raise RuntimeError(f"Unexpected response status: {response['status']}")

        except Exception as e:
            logger.error(f"Error calling worker method {method}: {e}")
            raise

    def __call__(self, *args, **kwargs):
        """Forward __call__ to worker process."""
        return self._call_worker("__call__", *args, **kwargs)

    def _has_attr(self, name: str) -> bool:
        """Check if the pipeline has an attribute/method."""
        # Check cache first
        if name in self._attr_cache:
            return self._attr_cache[name]

        # Check with worker process
        try:
            self._command_queue.put(
                {
                    "command": WorkerCommand.HAS_ATTR.value,
                    "attr_name": name,
                }
            )

            response = self._response_queue.get(timeout=DEFAULT_TIMEOUT)

            if response["status"] == WorkerResponse.RESULT.value:
                has_attr = _deserialize_tensors(response["result"])
                self._attr_cache[name] = has_attr
                return has_attr
            else:
                # On error, assume False
                self._attr_cache[name] = False
                return False
        except Exception as e:
            logger.error(f"Error checking attribute {name}: {e}")
            self._attr_cache[name] = False
            return False

    def __getattr__(self, name):
        """Forward attribute access to worker process.

        This allows accessing pipeline attributes like pipeline.state or pipeline.prepare
        by forwarding the call to the worker process.

        For hasattr() to work correctly, we check with the worker process first.
        """
        # Check if attribute exists (for hasattr() support)
        if not self._has_attr(name):
            raise AttributeError(f"Pipeline does not have attribute '{name}'")

        # Create a wrapper that calls the worker
        def method_wrapper(*args, **kwargs):
            return self._call_worker(name, *args, **kwargs)

        return method_wrapper
