"""FrameProcessor Proxy - Communicates with FrameProcessor in worker process."""

import logging
import multiprocessing as mp
import queue
import uuid

import torch
from aiortc.mediastreams import VideoFrame

from .pipeline_worker import WorkerCommand, WorkerResponse

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 60  # seconds


class FrameProcessorProxy:
    """Proxy object that communicates with FrameProcessor in worker process."""

    def __init__(
        self,
        frame_processor_id: str,
        command_queue: mp.Queue,
        response_queue: mp.Queue,
    ):
        self.frame_processor_id = frame_processor_id
        self._command_queue = command_queue
        self._response_queue = response_queue

    def put(self, frame: VideoFrame) -> bool:
        """Put a frame into the FrameProcessor buffer."""
        try:
            # Serialize VideoFrame to dict for inter-process communication
            frame_array = frame.to_ndarray(format="rgb24")
            frame_data = {"array": frame_array}

            self._command_queue.put(
                {
                    "command": WorkerCommand.PUT_FRAME.value,
                    "frame_processor_id": self.frame_processor_id,
                    "frame_data": frame_data,
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error putting frame: {e}")
            return False

    def get(self) -> torch.Tensor | None:
        """Get a processed frame from the FrameProcessor."""
        try:
            # Request a frame
            self._command_queue.put(
                {
                    "command": WorkerCommand.GET_FRAME.value,
                    "frame_processor_id": self.frame_processor_id,
                }
            )

            # Wait for response with timeout
            try:
                response = self._response_queue.get(timeout=1.0)

                if response["status"] == WorkerResponse.FRAME.value:
                    frame_data = response.get("frame_data")
                    if frame_data and frame_data.get("__tensor__"):
                        # Deserialize tensor from numpy array
                        return torch.from_numpy(frame_data["data"])
                    return None
                elif response["status"] == WorkerResponse.RESULT.value:
                    # No frame available
                    return None
                elif response["status"] == WorkerResponse.ERROR.value:
                    error_msg = response.get("error", "Unknown error")
                    logger.error(f"Error getting frame: {error_msg}")
                    return None
                else:
                    logger.warning(f"Unexpected response status: {response['status']}")
                    return None

            except queue.Empty:
                # Timeout - no frame available
                return None

        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    def get_current_pipeline_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS."""
        try:
            self._command_queue.put(
                {
                    "command": WorkerCommand.GET_FPS.value,
                    "frame_processor_id": self.frame_processor_id,
                }
            )

            try:
                response = self._response_queue.get(timeout=DEFAULT_TIMEOUT)

                if response["status"] == WorkerResponse.RESULT.value:
                    return response.get("result", 30.0)
                elif response["status"] == WorkerResponse.ERROR.value:
                    error_msg = response.get("error", "Unknown error")
                    logger.error(f"Error getting FPS: {error_msg}")
                    return 30.0
                else:
                    logger.warning(f"Unexpected response status: {response['status']}")
                    return 30.0

            except queue.Empty:
                logger.error("Timeout waiting for FPS response")
                return 30.0

        except Exception as e:
            logger.error(f"Error getting FPS: {e}")
            return 30.0

    def update_parameters(self, parameters: dict):
        """Update parameters that will be used in the next pipeline call."""
        try:
            self._command_queue.put(
                {
                    "command": WorkerCommand.UPDATE_PARAMETERS.value,
                    "frame_processor_id": self.frame_processor_id,
                    "parameters": parameters,
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            return False

    def start(self):
        """Start the FrameProcessor (already started when created)."""
        pass

    def stop(self):
        """Stop and destroy the FrameProcessor."""
        try:
            self._command_queue.put(
                {
                    "command": WorkerCommand.DESTROY_FRAME_PROCESSOR.value,
                    "frame_processor_id": self.frame_processor_id,
                }
            )

            # Wait for response
            try:
                response = self._response_queue.get(timeout=DEFAULT_TIMEOUT)
                if response["status"] == WorkerResponse.SUCCESS.value:
                    logger.info(f"FrameProcessor {self.frame_processor_id} stopped")
                else:
                    error_msg = response.get("error", "Unknown error")
                    logger.warning(f"Error stopping FrameProcessor: {error_msg}")
            except queue.Empty:
                logger.warning("Timeout waiting for FrameProcessor stop response")

        except Exception as e:
            logger.error(f"Error stopping FrameProcessor: {e}")
