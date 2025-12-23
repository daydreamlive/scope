"""FrameProcessor Proxy - Communicates with FrameProcessor in worker process via ZMQ."""

import json
import logging
from enum import Enum

import numpy as np
import torch
import zmq
from aiortc.mediastreams import VideoFrame

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 60000  # milliseconds


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


class FrameProcessorProxy:
    """Proxy object that communicates with FrameProcessor in worker process via ZMQ."""

    def __init__(
        self,
        frame_processor_id: str,
        command_socket: zmq.Socket,
        response_socket: zmq.Socket,
    ):
        self.frame_processor_id = frame_processor_id
        self._command_socket = command_socket
        self._response_socket = response_socket

    def put(self, frame: VideoFrame) -> bool:
        """Put a frame into the FrameProcessor buffer."""
        try:
            # Serialize VideoFrame to binary for efficient transfer
            frame_array = frame.to_ndarray(format="rgb24")

            # Send command with binary frame data
            command = {
                "command": WorkerCommand.PUT_FRAME.value,
                "frame_processor_id": self.frame_processor_id,
                "__has_binary__": True,
                "frame_info": {
                    "dtype": str(frame_array.dtype),
                    "shape": frame_array.shape,
                },
            }
            self._command_socket.send(json.dumps(command).encode("utf-8"), zmq.SNDMORE)
            self._command_socket.send(frame_array.tobytes())
            return True
        except Exception as e:
            logger.error(f"Error putting frame: {e}")
            return False

    def get(self) -> torch.Tensor | None:
        """Get a processed frame from the FrameProcessor."""
        try:
            # Request a frame
            command = {
                "command": WorkerCommand.GET_FRAME.value,
                "frame_processor_id": self.frame_processor_id,
            }
            self._command_socket.send(json.dumps(command).encode("utf-8"))

            # Wait for response with timeout
            if self._response_socket.poll(timeout=1000):  # 1 second
                message = self._response_socket.recv()
                response = json.loads(message.decode("utf-8"))

                if response["status"] == WorkerResponse.FRAME.value:
                    if response.get("__has_binary__"):
                        # Receive binary frame data
                        binary_data = self._response_socket.recv()
                        frame_info = response.get("frame_info", {})
                        arr = np.frombuffer(
                            binary_data, dtype=frame_info["dtype"]
                        ).reshape(frame_info["shape"])
                        return torch.from_numpy(arr.copy())
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
            else:
                # Timeout - no frame available
                return None

        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    def get_current_pipeline_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS."""
        try:
            command = {
                "command": WorkerCommand.GET_FPS.value,
                "frame_processor_id": self.frame_processor_id,
            }
            self._command_socket.send(json.dumps(command).encode("utf-8"))

            if self._response_socket.poll(timeout=DEFAULT_TIMEOUT):
                message = self._response_socket.recv()
                response = json.loads(message.decode("utf-8"))

                if response["status"] == WorkerResponse.RESULT.value:
                    return response.get("result", 30.0)
                elif response["status"] == WorkerResponse.ERROR.value:
                    error_msg = response.get("error", "Unknown error")
                    logger.error(f"Error getting FPS: {error_msg}")
                    return 30.0
                else:
                    logger.warning(f"Unexpected response status: {response['status']}")
                    return 30.0
            else:
                logger.error("Timeout waiting for FPS response")
                return 30.0

        except Exception as e:
            logger.error(f"Error getting FPS: {e}")
            return 30.0

    def update_parameters(self, parameters: dict):
        """Update parameters that will be used in the next pipeline call."""
        try:
            command = {
                "command": WorkerCommand.UPDATE_PARAMETERS.value,
                "frame_processor_id": self.frame_processor_id,
                "parameters": parameters,
            }
            self._command_socket.send(json.dumps(command).encode("utf-8"))
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
            command = {
                "command": WorkerCommand.DESTROY_FRAME_PROCESSOR.value,
                "frame_processor_id": self.frame_processor_id,
            }
            self._command_socket.send(json.dumps(command).encode("utf-8"))

            # Wait for response
            if self._response_socket.poll(timeout=DEFAULT_TIMEOUT):
                message = self._response_socket.recv()
                response = json.loads(message.decode("utf-8"))

                if response["status"] == WorkerResponse.SUCCESS.value:
                    logger.info(f"FrameProcessor {self.frame_processor_id} stopped")
                else:
                    error_msg = response.get("error", "Unknown error")
                    logger.warning(f"Error stopping FrameProcessor: {error_msg}")
            else:
                logger.warning("Timeout waiting for FrameProcessor stop response")

        except Exception as e:
            logger.error(f"Error stopping FrameProcessor: {e}")
