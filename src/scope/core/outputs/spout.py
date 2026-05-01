"""Spout output sink implementation (Windows)."""

import logging
from typing import ClassVar

import numpy as np
import torch

from .interface import OutputSink

logger = logging.getLogger(__name__)


class SpoutOutputSink(OutputSink):
    """Output sink that sends video frames via Spout on Windows.

    Wraps the existing SpoutSender from scope.server.spout.
    """

    source_id: ClassVar[str] = "spout"
    source_name: ClassVar[str] = "Spout"
    source_description: ClassVar[str] = (
        "Send video frames to Spout receivers on Windows "
        "like TouchDesigner, Resolume, OBS, etc."
    )

    def __init__(self):
        self._sender = None
        self._name = ""
        self._width = 0
        self._height = 0

    @classmethod
    def is_available(cls) -> bool:
        """Check if SpoutGL is installed."""
        try:
            import SpoutGL  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return self._name

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def create(self, name: str, width: int, height: int) -> bool:
        """Create the Spout sender."""
        try:
            from scope.server.spout import SpoutSender

            self.close()

            self._sender = SpoutSender(name, width, height)
            if self._sender.create():
                self._name = name
                self._width = width
                self._height = height
                logger.info(f"SpoutOutputSink created: '{name}' ({width}x{height})")
                return True
            else:
                logger.error("Failed to create SpoutSender")
                self._sender = None
                return False
        except ImportError:
            logger.error("SpoutGL not available")
            return False
        except Exception as e:
            logger.error(f"Error creating SpoutOutputSink: {e}")
            self._sender = None
            return False

    def send_frame(self, frame: np.ndarray | torch.Tensor) -> bool:
        """Send a video frame to Spout."""
        if self._sender is None:
            return False
        try:
            return self._sender.send(frame)
        except Exception as e:
            logger.error(f"Error sending Spout frame: {e}")
            return False

    def resize(self, width: int, height: int):
        """Update the sender resolution."""
        if self._sender is not None:
            self._sender.resize(width, height)
        self._width = width
        self._height = height

    def close(self):
        """Release Spout sender resources."""
        if self._sender is not None:
            try:
                self._sender.release()
            except Exception as e:
                logger.error(f"Error releasing SpoutSender: {e}")
            finally:
                self._sender = None
                self._name = ""
                self._width = 0
                self._height = 0
