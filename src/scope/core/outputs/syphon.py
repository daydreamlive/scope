"""Syphon output sink implementation (macOS)."""

import logging
import sys
from typing import ClassVar

import numpy as np
import torch

from .interface import OutputSink

logger = logging.getLogger(__name__)


class SyphonOutputSink(OutputSink):
    """Output sink that sends video frames via Syphon on macOS."""

    source_id: ClassVar[str] = "syphon"
    source_name: ClassVar[str] = "Syphon"
    source_description: ClassVar[str] = (
        "Send video frames to Syphon receivers on macOS "
        "like TouchDesigner, Resolume, OBS, etc."
    )

    def __init__(self):
        self._sender = None
        self._name = ""
        self._width = 0
        self._height = 0

    @classmethod
    def is_available(cls) -> bool:
        if sys.platform != "darwin":
            return False
        try:
            import syphon  # noqa: F401

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
        """Create the Syphon sender."""
        try:
            from scope.server.syphon.sender import SyphonSender

            self.close()

            self._sender = SyphonSender(name, width, height)
            if self._sender.create():
                self._name = name
                self._width = width
                self._height = height
                logger.info(f"SyphonOutputSink created: '{name}' ({width}x{height})")
                return True

            logger.error("Failed to create SyphonSender")
            self._sender = None
            return False
        except ImportError:
            logger.error("syphon-python not available")
            return False
        except Exception as e:
            logger.error(f"Error creating SyphonOutputSink: {e}")
            self._sender = None
            return False

    def send_frame(self, frame: np.ndarray | torch.Tensor) -> bool:
        """Send a video frame to Syphon."""
        if self._sender is None:
            return False
        try:
            return self._sender.send(frame)
        except Exception as e:
            logger.error(f"Error sending Syphon frame: {e}")
            return False

    def resize(self, width: int, height: int):
        """Update sender resolution."""
        if self._sender is not None:
            self._sender.resize(width, height)
        self._width = width
        self._height = height

    def close(self):
        """Release Syphon sender resources."""
        if self._sender is not None:
            try:
                self._sender.release()
            except Exception as e:
                logger.error(f"Error releasing SyphonSender: {e}")
            finally:
                self._sender = None
                self._name = ""
                self._width = 0
                self._height = 0
