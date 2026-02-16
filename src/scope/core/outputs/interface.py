"""Base interface for output sinks."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch


class OutputSink(ABC):
    """Abstract base class for video output sinks.

    An output sink sends processed video frames to an external destination
    (e.g. Spout, NDI output, etc.).
    """

    source_id: ClassVar[str]
    """Unique identifier for this output sink type."""

    source_name: ClassVar[str]
    """Human-readable name."""

    source_description: ClassVar[str]

    @classmethod
    def is_available(cls) -> bool:
        """Check if this output sink is available on this platform."""
        return True

    @abstractmethod
    def create(self, name: str, width: int, height: int) -> bool:
        """Create the output sink with the given name and dimensions.

        Returns:
            True if creation was successful.
        """
        pass

    @abstractmethod
    def send_frame(self, frame: np.ndarray | torch.Tensor) -> bool:
        """Send a video frame to the output destination.

        Args:
            frame: (H, W, C) numpy array or torch tensor.

        Returns:
            True if send was successful.
        """
        pass

    @abstractmethod
    def resize(self, width: int, height: int):
        """Update the output resolution."""
        pass

    @abstractmethod
    def close(self):
        """Release all resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
