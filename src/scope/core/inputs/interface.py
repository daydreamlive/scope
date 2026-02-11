"""Base interface for input sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class InputSourceInfo:
    """Information about a discovered input source."""

    name: str
    """Display name of the source."""

    identifier: str
    """Unique identifier used to connect."""

    metadata: dict | None = None
    """Optional additional metadata about the source."""


class InputSource(ABC):
    """Abstract base class for video input sources."""

    source_id: ClassVar[str]
    """Unique identifier for this input source type."""

    source_name: ClassVar[str]
    """Human-readable name."""

    source_description: ClassVar[str]

    @classmethod
    def is_available(cls) -> bool:
        """Check if this input source is available on this platform."""
        return True

    @abstractmethod
    def list_sources(self, timeout_ms: int = 5000) -> list[InputSourceInfo]:
        """List available sources on the network/system."""
        pass

    @abstractmethod
    def connect(self, identifier: str) -> bool:
        """Connect to a specific source.

        Returns:
            True if connection was successful.
        """
        pass

    @abstractmethod
    def receive_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        """Receive a video frame from the connected source.

        Returns:
            RGB numpy array with shape (H, W, 3) and dtype uint8,
            or None if no frame is available.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the current source."""
        pass

    def get_source_resolution(
        self, identifier: str, timeout_ms: int = 5000
    ) -> tuple[int, int] | None:
        """Probe a source's native resolution by connecting and reading one frame.

        Returns:
            (width, height) tuple, or None if resolution could not be determined.
        """
        return None

    def close(self):
        """Clean up all resources."""
        self.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
