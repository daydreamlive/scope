"""Base interface for input sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class InputSourceInfo:
    """Information about a discovered input source."""

    name: str
    """Display name of the source (e.g., 'OBS (MacBook Pro)')"""

    identifier: str
    """Unique identifier used to connect (e.g., NDI URL or Spout sender name)"""

    metadata: dict | None = None
    """Optional additional metadata about the source"""


class InputSource(ABC):
    """Abstract base class for video input sources.

    Input sources provide video frames to Scope from external sources
    like NDI, Spout/Syphon, capture cards, RTMP streams, etc.

    Implementations must define class attributes and implement all abstract methods.
    """

    # Class attributes - must be overridden by subclasses
    source_id: ClassVar[str]
    """Unique identifier for this input source type (e.g., 'ndi', 'spout')"""

    source_name: ClassVar[str]
    """Human-readable name (e.g., 'NDI', 'Spout/Syphon')"""

    source_description: ClassVar[str]
    """Description of the input source"""

    @classmethod
    def is_available(cls) -> bool:
        """Check if this input source is available on this platform.

        Override this to check for required libraries, hardware, etc.

        Returns:
            True if the input source can be used on this system.
        """
        return True

    @abstractmethod
    def list_sources(self, timeout_ms: int = 5000) -> list[InputSourceInfo]:
        """List available sources on the network/system.

        Args:
            timeout_ms: How long to wait for source discovery.

        Returns:
            List of discovered sources.
        """
        pass

    @abstractmethod
    def connect(self, identifier: str) -> bool:
        """Connect to a specific source.

        Args:
            identifier: The identifier from InputSourceInfo.identifier

        Returns:
            True if connection was successful.
        """
        pass

    @abstractmethod
    def receive_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        """Receive a video frame from the connected source.

        Args:
            timeout_ms: How long to wait for a frame.

        Returns:
            RGB numpy array with shape (H, W, 3) and dtype uint8,
            or None if no frame is available.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the current source."""
        pass

    def close(self):
        """Clean up all resources. Called when the input source is no longer needed."""
        self.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

