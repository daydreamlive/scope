"""Base interface for input sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


class InputSourceError(Exception):
    """Base class for input source errors raised during probe/connect."""


class InvalidSourceURLError(InputSourceError):
    """Raised when a source identifier/URL is malformed or disallowed."""


class SourceUnavailableError(InputSourceError):
    """Raised when a source exists but cannot be accessed (private, deleted, geo-blocked, etc.)."""


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

    @classmethod
    def get_definition(cls):
        """Return a :class:`NodeDefinition` describing this source as a node type.

        Input sources live in the unified :class:`NodeRegistry` alongside
        plain nodes and pipelines, so every iteration over the registry
        (definitions endpoint, pipeline schemas, etc.) must succeed on
        them. The default derives the definition from the class's
        ``source_id`` / ``source_name`` / ``source_description``, and
        includes any UI params returned by :meth:`get_source_ui_params`
        so the frontend can render the right controls without hardcoding
        each source type.
        """
        from scope.core.nodes.base import NodeDefinition, NodePort

        return NodeDefinition(
            node_type_id=cls.source_id,
            display_name=cls.source_name,
            category="source",
            description=cls.source_description,
            inputs=[],
            outputs=[
                NodePort(name="video", port_type="video", description="Video output"),
            ],
            params=cls.get_source_ui_params(),
            continuous=True,
        )

    @classmethod
    def get_source_ui_params(cls):
        """Return the UI controls the frontend should render for this source.

        The first param named ``source_name`` describes how to collect the
        identifier stored on ``GraphNode.source_name`` (URL, discovered-
        sender dropdown, asset picker, etc.). Additional params map to
        other ``GraphNode`` fields (e.g. ``source_flip_vertical``).

        The default returns no params. Subclasses override to declare
        their UI. Supported ``ui`` hints for a ``source_name`` param:

        - ``"input_kind"``: ``"url" | "discovered" | "asset"`` — which
          widget to render.
        - ``"placeholder"``: placeholder text for ``url`` inputs.
        - ``"help"``: short help text displayed under the control.
        - ``"probe"``: bool — call ``/input-sources/{id}/sources/{name}
          /resolution`` on change to validate and show the resolution.
        - ``"discovery_endpoint"``: override for ``discovered`` lists
          (defaults to ``/api/v1/input-sources/{source_id}/sources``).
        """
        return []

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
