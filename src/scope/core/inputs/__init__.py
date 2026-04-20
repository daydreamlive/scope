"""Input sources for Scope.

Input sources provide video frames from external sources like NDI, Spout, etc.
"""

from .interface import (
    InputSource,
    InputSourceError,
    InputSourceInfo,
    InvalidSourceURLError,
    SourceUnavailableError,
)

__all__ = [
    "InputSource",
    "InputSourceInfo",
    "InputSourceError",
    "InvalidSourceURLError",
    "SourceUnavailableError",
    "get_input_source_classes",
    "get_available_input_sources",
]


def get_input_source_classes() -> dict[str, type[InputSource]]:
    """Get a mapping of source_id -> InputSource subclass.

    Merges built-in sources (discovered via try-imports below) with any
    ``InputSource`` subclass registered through :class:`NodeRegistry`
    (e.g. by a plugin's ``register_nodes`` hook).

    Returns all known classes regardless of whether they are available on
    this platform.
    """
    sources: dict[str, type[InputSource]] = {}

    try:
        from .ndi import NDIInputSource

        sources[NDIInputSource.source_id] = NDIInputSource
    except Exception:
        pass

    try:
        from .spout import SpoutInputSource

        sources[SpoutInputSource.source_id] = SpoutInputSource
    except Exception:
        pass

    try:
        from .syphon import SyphonInputSource

        sources[SyphonInputSource.source_id] = SyphonInputSource
    except Exception:
        pass

    try:
        from .video_file import VideoFileInputSource

        sources[VideoFileInputSource.source_id] = VideoFileInputSource
    except Exception:
        pass

    # Merge plugin-registered sources from the unified NodeRegistry.
    # Plugin sources override built-ins with the same source_id so a
    # plugin can replace a broken built-in.
    try:
        from scope.core.nodes.registry import NodeRegistry

        for node_type_id in NodeRegistry.list_node_types():
            cls = NodeRegistry.get(node_type_id)
            if (
                cls is not None
                and isinstance(cls, type)
                and issubclass(cls, InputSource)
            ):
                sources[cls.source_id] = cls
    except Exception:
        pass

    return sources


def get_available_input_sources() -> list[type[InputSource]]:
    """Return a list of all built-in InputSource subclasses that are available."""
    return [cls for cls in get_input_source_classes().values() if cls.is_available()]
