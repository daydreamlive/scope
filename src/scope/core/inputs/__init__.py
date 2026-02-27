"""Input sources for Scope.

Input sources provide video frames from external sources like NDI, Spout, etc.
"""

from .interface import InputSource, InputSourceInfo

__all__ = [
    "InputSource",
    "InputSourceInfo",
    "get_input_source_classes",
    "get_available_input_sources",
]


def get_input_source_classes() -> dict[str, type[InputSource]]:
    """Get a mapping of source_id -> InputSource subclass for all built-in input sources.

    Returns all known classes regardless of whether they are available on this platform.
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

    return sources


def get_available_input_sources() -> list[type[InputSource]]:
    """Return a list of all built-in InputSource subclasses that are available."""
    return [cls for cls in get_input_source_classes().values() if cls.is_available()]
