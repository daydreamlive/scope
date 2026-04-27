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
    """Get a mapping of source_id -> InputSource subclass.

    Merges built-in sources with plugin-registered ones (any InputSource
    subclass passed to the ``register_nodes`` hook). A plugin cannot
    shadow a built-in: on id collision the built-in wins.
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

    try:
        from scope.core.plugins import get_plugin_input_sources

        for source_id, cls in get_plugin_input_sources().items():
            if source_id in sources:
                continue  # built-in already registered — don't let plugins override
            sources[source_id] = cls
    except Exception:
        pass

    return sources


def get_available_input_sources() -> list[type[InputSource]]:
    """Return a list of all built-in InputSource subclasses that are available."""
    return [cls for cls in get_input_source_classes().values() if cls.is_available()]
