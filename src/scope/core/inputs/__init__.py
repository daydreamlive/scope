"""Input sources for Scope.

Input sources provide video frames from external sources like NDI, Spout, etc.
"""

from .interface import InputSource, InputSourceInfo

__all__ = ["InputSource", "InputSourceInfo"]


def get_available_input_sources() -> list[type[InputSource]]:
    """Return a list of all built-in InputSource subclasses that are available."""
    sources: list[type[InputSource]] = []

    try:
        from .ndi import NDIInputSource

        if NDIInputSource.is_available():
            sources.append(NDIInputSource)
    except Exception:
        pass

    try:
        from .spout import SpoutInputSource

        if SpoutInputSource.is_available():
            sources.append(SpoutInputSource)
    except Exception:
        pass

    return sources
