"""Output sinks for Scope.

Output sinks send processed video frames to external destinations like Spout, NDI, etc.
"""

from .interface import OutputSink

__all__ = ["OutputSink", "get_output_sink_classes", "get_available_output_sinks"]


def get_output_sink_classes() -> dict[str, type[OutputSink]]:
    """Get a mapping of source_id -> OutputSink subclass for all built-in output sinks."""
    sinks: dict[str, type[OutputSink]] = {}

    try:
        from .spout import SpoutOutputSink

        sinks[SpoutOutputSink.source_id] = SpoutOutputSink
    except Exception:
        pass

    try:
        from .ndi import NDIOutputSink

        sinks[NDIOutputSink.source_id] = NDIOutputSink
    except Exception:
        pass

    return sinks


def get_available_output_sinks() -> list[type[OutputSink]]:
    """Return a list of all built-in OutputSink subclasses that are available."""
    return [cls for cls in get_output_sink_classes().values() if cls.is_available()]
