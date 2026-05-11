"""Output sinks for Scope.

Output sinks send processed video frames to external destinations like Spout, NDI, etc.
"""

from .interface import OutputSink

# Sink modes that publish frames to local OS-level IPC (Spout on Windows,
# NDI on the LAN, Syphon on macOS) rather than the WebRTC peer connection.
# These always run on the user's machine, even when the pipeline executes
# on a cloud runner — so the cloud-relay code strips them from the graph
# sent to the runner and re-creates the senders locally.
HARDWARE_SINK_MODES: frozenset[str] = frozenset({"spout", "ndi", "syphon"})

__all__ = [
    "HARDWARE_SINK_MODES",
    "OutputSink",
    "get_available_output_sinks",
    "get_output_sink_classes",
]


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

    try:
        from .syphon import SyphonOutputSink

        sinks[SyphonOutputSink.source_id] = SyphonOutputSink
    except Exception:
        pass

    return sinks


def get_available_output_sinks() -> list[type[OutputSink]]:
    """Return a list of all built-in OutputSink subclasses that are available."""
    return [cls for cls in get_output_sink_classes().values() if cls.is_available()]
