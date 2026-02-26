"""
Syphon I/O module for GPU texture sharing on macOS.

This module provides classes for receiving textures via Syphon,
enabling real-time video sharing between applications like TouchDesigner,
Resolume, and Python-based processing pipelines.

Requires: syphon-python (pip install syphon-python)
Platform: macOS only (requires macOS 11+)

Note: The receiver module imports macOS-only dependencies (Metal, syphon)
at module level. Import SyphonReceiver only on macOS or inside try/except.
"""

__all__ = ["SyphonSender"]


def __getattr__(name: str):
    if name == "SyphonSender":
        from .sender import SyphonSender

        return SyphonSender
    raise AttributeError(name)
