"""
Spout I/O module for bidirectional texture sharing.

This module provides classes for sending and receiving textures via Spout,
enabling real-time video sharing between applications like TouchDesigner,
Resolume, and Python-based processing pipelines.

Requires: SpoutGL (pip install SpoutGL)
Platform: Windows only
"""

from .receiver import SpoutReceiver
from .sender import SpoutSender

__all__ = ["SpoutReceiver", "SpoutSender"]
