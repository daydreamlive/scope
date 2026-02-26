"""Syphon input source implementation.

Uses the syphon-python library for macOS GPU texture sharing.
Requires macOS 11+ and the syphon-python package to be installed.
"""

import logging
import sys
from typing import ClassVar

import numpy as np

from .interface import InputSource, InputSourceInfo

logger = logging.getLogger(__name__)


class SyphonInputSource(InputSource):
    """Input source that receives video frames via Syphon on macOS.

    Wraps the SyphonReceiver from scope.server.syphon.
    Requires syphon-python to be installed (pip install syphon-python).
    """

    source_id: ClassVar[str] = "syphon"
    source_name: ClassVar[str] = "Syphon"
    source_description: ClassVar[str] = (
        "Receive video frames from Syphon servers on macOS. "
        "Compatible with apps like TouchDesigner, Resolume, OBS, etc."
    )

    def __init__(self):
        self._receiver = None
        self._connected = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if Syphon is available (macOS only, syphon-python installed).

        Also initializes the Syphon shared directory on the main thread so
        that worker threads can discover servers later.
        """
        if sys.platform != "darwin":
            return False
        try:
            import syphon  # noqa: F401

            from scope.server.syphon.receiver import ensure_directory_initialized

            ensure_directory_initialized()
            return True
        except ImportError:
            return False

    def list_sources(self, timeout_ms: int = 5000) -> list[InputSourceInfo]:
        """List available Syphon servers."""
        try:
            from scope.server.syphon.receiver import SyphonReceiver

            receiver = SyphonReceiver()
            try:
                servers = receiver.discover()
                return [
                    InputSourceInfo(
                        name=s.display_name,
                        identifier=s.uuid,
                        metadata={"app_name": s.app_name, "server_name": s.name},
                    )
                    for s in servers
                ]
            finally:
                receiver.release()
        except ImportError:
            logger.warning("syphon-python not available, cannot list servers")
            return []
        except Exception as e:
            logger.error(f"Error listing Syphon servers: {e}")
            return []

    def connect(self, identifier: str) -> bool:
        """Connect to a Syphon server by UUID or app name."""
        try:
            from scope.server.syphon.receiver import SyphonReceiver

            self.disconnect()

            self._receiver = SyphonReceiver()
            if self._receiver.connect(identifier):
                self._connected = True
                logger.info(
                    f"SyphonInputSource connected to '{identifier or 'first available'}'"
                )
                return True
            else:
                logger.error("Failed to connect SyphonReceiver")
                self._receiver = None
                return False
        except ImportError:
            logger.error("syphon-python not available")
            return False
        except Exception as e:
            logger.error(f"Error connecting SyphonInputSource: {e}")
            self._receiver = None
            return False

    def receive_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        """Receive a video frame. Returns (H, W, 3) RGB uint8 or None."""
        if self._receiver is None or not self._connected:
            return None

        try:
            return self._receiver.receive(as_rgb=True)
        except Exception as e:
            logger.error(f"Error receiving Syphon frame: {e}")
            return None

    def get_source_resolution(
        self, identifier: str, timeout_ms: int = 5000
    ) -> tuple[int, int] | None:
        """Probe a Syphon server's frame resolution by receiving one frame."""
        try:
            if not self.connect(identifier):
                return None

            import time

            elapsed = 0
            poll_interval_s = 0.1
            while elapsed < timeout_ms:
                frame = self.receive_frame(timeout_ms=100)
                if frame is not None:
                    h, w = frame.shape[:2]
                    return (w, h)
                time.sleep(poll_interval_s)
                elapsed += int(poll_interval_s * 1000)

            logger.warning(
                f"Timed out probing resolution for '{identifier}' after {timeout_ms}ms"
            )
            return None
        finally:
            self.disconnect()

    def disconnect(self):
        """Disconnect from the current Syphon server."""
        if self._receiver is not None:
            try:
                self._receiver.release()
            except Exception as e:
                logger.error(f"Error releasing SyphonReceiver: {e}")
            finally:
                self._receiver = None
                self._connected = False
