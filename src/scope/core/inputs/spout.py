"""Spout input source implementation (Windows)."""

import logging
from typing import ClassVar

import numpy as np

from .interface import InputSource, InputSourceInfo

logger = logging.getLogger(__name__)


class SpoutInputSource(InputSource):
    """Input source that receives video frames via Spout on Windows.

    Wraps the existing SpoutReceiver from scope.server.spout.
    Requires SpoutGL to be installed (pip install SpoutGL pyopengl).
    """

    source_id: ClassVar[str] = "spout"
    source_name: ClassVar[str] = "Spout"
    source_description: ClassVar[str] = (
        "Receive video frames from Spout senders on Windows "
        "like TouchDesigner, Resolume, OBS, etc."
    )

    def __init__(self):
        self._receiver = None
        self._connected = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if SpoutGL is installed."""
        try:
            import SpoutGL  # noqa: F401

            return True
        except ImportError:
            return False

    def list_sources(self, timeout_ms: int = 5000) -> list[InputSourceInfo]:
        """List available Spout senders."""
        try:
            from scope.server.spout.receiver import list_senders

            sender_names = list_senders()
            return [
                InputSourceInfo(name=name, identifier=name) for name in sender_names
            ]
        except ImportError:
            logger.warning("SpoutGL not available, cannot list senders")
            return []
        except Exception as e:
            logger.error(f"Error listing Spout senders: {e}")
            return []

    def connect(self, identifier: str) -> bool:
        """Connect to a Spout sender by name. Empty string connects to the active sender."""
        try:
            from scope.server.spout.receiver import SpoutReceiver

            self.disconnect()

            self._receiver = SpoutReceiver(name=identifier, width=512, height=512)
            if self._receiver.create():
                self._connected = True
                logger.info(f"SpoutInputSource connected to '{identifier or 'any'}'")
                return True
            else:
                logger.error("Failed to create SpoutReceiver")
                self._receiver = None
                return False
        except ImportError:
            logger.error("SpoutGL not available")
            return False
        except Exception as e:
            logger.error(f"Error connecting SpoutInputSource: {e}")
            self._receiver = None
            return False

    def receive_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        """Receive a video frame. Returns (H, W, 3) RGB uint8 or None."""
        if self._receiver is None or not self._connected:
            return None

        try:
            return self._receiver.receive(as_rgb=True)
        except Exception as e:
            logger.error(f"Error receiving Spout frame: {e}")
            return None

    def disconnect(self):
        """Disconnect from the current Spout sender."""
        if self._receiver is not None:
            try:
                self._receiver.release()
            except Exception as e:
                logger.error(f"Error releasing SpoutReceiver: {e}")
            finally:
                self._receiver = None
                self._connected = False
