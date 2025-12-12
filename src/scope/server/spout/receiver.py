"""
Spout Receiver - Receives textures from Spout senders.

This module provides a simple interface for receiving textures from
Spout-compatible applications like TouchDesigner, Resolume, etc.

Note: The Python SpoutGL wrapper API differs from the C++ SDK.
See: https://github.com/jlai/Python-SpoutGL

Raises:
    ImportError: If SpoutGL is not installed. Install with: pip install SpoutGL pyopengl
"""

import logging

import numpy as np
import SpoutGL

logger = logging.getLogger(__name__)


class SpoutReceiver:
    """
    Receives textures from a Spout sender.

    Example usage:
        receiver = SpoutReceiver("TouchDesigner")
        receiver.create()

        while running:
            frame = receiver.receive()
            if frame is not None:
                # Process frame (H, W, C) numpy array in [0, 255] uint8
                pass

        receiver.release()
    """

    def __init__(self, name: str = "", width: int = 1920, height: int = 1080):
        """
        Initialize the Spout receiver.

        Args:
            name: Name of the Spout sender to connect to.
                        Empty string connects to the active sender.
            width: Initial width for the receive buffer
            height: Initial height for the receive buffer
        """
        self.name = name
        self.width = width
        self.height = height
        self.receiver = None
        self._buffer = None
        self._is_connected = False
        self._frame_count = 0
        self._connected_sender = ""

    def create(self) -> bool:
        """
        Create and initialize the Spout receiver.

        Returns:
            True if creation was successful
        """
        try:
            self.receiver = SpoutGL.SpoutReceiver()

            # Initialize OpenGL context (required!)
            if hasattr(self.receiver, "createOpenGL"):
                self.receiver.createOpenGL()
                logger.info("OpenGL context created for receiver")

            # Set receiver name if specified (connects to specific sender)
            if self.name and hasattr(self.receiver, "setReceiverName"):
                self.receiver.setReceiverName(self.name)

            # Allocate initial buffer (RGBA = 4 channels)
            self._buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)

            logger.info(
                f"SpoutReceiver created, looking for sender: "
                f"'{self.name or 'any'}' ({self.width}x{self.height})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create SpoutReceiver: {e}")
            return False

    def receive(self, as_rgb: bool = False) -> np.ndarray | None:
        """
        Receive a frame from the Spout sender.

        Args:
            as_rgb: If True, return RGB (3 channels) instead of RGBA (4 channels).
                   This is more efficient when you only need RGB as it avoids
                   an extra copy/slice operation.

        Returns:
            Frame as numpy array (H, W, C) in uint8 [0, 255] format,
            or None if no frame is available.
        """
        if self.receiver is None:
            return None

        try:
            # Python SpoutGL API signature:
            # receiveImage(buffer: Optional[Buffer], GL_format: int, invert: bool, hostFBO: int) -> bool
            GL_RGBA = 0x1908  # OpenGL constant for RGBA format

            result = self.receiver.receiveImage(
                self._buffer,  # numpy array as buffer
                GL_RGBA,  # GL format constant (int)
                False,  # invert (bool)
                0,  # hostFBO (int)
            )

            if result:
                self._is_connected = True

                # Check if we need to resize (sender may have different dimensions)
                if hasattr(self.receiver, "isUpdated") and self.receiver.isUpdated():
                    self._handle_size_update()

                self._frame_count += 1

                if as_rgb:
                    # Return RGB copy directly (more efficient than copy + slice)
                    return np.ascontiguousarray(self._buffer[:, :, :3])
                else:
                    return self._buffer.copy()

            return None

        except Exception as e:
            logger.error(f"Error receiving Spout frame: {e}")
            return None

    def _handle_size_update(self):
        """Handle sender size changes."""
        try:
            new_width = self.width
            new_height = self.height

            if hasattr(self.receiver, "getSenderWidth"):
                new_width = self.receiver.getSenderWidth()
            if hasattr(self.receiver, "getSenderHeight"):
                new_height = self.receiver.getSenderHeight()

            if new_width != self.width or new_height != self.height:
                self.width = new_width
                self.height = new_height
                self._buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
                logger.info(f"Buffer resized to: {self.width}x{self.height}")

            if hasattr(self.receiver, "getSenderName"):
                self._connected_sender = self.receiver.getSenderName()

        except Exception as e:
            logger.warning(f"Could not handle size update: {e}")

    def is_connected(self) -> bool:
        """Check if connected to a sender."""
        return self._is_connected and self.receiver is not None

    def get_sender_name(self) -> str:
        """Get the name of the connected sender."""
        if self._connected_sender:
            return self._connected_sender

        if self.receiver is not None:
            try:
                if hasattr(self.receiver, "getSenderName"):
                    return self.receiver.getSenderName()
            except Exception:
                pass
        return ""

    def get_resolution(self) -> tuple[int, int]:
        """Get the current resolution (width, height)."""
        return (self.width, self.height)

    def get_frame_count(self) -> int:
        """Get the number of frames received."""
        return self._frame_count

    def release(self):
        """Release the Spout receiver resources."""
        if self.receiver is not None:
            try:
                if hasattr(self.receiver, "releaseReceiver"):
                    self.receiver.releaseReceiver()
                if hasattr(self.receiver, "closeOpenGL"):
                    self.receiver.closeOpenGL()
                logger.info("SpoutReceiver released")
            except Exception as e:
                logger.error(f"Error releasing SpoutReceiver: {e}")
            finally:
                self.receiver = None
                self._is_connected = False

    def __enter__(self):
        """Context manager entry."""
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def list_senders() -> list[str]:
    """
    List all available Spout senders.

    Returns:
        List of sender names currently available.
    """
    try:
        receiver = SpoutGL.SpoutReceiver()

        # Initialize OpenGL context
        if hasattr(receiver, "createOpenGL"):
            result = receiver.createOpenGL()
            logger.debug(f"createOpenGL result: {result}")

        # Get sender list
        if hasattr(receiver, "getSenderList"):
            senders = receiver.getSenderList()
            logger.info(f"getSenderList returned: {senders}")
            if hasattr(receiver, "closeOpenGL"):
                receiver.closeOpenGL()
            return senders if senders else []

        # Fallback: try getActiveSender
        if hasattr(receiver, "getActiveSender"):
            active = receiver.getActiveSender()
            logger.info(f"getActiveSender returned: {active}")
            if hasattr(receiver, "closeOpenGL"):
                receiver.closeOpenGL()
            return [active] if active else []

        if hasattr(receiver, "closeOpenGL"):
            receiver.closeOpenGL()
        return []

    except Exception as e:
        logger.warning(f"Could not list Spout senders: {e}", exc_info=True)
        return []
