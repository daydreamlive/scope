"""
Spout Sender - Sends textures to Spout receivers.

This module provides a simple interface for sending textures to
Spout-compatible applications like TouchDesigner, Resolume, etc.

Note: The Python SpoutGL wrapper API differs from the C++ SDK.
See: https://github.com/jlai/Python-SpoutGL
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Check if SpoutGL is available
try:
    import SpoutGL

    SPOUT_AVAILABLE = True
    # Log available methods for debugging
    logger.debug(f"SpoutGL sender methods: {dir(SpoutGL.SpoutSender())}")
except ImportError:
    SPOUT_AVAILABLE = False
    logger.warning("SpoutGL not available. Install with: pip install SpoutGL pyopengl")


class SpoutSender:
    """
    Sends textures to Spout receivers.

    Example usage:
        sender = SpoutSender("MyApp", 1920, 1080)
        sender.create()

        while running:
            # frame: (H, W, C) numpy array or torch tensor
            sender.send(frame)

        sender.release()
    """

    def __init__(self, name: str, width: int, height: int):
        """
        Initialize the Spout sender.

        Args:
            name: Name of this Spout sender (visible to receivers)
            width: Width of the output texture
            height: Height of the output texture
        """
        if not SPOUT_AVAILABLE:
            raise RuntimeError(
                "SpoutGL is not available. Install with: pip install SpoutGL pyopengl"
            )

        self.name = name
        self.width = width
        self.height = height
        self.sender = None
        self._frame_count = 0
        self._is_initialized = False

    def create(self) -> bool:
        """
        Create and initialize the Spout sender.

        Returns:
            True if creation was successful
        """
        try:
            self.sender = SpoutGL.SpoutSender()
            logger.info("SpoutSender object created")

            # Initialize OpenGL context (required!)
            if hasattr(self.sender, "createOpenGL"):
                result = self.sender.createOpenGL()
                logger.info(f"OpenGL context created for sender: {result}")
            else:
                logger.warning("createOpenGL not available on SpoutSender")

            # Set sender name
            if hasattr(self.sender, "setSenderName"):
                self.sender.setSenderName(self.name)
                logger.info(f"Sender name set to: {self.name}")
            else:
                logger.warning("setSenderName not available on SpoutSender")

            self._is_initialized = True

            logger.info(
                f"SpoutSender '{self.name}' created ({self.width}x{self.height})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create SpoutSender: {e}", exc_info=True)
            return False

    def send(self, frame: np.ndarray | torch.Tensor) -> bool:
        """
        Send a frame to Spout receivers.

        Args:
            frame: Image data as numpy array or torch tensor.
                   Expected format: (H, W, C) with C = 3 (RGB) or 4 (RGBA)
                   Values should be in [0, 1] float or [0, 255] uint8 range.

        Returns:
            True if send was successful
        """
        if self.sender is None:
            return False

        try:
            # Convert torch tensor to numpy if needed
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()

            # Ensure correct shape
            if frame.ndim != 3:
                logger.error(f"Expected 3D array (H, W, C), got shape {frame.shape}")
                return False

            h, w, c = frame.shape

            # Convert float [0, 1] to uint8 [0, 255]
            if frame.dtype in (np.float32, np.float64, np.float16):
                frame = (frame * 255).clip(0, 255).astype(np.uint8)

            # Handle different channel counts
            if c == 3:
                # RGB -> RGBA (add alpha channel)
                frame = np.concatenate(
                    [frame, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2
                )
            elif c != 4:
                logger.error(f"Expected 3 or 4 channels, got {c}")
                return False

            # Ensure contiguous array
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)

            # Update dimensions if changed
            if w != self.width or h != self.height:
                self.width = w
                self.height = h

            # Python SpoutGL API - try common signatures
            # Based on receiver API, likely: sendImage(buffer, width, height, GL_format, invert, hostFBO)
            GL_RGBA = 0x1908  # OpenGL constant for RGBA format

            try:
                # Try: sendImage(buffer, width, height, GL_format, invert, hostFBO)
                result = self.sender.sendImage(
                    frame,  # numpy array buffer
                    w,  # width
                    h,  # height
                    GL_RGBA,  # GL format
                    False,  # Don't invert
                    0,  # Host FBO
                )
            except TypeError:
                try:
                    # Alt: sendImage(buffer, GL_format, invert, hostFBO) - like receiver
                    result = self.sender.sendImage(
                        frame,
                        GL_RGBA,
                        False,
                        0,
                    )
                except TypeError:
                    # Last resort: minimal args
                    result = self.sender.sendImage(frame, w, h)

            if result:
                self._frame_count += 1
                return True

            return False

        except Exception as e:
            logger.error(f"Error sending Spout frame: {e}")
            return False

    def resize(self, width: int, height: int):
        """
        Update the sender resolution.

        Args:
            width: New width
            height: New height
        """
        self.width = width
        self.height = height
        logger.info(f"SpoutSender '{self.name}' resized to {width}x{height}")

    def is_initialized(self) -> bool:
        """Check if the sender is initialized."""
        return self._is_initialized

    def get_frame_count(self) -> int:
        """Get the number of frames sent."""
        return self._frame_count

    def get_stats(self) -> dict:
        """Get sender statistics."""
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "frames_sent": self._frame_count,
            "initialized": self._is_initialized,
        }

    def release(self):
        """Release the Spout sender resources."""
        if self.sender is not None:
            try:
                if hasattr(self.sender, "releaseSender"):
                    self.sender.releaseSender()
                if hasattr(self.sender, "closeOpenGL"):
                    self.sender.closeOpenGL()
                logger.info(f"SpoutSender '{self.name}' released")
            except Exception as e:
                logger.error(f"Error releasing SpoutSender: {e}")
            finally:
                self.sender = None
                self._is_initialized = False

    def __enter__(self):
        """Context manager entry."""
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
