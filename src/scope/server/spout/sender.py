"""
Spout Sender - Sends textures to Spout receivers.

This module provides a simple interface for sending textures to
Spout-compatible applications like TouchDesigner, Resolume, etc.

Note: The Python SpoutGL wrapper API differs from the C++ SDK.
See: https://github.com/jlai/Python-SpoutGL

Raises:
    ImportError: If SpoutGL is not installed. Install with: pip install SpoutGL pyopengl
"""

import logging

import numpy as np
import SpoutGL
import torch

logger = logging.getLogger(__name__)


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
            # Handle torch tensor input - do format conversion on GPU for performance

            if isinstance(frame, torch.Tensor):
                frame = self._prepare_tensor_on_gpu(frame)
            else:
                # NumPy array path (fallback)
                frame = self._prepare_numpy_array(frame)

            if frame is None:
                return False

            h, w, c = frame.shape

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

    def _prepare_tensor_on_gpu(self, frame: torch.Tensor) -> np.ndarray | None:
        """
        Prepare a torch tensor for Spout by doing format conversion on GPU.

        Args:
            frame: PyTorch tensor (H, W, C), float [0,1] or uint8 [0,255]

        Returns:
            NumPy array ready for sendImage, or None on error
        """
        # Ensure correct shape
        if frame.ndim != 3:
            logger.error(f"Expected 3D tensor (H, W, C), got shape {frame.shape}")
            return None

        h, w, c = frame.shape

        # Validate channel count
        if c not in (3, 4):
            logger.error(f"Expected 3 or 4 channels, got {c}")
            return None

        # Detach from computation graph
        frame = frame.detach()

        # Convert float [0, 1] to uint8 [0, 255] on GPU
        if frame.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            frame = (frame * 255).clamp(0, 255).to(torch.uint8)
        elif frame.dtype != torch.uint8:
            # Convert other integer types to uint8
            frame = frame.clamp(0, 255).to(torch.uint8)

        # Add alpha channel on GPU if needed (RGB -> RGBA)
        if c == 3:
            alpha = torch.full((h, w, 1), 255, dtype=torch.uint8, device=frame.device)
            frame = torch.cat([frame, alpha], dim=-1)

        # Ensure contiguous before transfer
        frame = frame.contiguous()

        # Transfer to CPU and convert to numpy
        # This is now much smaller (uint8 vs float32) and already formatted
        return frame.cpu().numpy()

    def _prepare_numpy_array(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Prepare a numpy array for Spout (CPU path fallback).

        Args:
            frame: NumPy array (H, W, C), float [0,1] or uint8 [0,255]

        Returns:
            NumPy array ready for sendImage, or None on error
        """
        # Ensure correct shape
        if frame.ndim != 3:
            logger.error(f"Expected 3D array (H, W, C), got shape {frame.shape}")
            return None

        h, w, c = frame.shape

        # Validate channel count
        if c not in (3, 4):
            logger.error(f"Expected 3 or 4 channels, got {c}")
            return None

        # Convert float [0, 1] to uint8 [0, 255]
        if frame.dtype in (np.float32, np.float64, np.float16):
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        # Add alpha channel if needed (RGB -> RGBA)
        if c == 3:
            frame = np.concatenate(
                [frame, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2
            )

        return frame

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
