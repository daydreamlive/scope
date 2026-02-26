"""
Syphon Sender - Sends textures to Syphon receivers on macOS.

This module provides a simple interface for sending textures to
Syphon-compatible applications like TouchDesigner, Resolume, OBS, etc.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SyphonSender:
    """Sends textures to Syphon receivers."""

    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self.width = width
        self.height = height
        self.server = None
        self._texture = None
        self._copy_image_to_mtl_texture = None

    def create(self) -> bool:
        """Create and initialize the Syphon sender."""
        try:
            import syphon
            from syphon.utils.numpy import copy_image_to_mtl_texture

            self.server = syphon.SyphonMetalServer(self.name)
            self._copy_image_to_mtl_texture = copy_image_to_mtl_texture
            self._texture = self._create_texture(self.width, self.height)
            if self._texture is None:
                logger.error("Failed to create Syphon texture")
                self.release()
                return False

            logger.info(
                f"SyphonSender '{self.name}' created ({self.width}x{self.height})"
            )
            return True
        except ImportError:
            logger.error("syphon-python not available")
            return False
        except Exception as e:
            logger.error(f"Failed to create SyphonSender: {e}", exc_info=True)
            self.release()
            return False

    def send(self, frame: np.ndarray | torch.Tensor) -> bool:
        """Send a frame to Syphon receivers."""
        if self.server is None or self._texture is None:
            return False

        try:
            image = self._prepare_frame(frame)
            if image is None:
                return False

            h, w, _ = image.shape
            if w != self.width or h != self.height:
                self.resize(w, h)
                if self._texture is None:
                    return False

            self._copy_image_to_mtl_texture(image, self._texture)
            self.server.publish_frame_texture(
                self._texture,
                size=(self.width, self.height),
                is_flipped=True,
            )
            return True
        except Exception as e:
            logger.error(f"Error sending Syphon frame: {e}")
            return False

    def resize(self, width: int, height: int):
        """Update sender texture size."""
        self.width = width
        self.height = height
        if self.server is None:
            return

        self._texture = self._create_texture(width, height)
        if self._texture is None:
            logger.error(
                f"Failed to resize SyphonSender '{self.name}' to {width}x{height}"
            )
        else:
            logger.info(f"SyphonSender '{self.name}' resized to {width}x{height}")

    def release(self):
        """Release Syphon sender resources."""
        if self.server is not None:
            try:
                self.server.stop()
                logger.info(f"SyphonSender '{self.name}' released")
            except Exception as e:
                logger.error(f"Error releasing SyphonSender: {e}")
            finally:
                self.server = None
                self._texture = None
                self._copy_image_to_mtl_texture = None

    def _create_texture(self, width: int, height: int):
        """Create an RGBA Metal texture used as the publish buffer."""
        if self.server is None:
            return None
        try:
            import Metal

            descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
                Metal.MTLPixelFormatRGBA8Unorm, width, height, False
            )
            return self.server.device.newTextureWithDescriptor_(descriptor)
        except Exception as e:
            logger.error(f"Failed to create Metal texture: {e}")
            return None

    def _prepare_frame(self, frame: np.ndarray | torch.Tensor) -> np.ndarray | None:
        """Normalize input frame into contiguous RGBA uint8 numpy array."""
        if isinstance(frame, torch.Tensor):
            frame = self._prepare_tensor(frame)
        else:
            frame = self._prepare_numpy(frame)
        if frame is None:
            return None
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        return frame

    def _prepare_tensor(self, frame: torch.Tensor) -> np.ndarray | None:
        if frame.ndim != 3:
            logger.error(f"Expected 3D tensor (H, W, C), got shape {frame.shape}")
            return None

        h, w, c = frame.shape
        if c not in (3, 4):
            logger.error(f"Expected 3 or 4 channels, got {c}")
            return None

        frame = frame.detach()
        if frame.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            frame = (frame * 255).clamp(0, 255).to(torch.uint8)
        elif frame.dtype != torch.uint8:
            frame = frame.clamp(0, 255).to(torch.uint8)

        if c == 3:
            alpha = torch.full((h, w, 1), 255, dtype=torch.uint8, device=frame.device)
            frame = torch.cat([frame, alpha], dim=-1)

        return frame.contiguous().cpu().numpy()

    def _prepare_numpy(self, frame: np.ndarray) -> np.ndarray | None:
        if frame.ndim != 3:
            logger.error(f"Expected 3D array (H, W, C), got shape {frame.shape}")
            return None

        h, w, c = frame.shape
        if c not in (3, 4):
            logger.error(f"Expected 3 or 4 channels, got {c}")
            return None

        if frame.dtype in (np.float16, np.float32, np.float64):
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)

        if c == 3:
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            frame = np.concatenate([frame, alpha], axis=2)

        return frame
