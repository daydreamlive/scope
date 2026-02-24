"""Virtual camera output sink implementation.

Sends processed video frames to a virtual camera device that appears
as a standard webcam in other applications (Zoom, OBS, Discord, etc.).

Uses pyvirtualcam which supports:
- Windows: OBS Virtual Camera (built into OBS 26+) or Unity Capture
- macOS: OBS Virtual Camera
- Linux: v4l2loopback kernel module
"""

import logging
from typing import ClassVar

import numpy as np
import torch

from .interface import OutputSink

logger = logging.getLogger(__name__)

_virtualcam_available: bool | None = None


def is_available() -> bool:
    """Check if pyvirtualcam is installed and a backend is available.

    On Linux, this checks if v4l2loopback devices exist.
    On Windows/macOS, this checks if OBS Virtual Camera is installed.
    """
    global _virtualcam_available
    if _virtualcam_available is not None:
        return _virtualcam_available

    try:
        import pyvirtualcam  # noqa: F401
    except ImportError:
        _virtualcam_available = False
        return False

    # Check if a backend is actually available by trying to create a camera
    # Use a small test resolution to minimize resource usage
    try:
        with pyvirtualcam.Camera(width=320, height=240, fps=1, print_fps=False):
            pass
        _virtualcam_available = True
        return True
    except RuntimeError:
        # Backend not available (e.g., v4l2loopback not loaded on Linux,
        # OBS not installed on Windows/macOS)
        _virtualcam_available = False
        return False
    except Exception:
        # Any other error, assume not available
        _virtualcam_available = False
        return False


class VirtualCameraOutputSink(OutputSink):
    """Output sink that sends video frames to a virtual camera device.

    The virtual camera appears as a standard webcam in any application
    that supports camera input (Zoom, Google Meet, OBS, Discord, etc.).
    """

    source_id: ClassVar[str] = "virtualcam"
    source_name: ClassVar[str] = "Virtual Camera"
    source_description: ClassVar[str] = (
        "Send video frames to a virtual camera that appears as a webcam "
        "in other applications like Zoom, OBS, Discord, etc."
    )

    def __init__(self):
        self._cam = None
        self._name = ""
        self._width = 0
        self._height = 0
        self._fps = 30
        self._device_name = ""

    @classmethod
    def is_available(cls) -> bool:
        """Check if pyvirtualcam and a virtual camera backend are available."""
        return is_available()

    @property
    def name(self) -> str:
        return self._name

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def device_name(self) -> str:
        """The actual device name reported by the virtual camera backend."""
        return self._device_name

    def create(self, name: str, width: int, height: int) -> bool:
        """Create the virtual camera device."""
        try:
            import pyvirtualcam

            self.close()

            # pyvirtualcam auto-selects the best backend for the platform
            # - Windows: 'obs' (OBS Virtual Camera) or 'unitycapture'
            # - macOS: 'obs'
            # - Linux: 'v4l2loopback'
            self._cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=self._fps,
                fmt=pyvirtualcam.PixelFormat.RGB,
                print_fps=False,
            )

            self._name = name
            self._width = width
            self._height = height
            self._device_name = self._cam.device

            logger.info(
                f"VirtualCameraOutputSink created: '{self._device_name}' "
                f"({width}x{height} @ {self._fps}fps)"
            )
            return True

        except ImportError:
            logger.error("pyvirtualcam not installed")
            return False
        except RuntimeError as e:
            # pyvirtualcam raises RuntimeError when no backend is available
            logger.error(f"Virtual camera backend not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error creating VirtualCameraOutputSink: {e}")
            self._cam = None
            return False

    def send_frame(self, frame: np.ndarray | torch.Tensor) -> bool:
        """Send a video frame to the virtual camera."""
        if self._cam is None:
            return False

        try:
            # Convert torch tensor to numpy
            if isinstance(frame, torch.Tensor):
                if frame.is_cuda:
                    frame = frame.cpu()
                frame = frame.numpy()

            # Ensure uint8
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)

            h, w = frame.shape[:2]
            channels = frame.shape[2] if frame.ndim == 3 else 1

            # Ensure RGB (3 channels)
            if channels == 4:
                # RGBA -> RGB
                frame = frame[:, :, :3]
            elif channels == 1:
                # Grayscale -> RGB
                frame = np.repeat(frame, 3, axis=2)

            # Resize if dimensions changed (pyvirtualcam has fixed dimensions)
            if w != self._width or h != self._height:
                from PIL import Image

                img = Image.fromarray(frame)
                img = img.resize((self._width, self._height), Image.BILINEAR)
                frame = np.array(img)

            # Ensure contiguous memory
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)

            self._cam.send(frame)
            return True

        except Exception as e:
            logger.error(f"Error sending virtual camera frame: {e}")
            return False

    def resize(self, width: int, height: int):
        """Resize the virtual camera.

        Note: pyvirtualcam doesn't support runtime resize, so frames are
        resized in send_frame(). Full recreation happens via frame_processor.
        """
        if width != self._width or height != self._height:
            logger.info(
                f"VirtualCamera resize requested: {self._width}x{self._height} -> "
                f"{width}x{height}. Frames will be scaled."
            )
            self._width = width
            self._height = height

    def close(self):
        """Release virtual camera resources."""
        if self._cam is not None:
            try:
                self._cam.close()
                logger.info(f"VirtualCameraOutputSink closed: '{self._device_name}'")
            except Exception as e:
                logger.error(f"Error closing virtual camera: {e}")
            finally:
                self._cam = None
                self._name = ""
                self._width = 0
                self._height = 0
                self._device_name = ""
