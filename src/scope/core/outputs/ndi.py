"""NDI output sink implementation.

Sends processed video frames and audio over the network via NDI.
Uses the shared NDI ctypes bindings from scope.core.ndi.
"""

import ctypes
import logging
from typing import ClassVar

import numpy as np
import torch

from scope.core.ndi import (
    NDI_FOURCC_RGBA,
    NDIlib_audio_frame_v2_t,
    NDIlib_send_create_t,
    NDIlib_video_frame_v2_t,
    load_library,
    setup_send_functions,
)
from scope.core.ndi import (
    is_available as ndi_is_available,
)

from .interface import OutputSink

logger = logging.getLogger(__name__)


class NDIOutputSink(OutputSink):
    """Output sink that sends video frames over the network via NDI.

    The sender name appears on the network for NDI receivers to discover.
    """

    source_id: ClassVar[str] = "ndi"
    source_name: ClassVar[str] = "NDI"
    source_description: ClassVar[str] = (
        "Send video frames via NDI (Network Device Interface) "
        "to receivers on the local network."
    )

    def __init__(self):
        self._send_instance = None
        self._name = ""
        self._width = 0
        self._height = 0
        self._lib = None

    @classmethod
    def is_available(cls) -> bool:
        return ndi_is_available()

    @property
    def name(self) -> str:
        return self._name

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def create(self, name: str, width: int, height: int) -> bool:
        """Create the NDI sender."""
        try:
            self.close()

            lib = load_library()
            setup_send_functions(lib)
            self._lib = lib

            if not self._lib.NDIlib_initialize():
                logger.error("Failed to initialize NDI library for sender")
                return False

            send_create = NDIlib_send_create_t()
            send_create.p_ndi_name = name.encode("utf-8")
            send_create.p_groups = None
            send_create.clock_video = True
            send_create.clock_audio = True

            self._send_instance = self._lib.NDIlib_send_create(
                ctypes.byref(send_create)
            )
            if not self._send_instance:
                logger.error(f"Failed to create NDI sender '{name}'")
                return False

            self._name = name
            self._width = width
            self._height = height
            logger.info(f"NDIOutputSink created: '{name}' ({width}x{height})")
            return True

        except Exception as e:
            logger.error(f"Error creating NDIOutputSink: {e}")
            self._send_instance = None
            return False

    def send_frame(self, frame: np.ndarray | torch.Tensor) -> bool:
        """Send a video frame over NDI."""
        if self._send_instance is None or self._lib is None:
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

            # Convert RGB to RGBA (NDI expects RGBA)
            if channels == 3:
                rgba = np.empty((h, w, 4), dtype=np.uint8)
                rgba[:, :, :3] = frame
                rgba[:, :, 3] = 255
                frame = rgba
            elif channels == 1:
                rgba = np.empty((h, w, 4), dtype=np.uint8)
                rgba[:, :, 0] = frame[:, :, 0]
                rgba[:, :, 1] = frame[:, :, 0]
                rgba[:, :, 2] = frame[:, :, 0]
                rgba[:, :, 3] = 255
                frame = rgba

            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)

            video_frame = NDIlib_video_frame_v2_t()
            video_frame.xres = w
            video_frame.yres = h
            video_frame.FourCC = NDI_FOURCC_RGBA
            video_frame.frame_rate_N = 30000
            video_frame.frame_rate_D = 1001
            video_frame.picture_aspect_ratio = 0.0
            video_frame.frame_format_type = 1  # progressive
            video_frame.timecode = -1  # auto
            video_frame.p_data = frame.ctypes.data
            video_frame.line_stride_in_bytes = w * 4
            video_frame.p_metadata = None
            video_frame.timestamp = -1  # auto

            self._lib.NDIlib_send_send_video_v2(
                self._send_instance, ctypes.byref(video_frame)
            )
            return True

        except Exception as e:
            logger.error(f"Error sending NDI frame: {e}")
            return False

    def send_audio(
        self,
        audio: np.ndarray | torch.Tensor,
        sample_rate: int,
        num_channels: int,
    ) -> bool:
        """Send audio samples over NDI.

        Args:
            audio: Float32 audio samples. Shape (S,) for mono or (C, S) for multi-channel.
                   Values should be in [-1.0, 1.0] range.
            sample_rate: Audio sample rate (e.g. 48000).
            num_channels: Number of audio channels (e.g. 1 for mono).

        Returns:
            True if send was successful.
        """
        if self._send_instance is None or self._lib is None:
            return False

        try:
            if isinstance(audio, torch.Tensor):
                if audio.is_cuda:
                    audio = audio.cpu()
                audio = audio.numpy()

            audio = np.asarray(audio, dtype=np.float32)

            # Ensure contiguous
            if not audio.flags["C_CONTIGUOUS"]:
                audio = np.ascontiguousarray(audio)

            # NDI expects interleaved float32 samples
            # For mono: shape (S,), for multi-channel: shape (C*S,) interleaved
            num_samples = audio.shape[-1] if audio.ndim > 1 else len(audio)

            audio_frame = NDIlib_audio_frame_v2_t()
            audio_frame.sample_rate = sample_rate
            audio_frame.no_channels = num_channels
            audio_frame.no_samples = num_samples
            audio_frame.timecode = -1  # auto
            audio_frame.p_data = audio.ctypes.data
            audio_frame.channel_stride_in_bytes = num_samples * 4  # float32 = 4 bytes
            audio_frame.p_metadata = None
            audio_frame.timestamp = -1  # auto

            self._lib.NDIlib_send_send_audio_v2(
                self._send_instance, ctypes.byref(audio_frame)
            )
            return True

        except Exception as e:
            logger.error(f"Error sending NDI audio: {e}")
            return False

    def resize(self, width: int, height: int):
        """Update output dimensions (NDI rebuilds frame struct per-send)."""
        self._width = width
        self._height = height

    def close(self):
        """Release NDI sender resources."""
        if self._send_instance is not None and self._lib is not None:
            try:
                self._lib.NDIlib_send_destroy(self._send_instance)
            except Exception as e:
                logger.error(f"Error destroying NDI sender: {e}")
            finally:
                self._send_instance = None
                self._name = ""
                self._width = 0
                self._height = 0
