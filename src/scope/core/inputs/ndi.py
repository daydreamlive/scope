"""NDI input source implementation.

Uses the shared NDI ctypes bindings from scope.core.ndi.
The user must install the NDI SDK/Tools on their system for this to work.
NDI SDK downloads: https://ndi.video/tools/
"""

import ctypes
import logging
from typing import ClassVar

import numpy as np

from scope.core.ndi import (
    NDI_BANDWIDTH_HIGHEST,
    NDI_COLOR_FORMAT_RGBX_RGBA,
    NDI_FOURCC_BGRA,
    NDI_FOURCC_BGRX,
    NDI_FOURCC_RGBA,
    NDI_FOURCC_RGBX,
    NDI_FRAME_TYPE_VIDEO,
    NDIlib_audio_frame_v2_t,
    NDIlib_find_create_t,
    NDIlib_metadata_frame_t,
    NDIlib_recv_create_v3_t,
    NDIlib_source_t,
    NDIlib_video_frame_v2_t,
    load_library,
    setup_recv_functions,
)
from scope.core.ndi import (
    is_available as ndi_is_available,
)

from .interface import InputSource, InputSourceInfo

logger = logging.getLogger(__name__)


class NDIInputSource(InputSource):
    """Input source that receives video frames via NDI.

    Uses ctypes to interface directly with the NDI SDK library.
    The user must install the NDI SDK on their system for this to work.
    """

    source_id: ClassVar[str] = "ndi"
    source_name: ClassVar[str] = "NDI"
    source_description: ClassVar[str] = (
        "Receive video frames via NDI (Network Device Interface). "
        "Requires the NDI SDK to be installed on the system. "
        "Download from https://ndi.video/tools/"
    )

    def __init__(self):
        self._lib = load_library()
        setup_recv_functions(self._lib)

        if not self._lib.NDIlib_initialize():
            raise RuntimeError("Failed to initialize NDI library")

        self._find_instance = None
        self._recv_instance = None
        self._connected_source_name: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if the NDI SDK is installed on this system."""
        return ndi_is_available()

    def list_sources(self, timeout_ms: int = 5000) -> list[InputSourceInfo]:
        """List available NDI sources on the network."""
        if self._find_instance is None:
            create_settings = NDIlib_find_create_t()
            create_settings.show_local_sources = True
            create_settings.p_groups = None
            create_settings.p_extra_ips = None

            self._find_instance = self._lib.NDIlib_find_create_v2(
                ctypes.byref(create_settings)
            )
            if not self._find_instance:
                logger.error("Failed to create NDI find instance")
                return []

        self._lib.NDIlib_find_wait_for_sources(self._find_instance, timeout_ms)

        num_sources = ctypes.c_uint32(0)
        sources_ptr = self._lib.NDIlib_find_get_current_sources(
            self._find_instance, ctypes.byref(num_sources)
        )

        sources = []
        for i in range(num_sources.value):
            source = sources_ptr[i]
            name = source.p_ndi_name.decode("utf-8") if source.p_ndi_name else ""
            url = source.p_url_address.decode("utf-8") if source.p_url_address else ""
            sources.append(
                InputSourceInfo(
                    name=name,
                    identifier=name,
                    metadata={"url": url} if url else None,
                )
            )

        logger.info(f"Found {len(sources)} NDI source(s)")
        return sources

    def connect(self, identifier: str) -> bool:
        """Connect to an NDI source by name."""
        if self._recv_instance:
            self._lib.NDIlib_recv_destroy(self._recv_instance)
            self._recv_instance = None

        ndi_source = NDIlib_source_t()
        ndi_source.p_ndi_name = identifier.encode("utf-8")
        ndi_source.p_url_address = None

        # Try to resolve the URL from discovered sources
        if self._find_instance:
            num_sources = ctypes.c_uint32(0)
            sources_ptr = self._lib.NDIlib_find_get_current_sources(
                self._find_instance, ctypes.byref(num_sources)
            )
            for i in range(num_sources.value):
                src = sources_ptr[i]
                src_name = src.p_ndi_name.decode("utf-8") if src.p_ndi_name else ""
                if src_name == identifier:
                    ndi_source.p_ndi_name = src.p_ndi_name
                    ndi_source.p_url_address = src.p_url_address
                    break

        recv_create = NDIlib_recv_create_v3_t()
        recv_create.source_to_connect_to = ndi_source
        recv_create.color_format = NDI_COLOR_FORMAT_RGBX_RGBA
        recv_create.bandwidth = NDI_BANDWIDTH_HIGHEST
        recv_create.allow_video_fields = False
        recv_create.p_ndi_recv_name = b"Scope"

        self._recv_instance = self._lib.NDIlib_recv_create_v3(ctypes.byref(recv_create))
        if not self._recv_instance:
            logger.error(f"Failed to create NDI receiver for '{identifier}'")
            return False

        self._connected_source_name = identifier
        logger.info(f"NDI connected to '{identifier}'")
        return True

    def receive_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        """Receive a video frame. Returns (H, W, 3) RGB uint8 or None."""
        if not self._recv_instance:
            return None

        video_frame = NDIlib_video_frame_v2_t()
        audio_frame = NDIlib_audio_frame_v2_t()
        metadata_frame = NDIlib_metadata_frame_t()

        frame_type = self._lib.NDIlib_recv_capture_v2(
            self._recv_instance,
            ctypes.byref(video_frame),
            ctypes.byref(audio_frame),
            ctypes.byref(metadata_frame),
            timeout_ms,
        )

        if frame_type != NDI_FRAME_TYPE_VIDEO:
            return None

        try:
            width = video_frame.xres
            height = video_frame.yres
            stride = video_frame.line_stride_in_bytes
            fourcc = video_frame.FourCC

            if fourcc in (
                NDI_FOURCC_RGBA,
                NDI_FOURCC_RGBX,
                NDI_FOURCC_BGRA,
                NDI_FOURCC_BGRX,
            ):
                bpp = 4
            else:
                bpp = 2

            if stride == width * bpp:
                buffer_size = height * stride
                buffer = (ctypes.c_uint8 * buffer_size).from_address(video_frame.p_data)
                frame_data = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    (height, width, bpp if bpp == 4 else -1)
                )
            else:
                frame_data = np.zeros((height, width, 4), dtype=np.uint8)
                for y in range(height):
                    row_start = video_frame.p_data + y * stride
                    row_buffer = (ctypes.c_uint8 * (width * 4)).from_address(row_start)
                    frame_data[y] = np.frombuffer(row_buffer, dtype=np.uint8).reshape(
                        (width, 4)
                    )

            if fourcc in (NDI_FOURCC_BGRA, NDI_FOURCC_BGRX):
                frame_data = frame_data[:, :, [2, 1, 0, 3]]

            rgb_frame = frame_data[:, :, :3].copy()
            return rgb_frame

        finally:
            self._lib.NDIlib_recv_free_video_v2(
                self._recv_instance, ctypes.byref(video_frame)
            )

    def get_source_resolution(
        self, identifier: str, timeout_ms: int = 5000
    ) -> tuple[int, int] | None:
        """Probe an NDI source's native resolution by receiving one frame."""
        was_connected = self._recv_instance is not None
        prev_source = self._connected_source_name

        try:
            if not self.connect(identifier):
                return None

            elapsed = 0
            poll_interval = 100
            while elapsed < timeout_ms:
                video_frame = NDIlib_video_frame_v2_t()
                audio_frame = NDIlib_audio_frame_v2_t()
                metadata_frame = NDIlib_metadata_frame_t()

                frame_type = self._lib.NDIlib_recv_capture_v2(
                    self._recv_instance,
                    ctypes.byref(video_frame),
                    ctypes.byref(audio_frame),
                    ctypes.byref(metadata_frame),
                    poll_interval,
                )

                if frame_type == NDI_FRAME_TYPE_VIDEO:
                    width = video_frame.xres
                    height = video_frame.yres
                    self._lib.NDIlib_recv_free_video_v2(
                        self._recv_instance, ctypes.byref(video_frame)
                    )
                    return (width, height)

                elapsed += poll_interval

            logger.warning(
                f"Timed out probing resolution for '{identifier}' after {timeout_ms}ms"
            )
            return None
        finally:
            self.disconnect()
            if was_connected and prev_source:
                self.connect(prev_source)

    def disconnect(self):
        """Disconnect from the current NDI source."""
        if self._recv_instance:
            self._lib.NDIlib_recv_destroy(self._recv_instance)
            self._recv_instance = None
        self._connected_source_name = None

    def close(self):
        """Clean up all NDI resources."""
        self.disconnect()

        if self._find_instance:
            self._lib.NDIlib_find_destroy(self._find_instance)
            self._find_instance = None

        try:
            self._lib.NDIlib_destroy()
        except Exception as e:
            logger.warning(f"Error destroying NDI library: {e}")
