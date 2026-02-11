"""NDI input source implementation using ctypes.

Interfaces with the NDI SDK library directly via ctypes.
The user must install the NDI SDK/Tools on their system for this to work.
No additional Python packages are required.

NDI SDK downloads: https://ndi.video/tools/
"""

import ctypes
import ctypes.util
import logging
import os
import platform
from typing import ClassVar

import numpy as np

from .interface import InputSource, InputSourceInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NDI ctypes structures
# ---------------------------------------------------------------------------


class _NDIlib_source_t(ctypes.Structure):
    _fields_ = [
        ("p_ndi_name", ctypes.c_char_p),
        ("p_url_address", ctypes.c_char_p),
    ]


class _NDIlib_find_create_t(ctypes.Structure):
    _fields_ = [
        ("show_local_sources", ctypes.c_bool),
        ("p_groups", ctypes.c_char_p),
        ("p_extra_ips", ctypes.c_char_p),
    ]


class _NDIlib_recv_create_v3_t(ctypes.Structure):
    _fields_ = [
        ("source_to_connect_to", _NDIlib_source_t),
        ("color_format", ctypes.c_int),
        ("bandwidth", ctypes.c_int),
        ("allow_video_fields", ctypes.c_bool),
        ("p_ndi_recv_name", ctypes.c_char_p),
    ]


class _NDIlib_video_frame_v2_t(ctypes.Structure):
    _fields_ = [
        ("xres", ctypes.c_int),
        ("yres", ctypes.c_int),
        ("FourCC", ctypes.c_int),
        ("frame_rate_N", ctypes.c_int),
        ("frame_rate_D", ctypes.c_int),
        ("picture_aspect_ratio", ctypes.c_float),
        ("frame_format_type", ctypes.c_int),
        ("timecode", ctypes.c_int64),
        ("p_data", ctypes.c_void_p),
        ("line_stride_in_bytes", ctypes.c_int),
        ("p_metadata", ctypes.c_char_p),
        ("timestamp", ctypes.c_int64),
    ]


class _NDIlib_audio_frame_v2_t(ctypes.Structure):
    _fields_ = [
        ("sample_rate", ctypes.c_int),
        ("no_channels", ctypes.c_int),
        ("no_samples", ctypes.c_int),
        ("timecode", ctypes.c_int64),
        ("p_data", ctypes.c_void_p),
        ("channel_stride_in_bytes", ctypes.c_int),
        ("p_metadata", ctypes.c_char_p),
        ("timestamp", ctypes.c_int64),
    ]


class _NDIlib_metadata_frame_t(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_int),
        ("timecode", ctypes.c_int64),
        ("p_data", ctypes.c_char_p),
    ]

# Color format constants
_NDI_COLOR_FORMAT_BGRX_BGRA = 0
_NDI_COLOR_FORMAT_UYVY_BGRA = 1
_NDI_COLOR_FORMAT_RGBX_RGBA = 2
_NDI_COLOR_FORMAT_UYVY_RGBA = 3
_NDI_COLOR_FORMAT_FASTEST = 100
_NDI_COLOR_FORMAT_BEST = 101

# Bandwidth constants
_NDI_BANDWIDTH_METADATA_ONLY = -10
_NDI_BANDWIDTH_AUDIO_ONLY = 10
_NDI_BANDWIDTH_LOWEST = 0
_NDI_BANDWIDTH_HIGHEST = 100

# Frame type constants
_NDI_FRAME_TYPE_NONE = 0
_NDI_FRAME_TYPE_VIDEO = 1
_NDI_FRAME_TYPE_AUDIO = 2
_NDI_FRAME_TYPE_METADATA = 3
_NDI_FRAME_TYPE_ERROR = 4
_NDI_FRAME_TYPE_STATUS_CHANGE = 100

# FourCC constants (as integers)
_NDI_FOURCC_UYVY = 0x59565955
_NDI_FOURCC_BGRA = 0x41524742
_NDI_FOURCC_BGRX = 0x58524742
_NDI_FOURCC_RGBA = 0x41424752
_NDI_FOURCC_RGBX = 0x58424752

# Module-level cache for NDI library availability
_ndi_available: bool | None = None
_ndi_lib: ctypes.CDLL | None = None


def _get_ndi_library_paths() -> list[str | None]:
    """Get platform-specific paths to search for the NDI library."""
    system = platform.system()

    if system == "Darwin":
        return [
            # NDI 6.x SDK
            "/Library/NDI SDK for Apple/lib/macOS/libndi.dylib",
            # NDI 5.x SDK
            "/Library/NDI SDK for Apple/lib/x64/libndi.5.dylib",
            "/Library/NDI SDK for Apple/lib/x64/libndi.dylib",
            # Alternative locations
            "/usr/local/lib/libndi.dylib",
            "/usr/local/lib/libndi.5.dylib",
            ctypes.util.find_library("ndi"),
        ]
    elif system == "Windows":
        paths: list[str | None] = [
            "Processing.NDI.Lib.x64.dll",
            ctypes.util.find_library("Processing.NDI.Lib.x64"),
        ]
        # Check NDI runtime environment variables (set by NDI Tools installer)
        for var in [
            "NDI_RUNTIME_DIR_V6",
            "NDI_RUNTIME_DIR_V5",
            "NDI_RUNTIME_DIR_V4",
            "NDI_RUNTIME_DIR_V3",
        ]:
            env_path = os.environ.get(var)
            if env_path:
                paths.append(os.path.join(env_path, "Processing.NDI.Lib.x64.dll"))
        return paths
    else:
        # Linux
        return [
            "/usr/lib/libndi.so",
            "/usr/lib/x86_64-linux-gnu/libndi.so",
            "/usr/local/lib/libndi.so",
            ctypes.util.find_library("ndi"),
        ]


def _load_ndi_library() -> ctypes.CDLL:
    """Load the NDI runtime library.

    Raises:
        RuntimeError: If the NDI library cannot be found.
    """
    for path in _get_ndi_library_paths():
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue

    raise RuntimeError(
        "NDI library not found. Please install NDI Tools from https://ndi.video/tools/\n"
        "On macOS, install 'NDI SDK for Apple' from the NDI website.\n"
        "On Windows, install 'NDI Tools' which includes the runtime.\n"
        "On Linux, install the NDI SDK and ensure libndi.so is on the library path."
    )


def _try_load_ndi() -> tuple[bool, ctypes.CDLL | None]:
    """Try to load the NDI library and return (available, lib)."""
    global _ndi_available, _ndi_lib
    if _ndi_available is not None:
        return _ndi_available, _ndi_lib

    try:
        lib = _load_ndi_library()
        _ndi_lib = lib
        _ndi_available = True
        return True, lib
    except RuntimeError:
        _ndi_available = False
        _ndi_lib = None
        return False, None


def _setup_ndi_functions(lib: ctypes.CDLL) -> None:
    """Set up ctypes function signatures for the NDI library."""
    lib.NDIlib_initialize.restype = ctypes.c_bool
    lib.NDIlib_initialize.argtypes = []

    lib.NDIlib_destroy.restype = None
    lib.NDIlib_destroy.argtypes = []

    lib.NDIlib_find_create_v2.restype = ctypes.c_void_p
    lib.NDIlib_find_create_v2.argtypes = [ctypes.POINTER(_NDIlib_find_create_t)]

    lib.NDIlib_find_destroy.restype = None
    lib.NDIlib_find_destroy.argtypes = [ctypes.c_void_p]

    lib.NDIlib_find_wait_for_sources.restype = ctypes.c_bool
    lib.NDIlib_find_wait_for_sources.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    lib.NDIlib_find_get_current_sources.restype = ctypes.POINTER(_NDIlib_source_t)
    lib.NDIlib_find_get_current_sources.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]

    lib.NDIlib_recv_create_v3.restype = ctypes.c_void_p
    lib.NDIlib_recv_create_v3.argtypes = [ctypes.POINTER(_NDIlib_recv_create_v3_t)]

    lib.NDIlib_recv_destroy.restype = None
    lib.NDIlib_recv_destroy.argtypes = [ctypes.c_void_p]

    lib.NDIlib_recv_capture_v2.restype = ctypes.c_int
    lib.NDIlib_recv_capture_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NDIlib_video_frame_v2_t),
        ctypes.POINTER(_NDIlib_audio_frame_v2_t),
        ctypes.POINTER(_NDIlib_metadata_frame_t),
        ctypes.c_uint32,
    ]

    lib.NDIlib_recv_free_video_v2.restype = None
    lib.NDIlib_recv_free_video_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NDIlib_video_frame_v2_t),
    ]

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
        available, lib = _try_load_ndi()
        if not available or lib is None:
            raise RuntimeError(
                "NDI SDK is not available. Install NDI Tools from https://ndi.video/tools/"
            )

        self._lib = lib
        _setup_ndi_functions(self._lib)

        if not self._lib.NDIlib_initialize():
            raise RuntimeError("Failed to initialize NDI library")

        self._find_instance = None
        self._recv_instance = None
        self._connected_source_name: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if the NDI SDK is installed on this system."""
        available, _ = _try_load_ndi()
        return available

    def list_sources(self, timeout_ms: int = 5000) -> list[InputSourceInfo]:
        """List available NDI sources on the network."""
        if self._find_instance is None:
            create_settings = _NDIlib_find_create_t()
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
            url = (
                source.p_url_address.decode("utf-8") if source.p_url_address else ""
            )
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

        ndi_source = _NDIlib_source_t()
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

        recv_create = _NDIlib_recv_create_v3_t()
        recv_create.source_to_connect_to = ndi_source
        recv_create.color_format = _NDI_COLOR_FORMAT_RGBX_RGBA
        recv_create.bandwidth = _NDI_BANDWIDTH_HIGHEST
        recv_create.allow_video_fields = False
        recv_create.p_ndi_recv_name = b"Scope"

        self._recv_instance = self._lib.NDIlib_recv_create_v3(
            ctypes.byref(recv_create)
        )
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

        video_frame = _NDIlib_video_frame_v2_t()
        audio_frame = _NDIlib_audio_frame_v2_t()
        metadata_frame = _NDIlib_metadata_frame_t()

        frame_type = self._lib.NDIlib_recv_capture_v2(
            self._recv_instance,
            ctypes.byref(video_frame),
            ctypes.byref(audio_frame),
            ctypes.byref(metadata_frame),
            timeout_ms,
        )

        if frame_type != _NDI_FRAME_TYPE_VIDEO:
            return None

        try:
            width = video_frame.xres
            height = video_frame.yres
            stride = video_frame.line_stride_in_bytes
            fourcc = video_frame.FourCC

            # Calculate bytes per pixel based on FourCC
            if fourcc in (
                _NDI_FOURCC_RGBA,
                _NDI_FOURCC_RGBX,
                _NDI_FOURCC_BGRA,
                _NDI_FOURCC_BGRX,
            ):
                bpp = 4
            else:
                # UYVY or other format
                bpp = 2

            # Copy frame data to numpy array
            if stride == width * bpp:
                # No padding, direct copy
                buffer_size = height * stride
                buffer = (ctypes.c_uint8 * buffer_size).from_address(
                    video_frame.p_data
                )
                frame_data = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    (height, width, bpp if bpp == 4 else -1)
                )
            else:
                # Handle stride/padding (row by row copy)
                frame_data = np.zeros((height, width, 4), dtype=np.uint8)
                for y in range(height):
                    row_start = video_frame.p_data + y * stride
                    row_buffer = (ctypes.c_uint8 * (width * 4)).from_address(row_start)
                    frame_data[y] = np.frombuffer(row_buffer, dtype=np.uint8).reshape(
                        (width, 4)
                    )

            # Convert BGRA to RGBA if needed
            if fourcc in (_NDI_FOURCC_BGRA, _NDI_FOURCC_BGRX):
                frame_data = frame_data[:, :, [2, 1, 0, 3]]

            # Return as RGB (H, W, 3) - strip alpha channel
            rgb_frame = frame_data[:, :, :3].copy()
            return rgb_frame

        finally:
            # Always free the video frame
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

            # Poll for a video frame to read its dimensions
            elapsed = 0
            poll_interval = 100
            while elapsed < timeout_ms:
                video_frame = _NDIlib_video_frame_v2_t()
                audio_frame = _NDIlib_audio_frame_v2_t()
                metadata_frame = _NDIlib_metadata_frame_t()

                frame_type = self._lib.NDIlib_recv_capture_v2(
                    self._recv_instance,
                    ctypes.byref(video_frame),
                    ctypes.byref(audio_frame),
                    ctypes.byref(metadata_frame),
                    poll_interval,
                )

                if frame_type == _NDI_FRAME_TYPE_VIDEO:
                    width = video_frame.xres
                    height = video_frame.yres
                    self._lib.NDIlib_recv_free_video_v2(
                        self._recv_instance, ctypes.byref(video_frame)
                    )
                    return (width, height)

                elapsed += poll_interval

            logger.warning(
                f"Timed out probing resolution for '{identifier}' "
                f"after {timeout_ms}ms"
            )
            return None
        finally:
            # Restore previous connection state
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
