"""NDI SDK ctypes bindings: library loading, structures, and constants.

Shared by both NDIInputSource (core/inputs/ndi) and NDIOutputSink (core/outputs/ndi).
The user must install the NDI SDK/Tools for this to work.
NDI SDK downloads: https://ndi.video/tools/
"""

import ctypes
import ctypes.util
import logging
import os
import platform

logger = logging.getLogger(__name__)


class NDIlib_source_t(ctypes.Structure):
    _fields_ = [
        ("p_ndi_name", ctypes.c_char_p),
        ("p_url_address", ctypes.c_char_p),
    ]


class NDIlib_find_create_t(ctypes.Structure):
    _fields_ = [
        ("show_local_sources", ctypes.c_bool),
        ("p_groups", ctypes.c_char_p),
        ("p_extra_ips", ctypes.c_char_p),
    ]


class NDIlib_recv_create_v3_t(ctypes.Structure):
    _fields_ = [
        ("source_to_connect_to", NDIlib_source_t),
        ("color_format", ctypes.c_int),
        ("bandwidth", ctypes.c_int),
        ("allow_video_fields", ctypes.c_bool),
        ("p_ndi_recv_name", ctypes.c_char_p),
    ]


class NDIlib_send_create_t(ctypes.Structure):
    _fields_ = [
        ("p_ndi_name", ctypes.c_char_p),
        ("p_groups", ctypes.c_char_p),
        ("clock_video", ctypes.c_bool),
        ("clock_audio", ctypes.c_bool),
    ]


class NDIlib_video_frame_v2_t(ctypes.Structure):
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


class NDIlib_audio_frame_v2_t(ctypes.Structure):
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


class NDIlib_metadata_frame_t(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_int),
        ("timecode", ctypes.c_int64),
        ("p_data", ctypes.c_char_p),
    ]


# Color formats
NDI_COLOR_FORMAT_BGRX_BGRA = 0
NDI_COLOR_FORMAT_UYVY_BGRA = 1
NDI_COLOR_FORMAT_RGBX_RGBA = 2
NDI_COLOR_FORMAT_UYVY_RGBA = 3
NDI_COLOR_FORMAT_FASTEST = 100
NDI_COLOR_FORMAT_BEST = 101

# Bandwidth
NDI_BANDWIDTH_METADATA_ONLY = -10
NDI_BANDWIDTH_AUDIO_ONLY = 10
NDI_BANDWIDTH_LOWEST = 0
NDI_BANDWIDTH_HIGHEST = 100

# Frame types
NDI_FRAME_TYPE_NONE = 0
NDI_FRAME_TYPE_VIDEO = 1
NDI_FRAME_TYPE_AUDIO = 2
NDI_FRAME_TYPE_METADATA = 3
NDI_FRAME_TYPE_ERROR = 4
NDI_FRAME_TYPE_STATUS_CHANGE = 100

# FourCC (as integers)
NDI_FOURCC_UYVY = 0x59565955
NDI_FOURCC_BGRA = 0x41524742
NDI_FOURCC_BGRX = 0x58524742
NDI_FOURCC_RGBA = 0x41424752
NDI_FOURCC_RGBX = 0x58424752

_ndi_available: bool | None = None
_ndi_lib: ctypes.CDLL | None = None


def _get_library_paths() -> list[str | None]:
    """Platform-specific paths to search for the NDI library."""
    system = platform.system()

    if system == "Darwin":
        return [
            "/Library/NDI SDK for Apple/lib/macOS/libndi.dylib",
            "/Library/NDI SDK for Apple/lib/x64/libndi.5.dylib",
            "/Library/NDI SDK for Apple/lib/x64/libndi.dylib",
            "/usr/local/lib/libndi.dylib",
            "/usr/local/lib/libndi.5.dylib",
            ctypes.util.find_library("ndi"),
        ]
    elif system == "Windows":
        paths: list[str | None] = [
            "Processing.NDI.Lib.x64.dll",
            ctypes.util.find_library("Processing.NDI.Lib.x64"),
        ]
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
        return [
            "/usr/lib/libndi.so",
            "/usr/lib/x86_64-linux-gnu/libndi.so",
            "/usr/local/lib/libndi.so",
            ctypes.util.find_library("ndi"),
        ]


def load_library() -> ctypes.CDLL:
    """Load the NDI runtime library (cached).

    Raises:
        RuntimeError: If the NDI library cannot be found.
    """
    global _ndi_available, _ndi_lib
    if _ndi_lib is not None:
        return _ndi_lib

    for path in _get_library_paths():
        if path:
            try:
                lib = ctypes.CDLL(path)
                _ndi_lib = lib
                _ndi_available = True
                return lib
            except OSError:
                continue

    _ndi_available = False
    raise RuntimeError(
        "NDI library not found. Please install NDI Tools from https://ndi.video/tools/\n"
        "On macOS, install 'NDI SDK for Apple' from the NDI website.\n"
        "On Windows, install 'NDI Tools' which includes the runtime.\n"
        "On Linux, install the NDI SDK and ensure libndi.so is on the library path."
    )


def is_available() -> bool:
    """Check if the NDI SDK is installed on this system."""
    global _ndi_available
    if _ndi_available is not None:
        return _ndi_available
    try:
        load_library()
        return True
    except RuntimeError:
        return False


_recv_setup_done = False
_send_setup_done = False


def setup_recv_functions(lib: ctypes.CDLL) -> None:
    """Set up ctypes signatures for NDI receiver functions."""
    global _recv_setup_done
    if _recv_setup_done:
        return
    _recv_setup_done = True

    lib.NDIlib_initialize.restype = ctypes.c_bool
    lib.NDIlib_initialize.argtypes = []

    lib.NDIlib_destroy.restype = None
    lib.NDIlib_destroy.argtypes = []

    lib.NDIlib_find_create_v2.restype = ctypes.c_void_p
    lib.NDIlib_find_create_v2.argtypes = [ctypes.POINTER(NDIlib_find_create_t)]

    lib.NDIlib_find_destroy.restype = None
    lib.NDIlib_find_destroy.argtypes = [ctypes.c_void_p]

    lib.NDIlib_find_wait_for_sources.restype = ctypes.c_bool
    lib.NDIlib_find_wait_for_sources.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    lib.NDIlib_find_get_current_sources.restype = ctypes.POINTER(NDIlib_source_t)
    lib.NDIlib_find_get_current_sources.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]

    lib.NDIlib_recv_create_v3.restype = ctypes.c_void_p
    lib.NDIlib_recv_create_v3.argtypes = [ctypes.POINTER(NDIlib_recv_create_v3_t)]

    lib.NDIlib_recv_destroy.restype = None
    lib.NDIlib_recv_destroy.argtypes = [ctypes.c_void_p]

    lib.NDIlib_recv_capture_v2.restype = ctypes.c_int
    lib.NDIlib_recv_capture_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NDIlib_video_frame_v2_t),
        ctypes.POINTER(NDIlib_audio_frame_v2_t),
        ctypes.POINTER(NDIlib_metadata_frame_t),
        ctypes.c_uint32,
    ]

    lib.NDIlib_recv_free_video_v2.restype = None
    lib.NDIlib_recv_free_video_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NDIlib_video_frame_v2_t),
    ]


def setup_send_functions(lib: ctypes.CDLL) -> None:
    """Set up ctypes signatures for NDI sender functions."""
    global _send_setup_done
    if _send_setup_done:
        return
    _send_setup_done = True

    lib.NDIlib_initialize.restype = ctypes.c_bool
    lib.NDIlib_initialize.argtypes = []

    lib.NDIlib_destroy.restype = None
    lib.NDIlib_destroy.argtypes = []

    lib.NDIlib_send_create.restype = ctypes.c_void_p
    lib.NDIlib_send_create.argtypes = [ctypes.POINTER(NDIlib_send_create_t)]

    lib.NDIlib_send_destroy.restype = None
    lib.NDIlib_send_destroy.argtypes = [ctypes.c_void_p]

    lib.NDIlib_send_send_video_v2.restype = None
    lib.NDIlib_send_send_video_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NDIlib_video_frame_v2_t),
    ]
