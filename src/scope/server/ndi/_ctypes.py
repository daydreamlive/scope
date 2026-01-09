from __future__ import annotations

import ctypes
import os
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path


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
        ("FourCC", ctypes.c_uint32),
        ("frame_rate_N", ctypes.c_int),
        ("frame_rate_D", ctypes.c_int),
        ("picture_aspect_ratio", ctypes.c_float),
        ("frame_format_type", ctypes.c_int),
        ("timecode", ctypes.c_int64),
        ("p_data", ctypes.POINTER(ctypes.c_uint8)),
        ("line_stride_in_bytes", ctypes.c_int),
        ("p_metadata", ctypes.c_char_p),
        ("timestamp", ctypes.c_int64),
    ]


@dataclass(frozen=True)
class NDILib:
    lib: ctypes.CDLL


def _try_load_libndi(path: str) -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


def _cyndilib_bundled_libndi_path() -> Path | None:
    try:
        import cyndilib  # type: ignore[import-not-found]
    except Exception:
        return None

    pkg_dir = Path(cyndilib.__file__).resolve().parent
    candidate = pkg_dir / "wrapper" / "bin" / "x86_64-linux-gnu" / "libndi.so"
    if candidate.exists():
        return candidate
    return None


def load_libndi() -> NDILib:
    env_path = os.environ.get("SCOPE_NDI_LIB_PATH")
    if env_path:
        lib = _try_load_libndi(env_path)
        if lib is None:
            raise RuntimeError(f"Failed to load NDI library at SCOPE_NDI_LIB_PATH={env_path!r}")
        return NDILib(lib=lib)

    # Prefer system install if present.
    for candidate in (find_library("ndi"), "libndi.so.6", "libndi.so"):
        if not candidate:
            continue
        lib = _try_load_libndi(candidate)
        if lib is not None:
            return NDILib(lib=lib)

    # Fallback: cyndilib ships a bundled libndi.so (useful in dev containers).
    bundled = _cyndilib_bundled_libndi_path()
    if bundled is not None:
        lib = _try_load_libndi(str(bundled))
        if lib is not None:
            return NDILib(lib=lib)

    raise RuntimeError(
        "NDI runtime not found. Install the NDI SDK/runtime (libndi.so.6), "
        "or set SCOPE_NDI_LIB_PATH, or install cyndilib (dev fallback)."
    )


def configure_libndi_prototypes(ndi: NDILib) -> None:
    lib = ndi.lib

    # init
    lib.NDIlib_initialize.restype = ctypes.c_bool
    lib.NDIlib_destroy.argtypes = []
    lib.NDIlib_version.restype = ctypes.c_char_p

    # find
    lib.NDIlib_find_create_v2.argtypes = [ctypes.POINTER(NDIlib_find_create_t)]
    lib.NDIlib_find_create_v2.restype = ctypes.c_void_p
    lib.NDIlib_find_destroy.argtypes = [ctypes.c_void_p]
    lib.NDIlib_find_wait_for_sources.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.NDIlib_find_wait_for_sources.restype = ctypes.c_bool
    lib.NDIlib_find_get_current_sources.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.NDIlib_find_get_current_sources.restype = ctypes.POINTER(NDIlib_source_t)

    # recv
    lib.NDIlib_recv_create_v3.argtypes = [ctypes.POINTER(NDIlib_recv_create_v3_t)]
    lib.NDIlib_recv_create_v3.restype = ctypes.c_void_p
    lib.NDIlib_recv_destroy.argtypes = [ctypes.c_void_p]
    lib.NDIlib_recv_connect.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_source_t)]
    lib.NDIlib_recv_capture_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NDIlib_video_frame_v2_t),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.NDIlib_recv_capture_v2.restype = ctypes.c_int
    lib.NDIlib_recv_free_video_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NDIlib_video_frame_v2_t),
    ]
    lib.NDIlib_recv_get_no_connections.argtypes = [ctypes.c_void_p]
    lib.NDIlib_recv_get_no_connections.restype = ctypes.c_int

    # send
    lib.NDIlib_send_create.argtypes = [ctypes.POINTER(NDIlib_send_create_t)]
    lib.NDIlib_send_create.restype = ctypes.c_void_p
    lib.NDIlib_send_destroy.argtypes = [ctypes.c_void_p]

    lib.NDIlib_send_send_video_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NDIlib_video_frame_v2_t),
    ]
    lib.NDIlib_send_send_video_v2.restype = None

    # Send-side get_no_connections takes a timeout_in_ms parameter.
    lib.NDIlib_send_get_no_connections.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.NDIlib_send_get_no_connections.restype = ctypes.c_int


# Enums (subset) from Processing.NDI.structs.h / Processing.NDI.Recv.h
NDIlib_frame_type_none = 0
NDIlib_frame_type_video = 1
NDIlib_frame_type_error = 4
NDIlib_frame_type_status_change = 100

NDIlib_recv_color_format_BGRX_BGRA = 0
NDIlib_recv_bandwidth_highest = 100

# Frame format type enum values (subset).
NDIlib_frame_format_type_interleaved = 0
NDIlib_frame_format_type_progressive = 1
NDIlib_frame_format_type_field_0 = 2
NDIlib_frame_format_type_field_1 = 3


def ndi_fourcc(a: str, b: str, c: str, d: str) -> int:
    return (ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24))


NDIlib_FourCC_type_BGRA = ndi_fourcc("B", "G", "R", "A")
NDIlib_FourCC_type_BGRX = ndi_fourcc("B", "G", "R", "X")

# When sending, NDI recommends timecode synthesis (INT64_MAX).
NDIlib_send_timecode_synthesize = 9223372036854775807
