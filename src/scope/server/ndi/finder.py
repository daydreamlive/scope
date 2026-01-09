from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Sequence

from ._ctypes import NDIlib_find_create_t, NDIlib_source_t
from .runtime import get_runtime


@dataclass(frozen=True)
class NDISource:
    name: str
    url_address: str | None


def list_sources(
    *,
    timeout_ms: int = 1000,
    extra_ips: Sequence[str] | None = None,
    show_local_sources: bool = True,
) -> list[NDISource]:
    """Discover sources using NDI's find API.

    Notes:
    - `extra_ips` should be a list of IPs (comma-separated in the NDI API) for cross-subnet/VPN discovery.
    - Strings returned by NDI are copied into Python strings before the finder is destroyed.
    """
    runtime = get_runtime()
    ndi = runtime.acquire()
    try:
        extra_ips_str = ",".join([ip.strip() for ip in (extra_ips or []) if ip.strip()])
        settings = NDIlib_find_create_t()
        settings.show_local_sources = bool(show_local_sources)
        settings.p_groups = None
        settings.p_extra_ips = extra_ips_str.encode("utf-8") if extra_ips_str else None

        finder = ndi.lib.NDIlib_find_create_v2(ctypes.byref(settings))
        if not finder:
            raise RuntimeError("NDIlib_find_create_v2() failed")
        try:
            ndi.lib.NDIlib_find_wait_for_sources(finder, int(timeout_ms))

            count = ctypes.c_uint32(0)
            sources_ptr = ndi.lib.NDIlib_find_get_current_sources(finder, ctypes.byref(count))

            sources: list[NDISource] = []
            for i in range(int(count.value)):
                src: NDIlib_source_t = sources_ptr[i]
                if not src.p_ndi_name:
                    continue
                name = src.p_ndi_name.decode("utf-8", errors="replace")
                url = src.p_url_address.decode("utf-8", errors="replace") if src.p_url_address else None
                sources.append(NDISource(name=name, url_address=url))
            return sources
        finally:
            ndi.lib.NDIlib_find_destroy(finder)
    finally:
        runtime.release()

