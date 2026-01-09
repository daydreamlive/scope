from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._ctypes import (
    NDIlib_recv_bandwidth_highest,
    NDIlib_recv_color_format_BGRX_BGRA,
    NDIlib_recv_create_v3_t,
    NDIlib_source_t,
    NDIlib_video_frame_v2_t,
    NDIlib_frame_type_error,
    NDIlib_frame_type_none,
    NDIlib_frame_type_status_change,
    NDIlib_frame_type_video,
)
from .finder import NDISource, list_sources
from .runtime import get_runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NDIReceiverStats:
    frames_received: int = 0
    frames_dropped_during_drain: int = 0
    last_frame_ts_s: float = 0.0


class NDIReceiver:
    """Thin NDI receiver wrapper (ctypes over libndi)."""

    def __init__(
        self,
        *,
        recv_name: str = "ScopeNDIRecv",
        color_format: int = NDIlib_recv_color_format_BGRX_BGRA,
        bandwidth: int = NDIlib_recv_bandwidth_highest,
        allow_video_fields: bool = True,
    ) -> None:
        self._recv_name = recv_name
        self._color_format = int(color_format)
        self._bandwidth = int(bandwidth)
        self._allow_video_fields = bool(allow_video_fields)

        self._runtime = get_runtime()
        self._ndi = None
        self._recv = None
        self._connected_url: bytes | None = None
        self._last_source: NDISource | None = None

        self._stats = NDIReceiverStats()

    def create(self) -> bool:
        if self._recv is not None:
            return True

        self._ndi = self._runtime.acquire()
        create = NDIlib_recv_create_v3_t()
        create.source_to_connect_to = NDIlib_source_t(None, None)
        create.color_format = self._color_format
        create.bandwidth = self._bandwidth
        create.allow_video_fields = self._allow_video_fields
        create.p_ndi_recv_name = self._recv_name.encode("utf-8")

        recv = self._ndi.lib.NDIlib_recv_create_v3(ctypes.byref(create))
        if not recv:
            self._runtime.release()
            self._ndi = None
            return False

        self._recv = recv
        return True

    def get_no_connections(self) -> int:
        if self._recv is None or self._ndi is None:
            return 0
        return int(self._ndi.lib.NDIlib_recv_get_no_connections(self._recv))

    def connect(self, *, url_address: str, source_name: str | None = None) -> None:
        if self._recv is None or self._ndi is None:
            raise RuntimeError("Receiver not created")

        url_bytes = url_address.encode("utf-8")
        self._connected_url = url_bytes
        self._last_source = NDISource(name=source_name or "", url_address=url_address)

        src = NDIlib_source_t(None, url_bytes)
        self._ndi.lib.NDIlib_recv_connect(self._recv, ctypes.byref(src))

    def connect_discovered(
        self,
        *,
        source_substring: str,
        extra_ips: Sequence[str] | None = None,
        timeout_ms: int = 1500,
    ) -> NDISource:
        sources = list_sources(
            timeout_ms=timeout_ms,
            extra_ips=extra_ips,
            show_local_sources=True,
        )
        if not sources:
            raise RuntimeError("No NDI sources discovered")

        needle = (source_substring or "").strip().lower()
        chosen: NDISource | None = None
        if not needle:
            chosen = sources[0]
        else:
            for s in sources:
                if needle in s.name.lower():
                    chosen = s
                    break

        if chosen is None:
            raise RuntimeError(
                f"No NDI source matched {source_substring!r}. Discovered: {[s.name for s in sources]}"
            )

        if not chosen.url_address:
            raise RuntimeError(f"NDI source {chosen.name!r} has no url_address; cannot connect reliably")

        self.connect(url_address=chosen.url_address, source_name=chosen.name)
        return chosen

    def receive_latest_rgb24(self, *, timeout_ms: int = 5) -> np.ndarray | None:
        if self._recv is None or self._ndi is None:
            return None

        # Drain-to-latest: keep only the newest available video frame.
        dropped = 0
        last_vf: NDIlib_video_frame_v2_t | None = None

        def free_if_needed(vf: NDIlib_video_frame_v2_t | None) -> None:
            if vf is None:
                return
            self._ndi.lib.NDIlib_recv_free_video_v2(self._recv, ctypes.byref(vf))

        try:
            # Block briefly for the first frame.
            first = NDIlib_video_frame_v2_t()
            ft = int(self._ndi.lib.NDIlib_recv_capture_v2(self._recv, ctypes.byref(first), None, None, int(timeout_ms)))
            if ft == NDIlib_frame_type_video:
                last_vf = first
            elif ft in (NDIlib_frame_type_none, NDIlib_frame_type_status_change):
                return None
            elif ft == NDIlib_frame_type_error:
                raise RuntimeError("NDI receiver capture error (connection lost?)")
            else:
                return None

            # Drain any queued frames (timeout=0).
            while True:
                nxt = NDIlib_video_frame_v2_t()
                ft = int(self._ndi.lib.NDIlib_recv_capture_v2(self._recv, ctypes.byref(nxt), None, None, 0))
                if ft != NDIlib_frame_type_video:
                    break
                free_if_needed(last_vf)
                dropped += 1
                last_vf = nxt

            if last_vf is None:
                return None

            h, w = int(last_vf.yres), int(last_vf.xres)
            if h <= 0 or w <= 0:
                return None

            stride = int(last_vf.line_stride_in_bytes) or (w * 4)
            raw = ctypes.string_at(last_vf.p_data, stride * h)

            # Copy/reshape with stride, then drop padding and convert BGRX->RGB.
            rows = np.frombuffer(raw, dtype=np.uint8).reshape((h, stride))
            bgrx = rows[:, : w * 4].reshape((h, w, 4))
            rgb = bgrx[:, :, 2::-1].copy()

            self._stats = NDIReceiverStats(
                frames_received=self._stats.frames_received + 1,
                frames_dropped_during_drain=self._stats.frames_dropped_during_drain + dropped,
                last_frame_ts_s=time.monotonic(),
            )
            return rgb
        finally:
            free_if_needed(last_vf)

    def get_stats(self) -> NDIReceiverStats:
        return self._stats

    def release(self) -> None:
        if self._recv is None:
            return
        try:
            if self._ndi is not None:
                self._ndi.lib.NDIlib_recv_destroy(self._recv)
        finally:
            self._recv = None
            self._ndi = None
            self._connected_url = None
            self._last_source = None
            self._runtime.release()

