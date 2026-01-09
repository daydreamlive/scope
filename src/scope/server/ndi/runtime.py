from __future__ import annotations

import logging
import threading

from ._ctypes import NDILib, configure_libndi_prototypes, load_libndi

logger = logging.getLogger(__name__)


class NDIRuntime:
    """Process-global NDI SDK lifecycle (refcounted)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refcount = 0
        self._ndi: NDILib | None = None
        self._initialized = False

    def acquire(self) -> NDILib:
        with self._lock:
            if self._ndi is None:
                self._ndi = load_libndi()
                configure_libndi_prototypes(self._ndi)

            if self._refcount == 0 and not self._initialized:
                ok = bool(self._ndi.lib.NDIlib_initialize())
                if not ok:
                    raise RuntimeError("NDIlib_initialize() failed (unsupported CPU or missing runtime?)")
                self._initialized = True
                try:
                    ver = self._ndi.lib.NDIlib_version()
                    if ver:
                        logger.info("NDI runtime initialized: %s", ver.decode("utf-8", errors="replace"))
                except Exception:
                    pass

            self._refcount += 1
            return self._ndi

    def release(self) -> None:
        with self._lock:
            if self._refcount <= 0:
                return
            self._refcount -= 1
            if self._refcount == 0 and self._initialized and self._ndi is not None:
                try:
                    self._ndi.lib.NDIlib_destroy()
                finally:
                    self._initialized = False


_RUNTIME = NDIRuntime()


def get_runtime() -> NDIRuntime:
    return _RUNTIME

