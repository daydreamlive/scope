"""Product-test instrumentation: retry/failure counters.

Gated behind ``SCOPE_TEST_INSTRUMENTATION=1``. In production (unset) every
call is a cheap no-op — the counter never allocates and the HTTP endpoint
refuses to register.

The purpose is to let product-level tests enforce "zero retries" as a hard
failure. A scenario that connects to cloud after one internal retry is a
pass-looking regression; instrumenting the retry site and asserting the
counter is zero at teardown turns that into a loud failure.

Counter names are free-form; callers agree on conventions:
  cloud_connect_attempts   — each call to LivepeerConnection.connect()
  cloud_connect_failures   — exceptions raised from connect()
  cloud_reconnects         — connect_background cancelling an in-flight task
  frames_dropped_video     — CloudRelay video queue full
  frames_dropped_audio     — CloudRelay audio queue full
  frontend_reconnects      — FE reported reconnect via POST
  unexpected_session_close — session_closed event not preceded by user stop

The test harness owns interpretation (e.g. which counters must be zero for
which scenarios). This module just counts.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict

_ENV_VAR = "SCOPE_TEST_INSTRUMENTATION"


def _is_enabled() -> bool:
    return os.environ.get(_ENV_VAR) == "1"


class RetryCounter:
    """Thread-safe counter registry. No-op when instrumentation is disabled."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._events: list[dict] = []

    def incr(self, name: str, *, by: int = 1, **context) -> None:
        """Increment counter ``name`` by ``by``. No-op unless enabled."""
        if not _is_enabled():
            return
        with self._lock:
            self._counts[name] += by
            if context:
                self._events.append({"name": name, "by": by, **context})

    def snapshot(self) -> dict[str, int]:
        """Return a copy of current counts. Empty dict when disabled."""
        if not _is_enabled():
            return {}
        with self._lock:
            return dict(self._counts)

    def events(self) -> list[dict]:
        """Return a copy of recorded events with context."""
        if not _is_enabled():
            return []
        with self._lock:
            return list(self._events)

    def reset(self) -> None:
        """Zero all counters. Used by tests between phases."""
        with self._lock:
            self._counts.clear()
            self._events.clear()


retry_counter = RetryCounter()


def is_enabled() -> bool:
    """Whether instrumentation is active. Check before wiring expensive probes."""
    return _is_enabled()
