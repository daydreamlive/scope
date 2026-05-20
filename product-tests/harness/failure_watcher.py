"""FailureWatcher — tails Scope logs and records test-initiated stop windows.

The invariant: every ``session_closed`` or fatal error observed during a test
that was NOT preceded by a test-initiated stop is an unexpected close — a
product-level failure.

Usage:

    with FailureWatcher(log_path) as watcher:
        ...drive the UI...
        watcher.mark_initiated_stop()   # test is about to click Stop
        ...drive the UI...
    # at teardown:
    assert watcher.unexpected_closes == 0

The watcher runs a background thread that greps the log file for known
failure patterns. It also timestamps test-initiated stops so a ``session_closed``
within the grace window is attributed to the test.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_FAILURE_PATTERNS = [
    re.compile(r"session_closed", re.IGNORECASE),
    re.compile(r"unexpected disconnect", re.IGNORECASE),
    re.compile(r"Failed to connect job", re.IGNORECASE),
    re.compile(r"forcibly closed", re.IGNORECASE),
    re.compile(r"CRITICAL", re.IGNORECASE),
]
_STOP_INITIATED_GRACE_SEC = 3.0


@dataclass
class FailureEvent:
    timestamp: float
    pattern: str
    line: str


@dataclass
class FailureWatcher:
    log_path: Path
    poll_interval: float = 0.25

    _thread: threading.Thread | None = field(default=None, init=False)
    _stop_flag: threading.Event = field(default_factory=threading.Event, init=False)
    _initiated_stops: list[float] = field(default_factory=list, init=False)
    _events: list[FailureEvent] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self) -> None:
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def mark_initiated_stop(self) -> None:
        """Record that the test is about to trigger a Scope-side stop."""
        with self._lock:
            self._initiated_stops.append(time.time())

    @property
    def unexpected_closes(self) -> int:
        """Count of session_closed / failure events not preceded by a test stop."""
        return sum(1 for e in self._events if not self._was_initiated(e.timestamp))

    @property
    def events(self) -> list[FailureEvent]:
        with self._lock:
            return list(self._events)

    def _was_initiated(self, event_ts: float) -> bool:
        with self._lock:
            return any(
                abs(event_ts - stop_ts) <= _STOP_INITIATED_GRACE_SEC
                for stop_ts in self._initiated_stops
            )

    def _run(self) -> None:
        # Wait for the log file to appear, then tail it.
        while not self._stop_flag.is_set() and not self.log_path.exists():
            time.sleep(self.poll_interval)

        if self._stop_flag.is_set():
            return

        try:
            with open(self.log_path, errors="replace") as fh:
                # Start at EOF so we don't re-scan boot output.
                fh.seek(0, 2)
                while not self._stop_flag.is_set():
                    line = fh.readline()
                    if not line:
                        time.sleep(self.poll_interval)
                        continue
                    for pat in _FAILURE_PATTERNS:
                        if pat.search(line):
                            with self._lock:
                                self._events.append(
                                    FailureEvent(
                                        timestamp=time.time(),
                                        pattern=pat.pattern,
                                        line=line.rstrip("\n"),
                                    )
                                )
                            break
        except FileNotFoundError:
            return
