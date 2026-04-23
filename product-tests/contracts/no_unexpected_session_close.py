"""Contract: no unexpected session close happened during the test.

``FailureWatcher`` tails the Scope log for ``session_closed`` events and
other failure patterns. Any event emitted outside a 3s grace window of a
test-initiated Stop counts as unexpected — which is the #2 failure mode:
the backend or remote-inference layer forcibly tore down the session.

Tests call ``failure_watcher.mark_initiated_stop()`` immediately before
any UI action that will legitimately close the session so this contract
doesn't trip on user-initiated teardowns.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.failure_watcher import FailureWatcher


class NoUnexpectedSessionCloseViolation(AssertionError):
    """Raised when unexpected_closes > 0."""


@dataclass
class NoUnexpectedSessionClose:
    """Contract — no session close outside a test-initiated Stop."""

    watcher: FailureWatcher

    def count(self) -> int:
        return self.watcher.unexpected_closes

    def assert_clean(self) -> None:
        n = self.count()
        if n > 0:
            sample = [e.line for e in self.watcher.events][:5]
            raise NoUnexpectedSessionCloseViolation(
                f"unexpected_close_count={n}; sample={sample}"
            )
