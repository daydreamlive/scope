"""Contract: no banned retry counter ticked during the test.

The RetryCounter instrumentation (src/scope/server/retry_counter.py)
is gated by ``SCOPE_TEST_INSTRUMENTATION=1`` and tracks per-site retry
and failure events. A test passes only if every banned counter is zero
at teardown.

Counters outside the banned set (e.g. cloud_connect_attempts=1) are
expected during a normal session and do not fail the contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.retry_probe import RetryProbe

BANNED_COUNTERS: tuple[str, ...] = (
    "cloud_connect_failures",
    "cloud_reconnects",
    "frames_dropped_video",
    "frames_dropped_audio",
    "frontend_reconnects",
    "frontend_pc_failed",
    "frontend_offer_failed",
    "frontend_start_stream_failed",
    "unexpected_session_close",
)


class NoRetriesViolation(AssertionError):
    """Raised when at least one banned counter is > 0."""


@dataclass
class NoRetries:
    """Contract — banned retry counters must be zero."""

    probe: RetryProbe

    def check(self) -> dict[str, int]:
        """Return a dict of {counter: value} for any ticked banned counter."""
        counts = self.probe.snapshot()
        return {k: v for k, v in counts.items() if v > 0 and k in BANNED_COUNTERS}

    def assert_clean(self) -> None:
        violations = self.check()
        if violations:
            events = self.probe.events()
            raise NoRetriesViolation(
                f"Banned retry counters ticked: {violations}; events={events}"
            )
