"""Hard-fail gate helpers.

Every scenario runs the same checklist at teardown. Rather than copying
the list of banned counters everywhere, tests call one helper that
populates the report and flips ``report.fail()`` for any violation.
"""

from __future__ import annotations

from .driver import PlaywrightDriver
from .failure_watcher import FailureWatcher
from .report import TestReport
from .retry_probe import RetryProbe

# Counters that MUST be zero for every product-quality pass.
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


def enforce_zero_retries(report: TestReport, probe: RetryProbe) -> int:
    """Populate retry_count on the report, fail if any banned counter > 0.

    Returns the summed retry count.
    """
    try:
        counts = probe.snapshot()
    except Exception as e:
        report.fail(f"could not reach /_debug/retry_stats: {e}")
        return 0

    banned = {k: v for k, v in counts.items() if v > 0 and k in BANNED_COUNTERS}
    total = sum(banned.values())
    report.measure("retry_count", total)
    if banned:
        events = probe.events()
        report.fail(f"retry counters non-zero: {banned}; events={events}")
    return total


def enforce_zero_unexpected_closes(report: TestReport, watcher: FailureWatcher) -> int:
    n = watcher.unexpected_closes
    report.measure("unexpected_close_count", n)
    if n > 0:
        sample = [e.line for e in watcher.events][:5]
        report.fail(f"unexpected_close_count={n}; sample={sample}")
    return n


def enforce_zero_ui_errors(report: TestReport, driver: PlaywrightDriver) -> int:
    n = driver.error_toast_count()
    report.measure("ui_error_events", n)
    if n > 0:
        report.fail(f"ui_error_events={n}")
    return n


def enforce_all_gates(
    report: TestReport,
    probe: RetryProbe,
    watcher: FailureWatcher,
    driver: PlaywrightDriver,
) -> None:
    """Run every gate; each populates the report independently."""
    enforce_zero_retries(report, probe)
    enforce_zero_unexpected_closes(report, watcher)
    enforce_zero_ui_errors(report, driver)
