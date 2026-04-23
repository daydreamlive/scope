"""Chaos — tab is backgrounded / foregrounded repeatedly during a stream.

Browsers aggressively throttle timers, rAF, and sometimes WebRTC media
in hidden tabs. Realistic user: switches to Slack, comes back, switches
to email, comes back. The stream must remain live and no retry counter
may tick.

We fake ``document.hidden`` + ``visibilitychange`` via page script
injection because Playwright doesn't expose a first-class API for real
OS-level tab focus. This is enough to exercise the frontend handlers
that listen for visibility, which is where most breakage lives.
"""

from __future__ import annotations

import time

import pytest
from harness import flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness

# Minimal JS that overrides document.visibilityState + document.hidden and
# dispatches the visibilitychange event, matching how Chrome behaves when
# a tab is backgrounded. We redefine the properties once per page load;
# on reload the override disappears so tests are self-cleaning.
_SET_HIDDEN = """(hidden) => {
    Object.defineProperty(document, 'visibilityState', {
        configurable: true, get: () => hidden ? 'hidden' : 'visible'
    });
    Object.defineProperty(document, 'hidden', {
        configurable: true, get: () => !!hidden
    });
    document.dispatchEvent(new Event('visibilitychange'));
}"""


@pytest.mark.chaos
def test_tab_visibility_churn_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Alternate hidden/visible 10x across 30s; stream must stay alive."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    # Record currentTime before to prove the video actually keeps advancing
    # across visibility changes (i.e. hidden-tab throttling didn't freeze us).
    t_before = driver.page.evaluate(
        """() => document.querySelector('[data-testid="sink-video"]')?.currentTime ?? 0"""
    )

    CYCLES = 10
    for _ in range(CYCLES):
        driver.page.evaluate(_SET_HIDDEN, True)
        time.sleep(1.5)
        driver.page.evaluate(_SET_HIDDEN, False)
        time.sleep(1.5)

    t_after = driver.page.evaluate(
        """() => document.querySelector('[data-testid="sink-video"]')?.currentTime ?? 0"""
    )
    advance = float(t_after) - float(t_before)
    report.measure("video_currenttime_advance_sec", int(advance * 1000) / 1000.0)
    report.measure("visibility_cycles", CYCLES)

    # We cycled over ~30s. Even with heavy throttling we expect measurable
    # advance. If it's flat the media pipeline actually froze.
    if advance < 5.0:
        report.fail(
            f"video.currentTime advanced only {advance:.2f}s across {CYCLES} "
            "visibility cycles — media pipeline likely froze while hidden"
        )

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
